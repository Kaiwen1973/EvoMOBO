from __future__ import annotations

import abc
import logging
from typing import Any, Optional, Union

import numpy as np
import torch
from scipy.stats import spearmanr

from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.memoize import clear_cache_hook

from evomobo import transforms as gfp_transforms
from evomobo.models.base_surrogate import BaseSurrogate
from evomobo.models.metrics import quantile_calibration

from .gp_utils import fit_gp_surrogate

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]


class BaseGPSurrogate(BaseSurrogate, abc.ABC):
    def __init__(
        self,
        max_shift: int,
        mask_size: int,
        gp_lr: float,
        enc_lr: float,
        bs: int,
        num_epochs: int,
        holdout_ratio: float,
        early_stopping: bool = True, 
        patience: int = 50,
        eval_period: int = 1,
        tokenizer: Any = None,
        encoder: torch.nn.Module = None,
        pretrain_lr: Optional[float] = None,
        encoder_wd: float = 0.0,
        bootstrap_ratio: Optional[float] = None,
        min_num_train: int = 128,
        task_noise_init: float = 0.01,
        lengthscale_init: float = 0.7,  
        *args,
        **kwargs,
    ):
        self.gp_lr = gp_lr
        self.enc_lr = enc_lr
        self.pretrain_lr = pretrain_lr
        self.bs = int(bs)
        self.encoder_wd = float(encoder_wd)

        self.num_epochs = int(num_epochs)
        self.holdout_ratio = float(holdout_ratio)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.eval_period = int(eval_period)

        self.bootstrap_ratio = bootstrap_ratio
        self.min_num_train = int(min_num_train)
        self.task_noise_init = float(task_noise_init)
        self.lengthscale_init = float(lengthscale_init) 

        self.tokenizer = tokenizer

        self._set_transforms(tokenizer, max_shift, mask_size)
        self.dtype = torch.float32

    def get_features(self, seq_array, batch_size: Optional[int] = None, transform: bool = True) -> torch.Tensor:
        if transform:
            if not isinstance(seq_array, np.ndarray):
                seq_array = np.asarray(seq_array, dtype=object)

            original_shape = seq_array.shape
            flat_seq_array = seq_array.reshape(-1)

            if self.training:
                token_tensors = [self.train_transform(seq) for seq in flat_seq_array]
            else:
                token_tensors = [self.test_transform(seq) for seq in flat_seq_array]

            enc_seq_array = gfp_transforms.padding_collate_fn(
                token_tensors, self.tokenizer.padding_idx
            )  
        else:
            original_shape = seq_array.shape[:-1]
            enc_seq_array = seq_array.reshape(-1, seq_array.shape[-1])

        enc_seq_array = enc_seq_array.to(self.device)

        if batch_size is None or enc_seq_array.size(0) <= batch_size:
            features = self.encoder(enc_seq_array)
        else:
            feat_list = []
            for start in range(0, enc_seq_array.size(0), batch_size):
                end = start + batch_size
                feat_list.append(self.encoder(enc_seq_array[start:end]))
            features = torch.cat(feat_list, dim=0)

        return features.reshape(*original_shape, -1)

    def reshape_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets

    def predict(self, inputs: ArrayLike, num_samples: int = 1, latent: bool = False):
        self.eval()
        with torch.inference_mode():
            features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
            pred_dist = self(features)
            pred_dist = pred_dist if latent else self.likelihood(pred_dist)

        pred_mean = pred_dist.mean.cpu()
        pred_std = pred_dist.variance.sqrt().cpu()
        batch_shape = features.shape[:-1]

        if pred_mean.ndim == 1:
            pred_mean = pred_mean.unsqueeze(-1)
            pred_std = pred_std.unsqueeze(-1)
        elif pred_mean.shape[:-1] != batch_shape:
            pred_mean = pred_mean.transpose(-1, -2)
            pred_std = pred_std.transpose(-1, -2)

        assert pred_mean.shape[:-1] == batch_shape, f"{pred_mean.shape[:-1]} != {batch_shape}"

        diag_dist = torch.distributions.Normal(pred_mean, pred_std)
        samples = diag_dist.sample((num_samples,))
        return samples, pred_mean, pred_std

    def predict_mvn(self, inputs: ArrayLike, latent: bool = False):
        self.eval()
        with torch.inference_mode():
            features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
            pred_dist = self(features)
            pred_dist = pred_dist if latent else self.likelihood(pred_dist)
        return pred_dist

    def evaluate(self, loader, split: str = "", *args, **kwargs):
        self.eval()
        targets, y_mean, y_std, f_std = [], [], [], []

        with torch.no_grad():
            for input_batch, target_batch in loader:
                features = self.get_features(input_batch, batch_size=self.bs, transform=False)
                f_dist = self(features)
                y_dist = self.likelihood(f_dist)

                target_batch = self.reshape_targets(target_batch)
                targets.append(target_batch.to(features.device).cpu())

                if y_dist.mean.shape == target_batch.shape:
                    f_std.append(f_dist.variance.sqrt().cpu())
                    y_mean.append(y_dist.mean.cpu())
                    y_std.append(y_dist.variance.sqrt().cpu())
                else:
                    f_std.append(f_dist.variance.sqrt().cpu().transpose(-1, -2))
                    y_mean.append(y_dist.mean.cpu().transpose(-1, -2))
                    y_std.append(y_dist.variance.sqrt().cpu().transpose(-1, -2))

        try:
            targets = torch.cat(targets).view(len(loader.dataset), -1)
            cat_dim = 0
        except Exception:
            targets = torch.cat(targets, -1).view(len(loader.dataset), -1)
            cat_dim = -1

        f_std = torch.cat(f_std, cat_dim).view(len(loader.dataset), -1)
        y_mean = torch.cat(y_mean, cat_dim).view(len(loader.dataset), -1)
        y_std = torch.cat(y_std, cat_dim).view(len(loader.dataset), -1)

        assert y_mean.shape == targets.shape

        rmse = (y_mean - targets).pow(2).mean().sqrt()
        nll = -torch.distributions.Normal(y_mean, y_std).log_prob(targets).mean()

        cal_metrics = quantile_calibration(y_mean, y_std, targets)
        ece = cal_metrics["ece"]
        occ_diff = cal_metrics["occ_diff"]

        spearman_rho = 0.0
        for idx in range(targets.size(-1)):
            corr = spearmanr(targets[..., idx], y_mean[..., idx]).correlation
            spearman_rho += float(corr) / targets.size(-1)

        metrics = {
            f"{split}_nll": float(nll.item()),
            f"{split}_rmse": float(rmse.item()),
            f"{split}_s_rho": float(spearman_rho),
            f"{split}_ece": float(ece),
            f"{split}_occ_diff": float(occ_diff),
            f"{split}_post_var": float((f_std ** 2).mean().item()),
        }

        if hasattr(self.likelihood, "task_noises"):
            metrics["noise"] = float(self.likelihood.task_noises.mean().item())
        elif hasattr(self.likelihood, "noise"):
            metrics["noise"] = float(self.likelihood.noise.mean().item())

        covar_module = self.model.covar_module if hasattr(self, "model") else self.covar_module

        if hasattr(covar_module, "base_kernel") and hasattr(covar_module.base_kernel, "lengthscale"):
            metrics["lengthscale"] = float(covar_module.base_kernel.lengthscale.mean().item())
        elif hasattr(covar_module, "data_covar_module"):
            metrics["lengthscale"] = float(covar_module.data_covar_module.lengthscale.mean().item())
        elif hasattr(covar_module, "lengthscale"):
            metrics["lengthscale"] = float(covar_module.lengthscale.mean().item())

        if hasattr(covar_module, "outputscale"):
            metrics["outputscale"] = float(covar_module.outputscale.mean().item())

        return metrics

    @property
    def param_groups(self):
        gp_hypers = dict(params=[], lr=self.gp_lr)
        noise_group = dict(params=[], lr=self.gp_lr)
        inducing_point_group = dict(params=[], lr=self.gp_lr)
        variational_group = dict(params=[], lr=self.gp_lr)

        for name, param in self.named_parameters():
            if name.split(".")[0] == "encoder":
                continue
            if "noise" in name:
                noise_group["params"].append(param)
            elif "inducing_points" in name:
                inducing_point_group["params"].append(param)
            elif "variational_distribution" in name:
                variational_group["params"].append(param)
            else:
                gp_hypers["params"].append(param)

        param_groups = [gp_hypers]

        if hasattr(self, "encoder") and hasattr(self.encoder, "param_groups"):
            if self.pretrain_lr is not None:
                param_groups.extend(
                    self.encoder.param_groups(self.enc_lr, self.encoder_wd, pretrain_lr=self.pretrain_lr)
                )
            else:
                param_groups.extend(self.encoder.param_groups(self.enc_lr, self.encoder_wd))

        if noise_group["params"]:
            param_groups.append(noise_group)
        if inducing_point_group["params"]:
            param_groups.append(inducing_point_group)
        if variational_group["params"]:
            param_groups.append(variational_group)

        return param_groups


class MultiTaskExactGP(BaseGPSurrogate, KroneckerMultiTaskGP):
    def __init__(
        self,
        feature_dim: int,
        out_dim: int,
        encoder: torch.nn.Module,
        likelihood=None,
        covar_module=None,
        outcome_transform=None,
        input_transform=None,
        *args,
        **kwargs,
    ):
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)

        dummy_X = torch.randn(2, feature_dim, device=self.device, dtype=self.dtype)
        dummy_Y = torch.randn(2, out_dim, device=self.device, dtype=self.dtype)

        if covar_module is not None:
            covar_module = covar_module.to(self.device, self.dtype)

        KroneckerMultiTaskGP.__init__(
            self,
            dummy_X,
            dummy_Y,
            likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            *args,
            **kwargs,
        )

        self.to(self.device, self.dtype)
        if self.likelihood is not None:
            self.likelihood.to(self.device, self.dtype)
            if hasattr(self.likelihood, "initialize"):
                self.likelihood.initialize(task_noises=self.task_noise_init)

        self.encoder = encoder.to(self.device, self.dtype)

    def forward(self, X: ArrayLike):
        features = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X
        return KroneckerMultiTaskGP.forward(self, features)

    def posterior(self, X: ArrayLike, output_indices=None, observation_noise: bool = False, **kwargs):
        features = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X
        return KroneckerMultiTaskGP.posterior(
            self, features, output_indices, observation_noise, **kwargs
        )

    def clear_cache(self):
        clear_cache_hook(self)
        self.prediction_strategy = None

    def set_train_data(self, X=None, targets=None, strict: bool = True):
        self.clear_cache()
        train_features = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X
        KroneckerMultiTaskGP.set_train_data(self, train_features, targets.to(train_features), strict)

    def fit(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        reset: bool = False,
        log_prefix: str = "multi_task_gp",
        **kwargs,
    ):
        if reset:
            raise NotImplementedError

        fit_kwargs = dict(
            surrogate=self,
            mll=ExactMarginalLogLikelihood(self.likelihood, self),
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            train_bs=None,
            eval_bs=self.bs,
            shuffle_train=True,
            log_prefix=log_prefix,
        )
        fit_kwargs.update(kwargs)
        return fit_gp_surrogate(**fit_kwargs)

    def reshape_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape[-1] > 1:
            return targets
        return targets.squeeze(-1)
