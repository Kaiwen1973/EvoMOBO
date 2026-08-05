from __future__ import annotations

import copy
import logging
import math
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import BlockDiagLazyTensor, RootLazyTensor
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from evomobo import transforms as gfp_transforms
from evomobo.models.gp_models import BaseGPSurrogate
from evomobo.models.gp_utils import _resolve_gp_records_dir
from evomobo.models.lanmt import lanmt_eval_epoch, lanmt_train_step
from evomobo.models.lm_elements import LanguageModel
from evomobo.models.metrics import quantile_calibration
from evomobo.models.mlm import mlm_eval_epoch, mlm_train_step
from evomobo.models.shared_elements import check_early_stopping
from evomobo.utils import draw_bootstrap, weighted_resampling
from evomobo.wandb_utils import wandb_log


logger = logging.getLogger(__name__)


class MultiTaskGaussianMLP(BaseGPSurrogate):
    """Gaussian-output MLP over encoder features.

    The model predicts an independent Gaussian marginal for every candidate and
    objective. These marginals are assembled into a joint GPyTorch posterior so
    the existing Monte Carlo acquisition functions can sample them without a
    surrogate-specific adapter.
    """

    def __init__(
        self,
        feature_dim: int,
        out_dim: int,
        encoder: torch.nn.Module,
        hidden_dims: Sequence[int] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
        latent_variance_init: float = 0.1,
        observation_noise_init: float = 0.05,
        min_variance: float = 1e-5,
        min_noise: float = 1e-4,
        standardize_features: bool = True,
        *args,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)

        self.feature_dim = int(feature_dim)
        self.out_dim = int(out_dim)
        self.min_variance = float(min_variance)
        self.min_noise = float(min_noise)
        self.standardize_features = bool(standardize_features)
        hidden_dims = tuple(int(width) for width in hidden_dims)

        if self.feature_dim < 1 or self.out_dim < 1:
            raise ValueError("feature_dim and out_dim must be positive.")
        if any(width < 1 for width in hidden_dims):
            raise ValueError("All hidden dimensions must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if float(latent_variance_init) <= 0.0:
            raise ValueError("latent_variance_init must be positive.")
        if float(observation_noise_init) <= 0.0:
            raise ValueError("observation_noise_init must be positive.")

        activation_cls = {
            "relu": torch.nn.ReLU,
            "silu": torch.nn.SiLU,
            "tanh": torch.nn.Tanh,
        }.get(str(activation).lower())
        if activation_cls is None:
            raise ValueError(f"Unsupported activation: {activation!r}.")

        layers = []
        input_dim = self.feature_dim
        for hidden_dim in hidden_dims:
            layers.extend((torch.nn.Linear(input_dim, hidden_dim), activation_cls()))
            if float(dropout) > 0.0:
                layers.append(torch.nn.Dropout(float(dropout)))
            input_dim = hidden_dim
        self.backbone = torch.nn.Sequential(*layers)
        self.mean_head = torch.nn.Linear(input_dim, self.out_dim)
        self.variance_head = torch.nn.Linear(input_dim, self.out_dim)

        torch.nn.init.zeros_(self.mean_head.bias)
        torch.nn.init.zeros_(self.variance_head.weight)
        variance_bias = self._inverse_softplus(
            max(float(latent_variance_init) - self.min_variance, 1e-8)
        )
        torch.nn.init.constant_(self.variance_head.bias, variance_bias)

        noise_value = self._inverse_softplus(
            max(float(observation_noise_init) - self.min_noise, 1e-8)
        )
        self.raw_observation_noise = torch.nn.Parameter(
            torch.full((self.out_dim,), noise_value)
        )
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))

        self.encoder = encoder
        self.to(self.device, self.dtype)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return math.log(math.expm1(float(value)))

    @property
    def num_outputs(self) -> int:
        return self.out_dim

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def observation_noise(self) -> torch.Tensor:
        return (
            torch.nn.functional.softplus(self.raw_observation_noise)
            + self.min_noise
        )

    def _set_feature_stats(self, features: torch.Tensor) -> None:
        if not self.standardize_features:
            return
        flattened = features.detach().reshape(-1, self.feature_dim)
        self.feature_mean.copy_(flattened.mean(dim=0))
        self.feature_scale.copy_(
            flattened.std(dim=0, unbiased=False).clamp_min(1e-5)
        )

    def _standardize(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {features.shape[-1]}."
            )
        if not self.standardize_features:
            return features
        return (features - self.feature_mean) / self.feature_scale

    def _network_outputs(
        self,
        features: torch.Tensor,
        output_indices=None,
    ):
        hidden = self.backbone(self._standardize(features))
        mean = self.mean_head(hidden)
        variance = (
            torch.nn.functional.softplus(self.variance_head(hidden))
            + self.min_variance
        )
        noise = self.observation_noise

        if output_indices is not None:
            index = torch.as_tensor(
                output_indices,
                dtype=torch.long,
                device=mean.device,
            )
            mean = mean.index_select(-1, index)
            variance = variance.index_select(-1, index)
            noise = noise.index_select(0, index)
        return mean, variance, noise

    def _distribution_from_features(
        self,
        features: torch.Tensor,
        observation_noise: bool = False,
        output_indices=None,
    ) -> MultitaskMultivariateNormal:
        mean, variance, noise = self._network_outputs(
            features,
            output_indices=output_indices,
        )
        if observation_noise:
            variance = variance + noise

        task_variances = variance.transpose(-1, -2).clamp_min(
            self.min_variance
        )
        task_roots = torch.diag_embed(task_variances.sqrt())
        covariance = BlockDiagLazyTensor(RootLazyTensor(task_roots))
        return MultitaskMultivariateNormal(
            mean,
            covariance,
            interleaved=False,
        )

    def forward(self, X):
        features = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X
        return self._distribution_from_features(features, observation_noise=False)

    def posterior(
        self,
        X,
        output_indices=None,
        observation_noise: bool = False,
        **kwargs,
    ) -> GPyTorchPosterior:
        features = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X
        distribution = self._distribution_from_features(
            features,
            observation_noise=bool(observation_noise),
            output_indices=output_indices,
        )
        return GPyTorchPosterior(distribution)

    def predict_mvn(self, inputs, latent: bool = False):
        self.eval()
        with torch.inference_mode():
            features = (
                self.get_features(inputs, self.bs)
                if isinstance(inputs, np.ndarray)
                else inputs
            )
            return self._distribution_from_features(
                features,
                observation_noise=not latent,
            )

    def predict(self, inputs, num_samples: int = 1, latent: bool = False):
        distribution = self.predict_mvn(inputs, latent=latent)
        mean = distribution.mean.cpu()
        std = distribution.variance.sqrt().cpu()
        samples = torch.distributions.Normal(mean, std).sample(
            (int(num_samples),)
        )
        return samples, mean, std

    def clear_cache(self):
        return None

    def set_train_data(self, X=None, targets=None, strict: bool = True):
        return None

    def _negative_log_likelihood(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        update_feature_stats: bool,
    ) -> torch.Tensor:
        if update_feature_stats:
            self._set_feature_stats(features)
        mean, latent_variance, noise = self._network_outputs(features)
        targets = targets.reshape(-1, self.out_dim).to(mean)
        predictive_variance = latent_variance + noise
        return 0.5 * (
            torch.log(2.0 * math.pi * predictive_variance)
            + (targets - mean).square() / predictive_variance
        ).mean()

    def _supervised_step(
        self,
        optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        features = self.get_features(
            inputs.to(self.device),
            self.bs,
            transform=False,
        )
        loss = self._negative_log_likelihood(
            features,
            targets.to(features),
            update_feature_stats=True,
        )
        loss.backward()
        optimizer.step()
        return loss

    def evaluate(self, loader, split: str = "", *args, **kwargs):
        self.eval()
        targets, y_mean, y_std, f_std = [], [], [], []
        with torch.no_grad():
            for input_batch, target_batch in loader:
                features = self.get_features(
                    input_batch,
                    batch_size=self.bs,
                    transform=False,
                )
                f_dist = self._distribution_from_features(
                    features,
                    observation_noise=False,
                )
                y_dist = self._distribution_from_features(
                    features,
                    observation_noise=True,
                )
                targets.append(target_batch.reshape(-1, self.out_dim).cpu())
                f_std.append(f_dist.variance.sqrt().cpu())
                y_mean.append(y_dist.mean.cpu())
                y_std.append(y_dist.variance.sqrt().cpu())

        targets = torch.cat(targets, dim=0)
        f_std = torch.cat(f_std, dim=0)
        y_mean = torch.cat(y_mean, dim=0)
        y_std = torch.cat(y_std, dim=0)

        rmse = (y_mean - targets).pow(2).mean().sqrt()
        nll = -torch.distributions.Normal(y_mean, y_std).log_prob(targets).mean()
        cal_metrics = quantile_calibration(y_mean, y_std, targets)

        spearman_rho = 0.0
        for idx in range(targets.size(-1)):
            corr = spearmanr(targets[..., idx], y_mean[..., idx]).correlation
            spearman_rho += float(corr) / targets.size(-1)

        return {
            f"{split}_nll": float(nll.item()),
            f"{split}_rmse": float(rmse.item()),
            f"{split}_s_rho": float(spearman_rho),
            f"{split}_ece": float(cal_metrics["ece"]),
            f"{split}_occ_diff": float(cal_metrics["occ_diff"]),
            f"{split}_post_var": float((f_std**2).mean().item()),
            "noise": float(self.observation_noise.mean().item()),
            "gaussian_mlp_latent_var": float((f_std**2).mean().item()),
        }

    def fit(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        reset: bool = False,
        log_prefix: str = "multi_task_gaussian_mlp",
        encoder_obj: Optional[str] = "mll",
        resampling_temp: Optional[float] = None,
        select_crit_key: str = "val_nll",
        save_csv: bool = True,
        csv_dir: Optional[str] = ".",
        csv_name: Optional[str] = None,
        **kwargs,
    ):
        if reset:
            raise NotImplementedError
        if encoder_obj not in ("mll", "mlm", "lanmt", None):
            raise ValueError(f"Unsupported encoder objective: {encoder_obj!r}")

        num_val = 0 if X_val is None else int(X_val.shape[0])
        logger.info(
            "%d train, %d val, %d test",
            int(X_train.shape[0]),
            num_val,
            int(X_test.shape[0]),
        )

        if self.bootstrap_ratio is not None:
            X_train, Y_train = draw_bootstrap(
                X_train,
                Y_train,
                bootstrap_ratio=self.bootstrap_ratio,
                min_samples=self.min_num_train,
            )
        if resampling_temp is not None:
            _, _, train_idxs = weighted_resampling(-Y_train, k=resampling_temp)
            X_train, Y_train = X_train[train_idxs], Y_train[train_idxs]
            if X_val is not None and Y_val is not None:
                _, _, val_idxs = weighted_resampling(-Y_val, k=resampling_temp)
                X_val, Y_val = X_val[val_idxs], Y_val[val_idxs]

        def collate_fn(batch):
            return gfp_transforms.padding_collate_fn(
                batch,
                self.tokenizer.padding_idx,
            )

        train_dataset, test_dataset = self._get_datasets(
            X_train,
            X_test,
            Y_train,
            Y_test,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            shuffle=True,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=max(1, min(self.bs, len(test_dataset))),
            shuffle=False,
            collate_fn=collate_fn,
        )

        val_loader = None
        if num_val > 0:
            _, val_dataset = self._get_datasets(
                X_train,
                X_val,
                Y_train,
                Y_val,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=max(1, min(self.bs, len(val_dataset))),
                shuffle=False,
                collate_fn=collate_fn,
            )

        has_encoder = len(list(self.encoder.parameters())) > 0
        is_lm_encoder = has_encoder and isinstance(self.encoder, LanguageModel)
        if not is_lm_encoder and encoder_obj in ("mlm", "lanmt"):
            logger.info(
                "encoder_obj='%s' requested, but encoder is not a LanguageModel; "
                "fallback to 'mll'.",
                encoder_obj,
            )
            effective_encoder_obj = "mll"
        else:
            effective_encoder_obj = encoder_obj

        logger.info("\n---- preparing Gaussian MLP checkpoint ----")
        self.eval()
        self.requires_grad_(False)
        if self.standardize_features:
            with torch.no_grad():
                initial_inputs, _ = next(iter(train_loader))
                initial_features = self.get_features(
                    initial_inputs.to(self.device),
                    self.bs,
                    transform=False,
                )
                self._set_feature_stats(initial_features)

        start_metrics = {"epoch": 0}
        if val_loader is not None:
            start_metrics.update(self.evaluate(val_loader, split="val"))
        start_metrics.update(self.evaluate(test_loader, split="test"))
        if is_lm_encoder and effective_encoder_obj == "mlm":
            if val_loader is not None:
                start_metrics.update(
                    mlm_eval_epoch(
                        self.encoder,
                        val_loader,
                        self.encoder.mask_ratio,
                        split="val",
                    )
                )
            start_metrics.update(
                mlm_eval_epoch(
                    self.encoder,
                    test_loader,
                    self.encoder.mask_ratio,
                    split="test",
                )
            )
        elif is_lm_encoder and effective_encoder_obj == "lanmt":
            if val_loader is not None:
                start_metrics.update(
                    lanmt_eval_epoch(self.encoder.model, val_loader, split="val")
                )
            start_metrics.update(
                lanmt_eval_epoch(self.encoder.model, test_loader, split="test")
            )

        best_score = start_metrics.get(select_crit_key)
        best_score_epoch = 0
        self.cpu()
        best_weights = copy.deepcopy(self.state_dict())
        self.to(self.device)

        optimizer = torch.optim.Adam(self.param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=math.ceil(self.patience / 2.0),
            threshold=1e-3,
        )
        records = [start_metrics]
        best_loss, best_loss_epoch = None, 0

        logger.info("\n---- fitting Gaussian MLP and encoder ----")
        for epoch_idx in range(self.num_epochs):
            avg_train_loss = 0.0

            if has_encoder and effective_encoder_obj == "mll":
                self.requires_grad_(False)
                self.encoder.requires_grad_(True)
                for inputs, targets in train_loader:
                    loss = self._supervised_step(optimizer, inputs, targets)
                    avg_train_loss += float(loss.detach().item())

            self.train()
            for inputs, targets in train_loader:
                if is_lm_encoder and effective_encoder_obj == "mlm":
                    self.encoder.requires_grad_(True)
                    auxiliary_loss, _, _ = mlm_train_step(
                        self.encoder,
                        optimizer,
                        inputs,
                        self.encoder.mask_ratio,
                        loss_scale=1.0,
                    )
                elif is_lm_encoder and effective_encoder_obj == "lanmt":
                    self.encoder.requires_grad_(True)
                    auxiliary_loss, _, _ = lanmt_train_step(
                        self.encoder.model,
                        optimizer,
                        inputs,
                        loss_scale=1.0,
                    )
                else:
                    auxiliary_loss = torch.zeros((), device=self.device)

                self.requires_grad_(True)
                supervised_loss = self._supervised_step(
                    optimizer,
                    inputs,
                    targets,
                )
                avg_train_loss += float(
                    (auxiliary_loss.detach() + supervised_loss.detach()).item()
                )

            scheduler.step(avg_train_loss)
            metrics = {
                "epoch": epoch_idx + 1,
                "train_loss": float(avg_train_loss),
            }

            if (epoch_idx + 1) % self.eval_period == 0:
                self.requires_grad_(False)
                self.eval()
                if val_loader is not None:
                    metrics.update(self.evaluate(val_loader, split="val"))
                metrics.update(self.evaluate(test_loader, split="test"))
                if is_lm_encoder and effective_encoder_obj == "mlm":
                    if val_loader is not None:
                        metrics.update(
                            mlm_eval_epoch(
                                self.encoder,
                                val_loader,
                                self.encoder.mask_ratio,
                                split="val",
                            )
                        )
                    metrics.update(
                        mlm_eval_epoch(
                            self.encoder,
                            test_loader,
                            self.encoder.mask_ratio,
                            split="test",
                        )
                    )
                elif is_lm_encoder and effective_encoder_obj == "lanmt":
                    if val_loader is not None:
                        metrics.update(
                            lanmt_eval_epoch(
                                self.encoder.model,
                                val_loader,
                                split="val",
                            )
                        )
                    metrics.update(
                        lanmt_eval_epoch(
                            self.encoder.model,
                            test_loader,
                            split="test",
                        )
                    )

            select_crit = metrics.get(select_crit_key)
            stop = False
            if select_crit is not None:
                best_score, best_score_epoch, best_weights, _ = check_early_stopping(
                    model=self,
                    best_score=best_score,
                    best_epoch=best_score_epoch,
                    best_weights=best_weights,
                    curr_score=select_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=self.patience,
                    save_weights=True,
                )

            best_loss, best_loss_epoch, _, stop = check_early_stopping(
                model=self,
                best_score=best_loss,
                best_epoch=best_loss_epoch,
                best_weights=None,
                curr_score=metrics["train_loss"],
                curr_epoch=epoch_idx + 1,
                patience=self.patience,
                save_weights=False,
            )
            metrics.update(
                {
                    "best_score": best_score,
                    "best_epoch": int(best_score_epoch),
                    "best_loss": best_loss,
                    "best_loss_epoch": int(best_loss_epoch),
                }
            )
            records.append(metrics)
            wandb_log(
                {
                    f"{log_prefix}/{key}": value
                    for key, value in metrics.items()
                }
            )

            if self.early_stopping and stop:
                break

        logger.info(
            "\n---- loading best Gaussian MLP checkpoint: epoch=%d, %s=%s ----",
            best_score_epoch,
            select_crit_key,
            best_score,
        )
        self.load_state_dict(best_weights, strict=False)
        self.requires_grad_(False)
        self.eval()

        if save_csv:
            try:
                if csv_name is None:
                    csv_name = f"gaussian_mlp_training_{log_prefix}.csv"
                resolved_dir = _resolve_gp_records_dir(csv_dir)
                os.makedirs(resolved_dir, exist_ok=True)
                csv_path = os.path.join(resolved_dir, csv_name)
                pd.DataFrame(records).to_csv(csv_path, index=False)
                logger.info("\ntraining records saved to: %s", csv_path)
            except Exception as exc:
                logger.exception(
                    "Failed to save Gaussian MLP training records CSV: %s",
                    exc,
                )

        return records
