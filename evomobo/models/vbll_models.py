from __future__ import annotations

import logging
import math
import os
from typing import Optional

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


class MultiTaskVBLL(BaseGPSurrogate):
    """Variational Bayesian linear last layer over encoder features.

    Each objective has a full-covariance Gaussian posterior over its linear
    weights. The resulting predictive distribution is jointly Gaussian across
    candidates, so the existing Sobol sampler and qNEHVI implementation can be
    used without an acquisition-specific adapter.
    """

    def __init__(
        self,
        feature_dim: int,
        out_dim: int,
        encoder: torch.nn.Module,
        prior_scale: float = 1.0,
        posterior_scale_init: float = 0.1,
        observation_noise_init: float = 0.15,
        min_scale: float = 1e-5,
        min_noise: float = 1e-4,
        kl_weight: float = 1.0,
        standardize_features: bool = True,
        *args,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)

        if int(feature_dim) < 1 or int(out_dim) < 1:
            raise ValueError("feature_dim and out_dim must be positive.")
        if float(prior_scale) <= 0.0:
            raise ValueError("prior_scale must be positive.")
        if float(posterior_scale_init) <= 0.0:
            raise ValueError("posterior_scale_init must be positive.")
        if float(observation_noise_init) <= 0.0:
            raise ValueError("observation_noise_init must be positive.")

        self.feature_dim = int(feature_dim)
        self.out_dim = int(out_dim)
        self.prior_scale = float(prior_scale)
        self.min_scale = float(min_scale)
        self.min_noise = float(min_noise)
        self.kl_weight = float(kl_weight)
        self.standardize_features = bool(standardize_features)

        linear_dim = self.feature_dim + 1
        self.weight_mean = torch.nn.Parameter(
            torch.zeros(self.out_dim, linear_dim)
        )

        raw_tril = torch.zeros(self.out_dim, linear_dim, linear_dim)
        raw_diag = self._inverse_softplus(
            max(float(posterior_scale_init) - self.min_scale, 1e-8)
        )
        raw_tril.diagonal(dim1=-2, dim2=-1).fill_(raw_diag)
        self.raw_weight_tril = torch.nn.Parameter(raw_tril)

        raw_noise = self._inverse_softplus(
            max(float(observation_noise_init) - self.min_noise, 1e-8)
        )
        self.raw_noise = torch.nn.Parameter(
            torch.full((self.out_dim,), raw_noise)
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
    def weight_scale_tril(self) -> torch.Tensor:
        lower = torch.tril(self.raw_weight_tril)
        raw_diag = lower.diagonal(dim1=-2, dim2=-1)
        positive_diag = torch.nn.functional.softplus(raw_diag) + self.min_scale
        return lower - torch.diag_embed(raw_diag) + torch.diag_embed(positive_diag)

    @property
    def noise_variance(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_noise) + self.min_noise

    def _set_feature_stats(self, features: torch.Tensor) -> None:
        if not self.standardize_features:
            return
        flattened = features.detach().reshape(-1, self.feature_dim)
        self.feature_mean.copy_(flattened.mean(dim=0))
        self.feature_scale.copy_(flattened.std(dim=0, unbiased=False).clamp_min(1e-5))

    def _linear_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, got {features.shape[-1]}."
            )
        if self.standardize_features:
            features = (features - self.feature_mean) / self.feature_scale
        intercept = torch.ones_like(features[..., :1])
        return torch.cat((features, intercept), dim=-1)

    def _select_outputs(self, output_indices=None):
        if output_indices is None:
            return self.weight_mean, self.weight_scale_tril, self.noise_variance
        index = torch.as_tensor(
            output_indices,
            dtype=torch.long,
            device=self.weight_mean.device,
        )
        return (
            self.weight_mean.index_select(0, index),
            self.weight_scale_tril.index_select(0, index),
            self.noise_variance.index_select(0, index),
        )

    def _distribution_from_features(
        self,
        features: torch.Tensor,
        observation_noise: bool = False,
        output_indices=None,
    ) -> MultitaskMultivariateNormal:
        phi = self._linear_features(features)
        weight_mean, weight_tril, noise = self._select_outputs(output_indices)

        mean = torch.einsum("...np,mp->...nm", phi, weight_mean)
        projected = torch.einsum("...np,mpk->...mnk", phi, weight_tril)

        # Float64 preserves the stabilizing jitter.
        stable_projected = projected.to(torch.float64)
        task_covariances = torch.einsum(
            "...mnk,...mqk->...mnq",
            stable_projected,
            stable_projected,
        )
        task_covariances = 0.5 * (
            task_covariances + task_covariances.transpose(-1, -2)
        )

        num_points = phi.shape[-2]
        num_tasks = weight_mean.shape[0]
        eye_points = torch.eye(
            num_points,
            dtype=task_covariances.dtype,
            device=phi.device,
        )
        jitter = max(self.min_scale**2, 1e-8)
        diagonal_variance = torch.full_like(
            noise,
            jitter,
            dtype=task_covariances.dtype,
        )
        if observation_noise:
            diagonal_variance = diagonal_variance + noise.to(
                task_covariances.dtype
            )
        task_covariances = task_covariances + (
            diagonal_variance.view(num_tasks, 1, 1) * eye_points
        )

        task_roots = torch.linalg.cholesky(task_covariances).to(projected)
        full_covariance = BlockDiagLazyTensor(RootLazyTensor(task_roots))
        return MultitaskMultivariateNormal(
            mean,
            full_covariance,
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
        pred_dist = self.predict_mvn(inputs, latent=latent)
        pred_mean = pred_dist.mean.cpu()
        pred_std = pred_dist.variance.sqrt().cpu()
        marginal = torch.distributions.Normal(pred_mean, pred_std)
        samples = marginal.sample((int(num_samples),))
        return samples, pred_mean, pred_std

    def clear_cache(self):
        return None

    def set_train_data(self, X=None, targets=None, strict: bool = True):
        # Fitted state lives in the variational parameters.
        return None

    def _negative_elbo(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        update_feature_stats: bool,
    ) -> torch.Tensor:
        if update_feature_stats:
            self._set_feature_stats(features)
        phi = self._linear_features(features)
        targets = targets.reshape(-1, self.out_dim).to(phi)

        weight_tril = self.weight_scale_tril
        mean = torch.einsum("np,mp->nm", phi, self.weight_mean)
        projected = torch.einsum("np,mpk->nmk", phi, weight_tril)
        latent_variance = projected.square().sum(dim=-1)
        noise = self.noise_variance.view(1, -1)

        expected_nll = 0.5 * (
            torch.log(2.0 * math.pi * noise)
            + ((targets - mean).square() + latent_variance) / noise
        )

        prior_variance = self.prior_scale**2
        posterior_trace = weight_tril.square().sum(dim=(-2, -1))
        mean_square = self.weight_mean.square().sum(dim=-1)
        posterior_logdet = 2.0 * torch.log(
            weight_tril.diagonal(dim1=-2, dim2=-1)
        ).sum(dim=-1)
        linear_dim = self.weight_mean.shape[-1]
        kl = 0.5 * (
            (posterior_trace + mean_square) / prior_variance
            - linear_dim
            + linear_dim * math.log(prior_variance)
            - posterior_logdet
        ).sum()

        return (
            expected_nll.sum() + self.kl_weight * kl
        ) / max(1, targets.shape[0])

    def _supervised_step(
        self,
        optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        update_feature_stats: bool = True,
    ) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        features = self.get_features(
            inputs.to(self.device),
            self.bs,
            transform=False,
        )
        loss = self._negative_elbo(
            features,
            targets.to(features),
            update_feature_stats=update_feature_stats,
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
            "noise": float(self.noise_variance.mean().item()),
            "vbll_weight_std": float(
                self.weight_scale_tril.diagonal(dim1=-2, dim2=-1).mean().item()
            ),
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
        log_prefix: str = "multi_task_vbll",
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

        logger.info("\n---- preparing VBLL checkpoint ----")
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

        # Epoch 0 is an unfitted prior and cannot be selected.
        best_score = None
        best_score_epoch = 0
        best_weights = None

        optimizer = torch.optim.Adam(self.param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=math.ceil(self.patience / 2.0),
            threshold=1e-3,
        )
        records = [start_metrics]
        best_loss, best_loss_epoch = None, 0

        logger.info("\n---- fitting VBLL and encoder ----")
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

        if best_weights is not None:
            logger.info(
                "\n---- loading best trained VBLL checkpoint: "
                "epoch=%d, %s=%s ----",
                best_score_epoch,
                select_crit_key,
                best_score,
            )
            self.load_state_dict(best_weights, strict=False)
        else:
            # Keep the final state when the selection metric is unavailable.
            best_score_epoch = int(records[-1].get("epoch", self.num_epochs))
            records[-1]["best_epoch"] = best_score_epoch
            logger.warning(
                "Selection metric '%s' was not evaluated; retaining the final "
                "trained VBLL state at epoch=%d.",
                select_crit_key,
                best_score_epoch,
            )
        self.requires_grad_(False)
        self.eval()

        if save_csv:
            try:
                if csv_name is None:
                    csv_name = f"vbll_training_{log_prefix}.csv"
                resolved_dir = _resolve_gp_records_dir(csv_dir)
                os.makedirs(resolved_dir, exist_ok=True)
                csv_path = os.path.join(resolved_dir, csv_name)
                pd.DataFrame(records).to_csv(csv_path, index=False)
                logger.info("\ntraining records saved to: %s", csv_path)
            except Exception as exc:
                logger.exception("Failed to save VBLL training records CSV: %s", exc)

        return records
