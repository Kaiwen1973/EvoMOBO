from __future__ import annotations

import math

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel

from evomobo.models.gp_models import MultiTaskExactGP


class InfiniteWidthBNNKernel(Kernel):
    """Infinite-width fully connected ReLU BNN (NNGP) kernel.

    This is a small compatibility implementation of BoTorch's
    ``InfiniteWidthBNNKernel`` for the GPyTorch 1.6 stack used by this project.
    Keeping it local avoids upgrading the existing GP/qNEHVI environment.
    """

    has_lengthscale = False

    def __init__(
        self,
        depth: int = 3,
        batch_shape: torch.Size | None = None,
        active_dims=None,
        acos_eps: float = 1e-7,
        weight_var_init: float = 1.0,
        bias_var_init: float = 1.0,
        **kwargs,
    ):
        if int(depth) < 0:
            raise ValueError("depth must be non-negative.")
        if not 0.0 < float(acos_eps) < 1.0:
            raise ValueError("acos_eps must be between 0 and 1.")

        batch_shape = torch.Size() if batch_shape is None else torch.Size(batch_shape)
        super().__init__(
            batch_shape=batch_shape,
            active_dims=active_dims,
            **kwargs,
        )
        self.depth = int(depth)
        self.acos_eps = float(acos_eps)

        parameter_shape = (*self.batch_shape, 1, 1)
        self.register_parameter(
            "raw_weight_var",
            torch.nn.Parameter(torch.zeros(parameter_shape)),
        )
        self.register_constraint("raw_weight_var", Positive())
        self.register_parameter(
            "raw_bias_var",
            torch.nn.Parameter(torch.zeros(parameter_shape)),
        )
        self.register_constraint("raw_bias_var", Positive())

        self.weight_var = float(weight_var_init)
        self.bias_var = float(bias_var_init)

    @property
    def weight_var(self) -> torch.Tensor:
        return self.raw_weight_var_constraint.transform(self.raw_weight_var)

    @weight_var.setter
    def weight_var(self, value) -> None:
        value = torch.as_tensor(value).to(self.raw_weight_var)
        self.initialize(
            raw_weight_var=self.raw_weight_var_constraint.inverse_transform(value)
        )

    @property
    def bias_var(self) -> torch.Tensor:
        return self.raw_bias_var_constraint.transform(self.raw_bias_var)

    @bias_var.setter
    def bias_var(self, value) -> None:
        value = torch.as_tensor(value).to(self.raw_bias_var)
        self.initialize(
            raw_bias_var=self.raw_bias_var_constraint.inverse_transform(value)
        )

    def _initialize_var(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.weight_var * torch.sum(x * x, dim=-1, keepdim=True) / x.shape[-1]
            + self.bias_var
        )

    def _update_var(self, covariance: torch.Tensor) -> torch.Tensor:
        return self.weight_var * covariance / 2.0 + self.bias_var

    def _full_covar(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        covariance = (
            self.weight_var * x1.matmul(x2.transpose(-2, -1)) / x1.shape[-1]
            + self.bias_var
        )

        for layer in range(self.depth):
            if layer == 0:
                var1 = self._initialize_var(x1)
                var2 = self._initialize_var(x2)
            else:
                var1 = self._update_var(var1)
                var2 = self._update_var(var2)

            sqrt_term = torch.sqrt(
                var1.matmul(var2.transpose(-2, -1)).clamp_min(1e-30)
            )
            cosine = (covariance / sqrt_term).clamp(
                min=-1.0 + self.acos_eps,
                max=1.0 - self.acos_eps,
            )
            theta = torch.acos(cosine)
            theta_term = torch.sin(theta) + (math.pi - theta) * cosine
            covariance = (
                self.weight_var / (2.0 * math.pi) * sqrt_term * theta_term
                + self.bias_var
            )

        return covariance

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise RuntimeError("last_dim_is_batch is not supported by this kernel.")
        if diag:
            if torch.equal(x1, x2):
                variance = self._initialize_var(x1)
                for _ in range(self.depth):
                    variance = self._update_var(variance)
                return variance.squeeze(-1)
            return self._full_covar(x1, x2).diagonal(dim1=-2, dim2=-1)
        return self._full_covar(x1, x2)


class IBNNMultiTaskExactGP(MultiTaskExactGP):
    """Existing multi-task exact GP using an infinite-width BNN kernel."""

    def evaluate(self, loader, split: str = "", *args, **kwargs):
        metrics = self._evaluate_without_kernel_summary(loader, split=split)
        data_kernel = self.covar_module.data_covar_module
        metrics.update(
            {
                "ibnn_weight_var": float(data_kernel.weight_var.mean().item()),
                "ibnn_bias_var": float(data_kernel.bias_var.mean().item()),
                "ibnn_depth": int(data_kernel.depth),
            }
        )
        return metrics

    def _evaluate_without_kernel_summary(self, loader, split: str):
        """Run the normal GP metrics without requesting a lengthscale."""
        from scipy.stats import spearmanr

        from evomobo.models.metrics import quantile_calibration

        self.eval()
        targets, y_mean, y_std, f_std = [], [], [], []

        with torch.no_grad():
            for input_batch, target_batch in loader:
                features = self.get_features(
                    input_batch,
                    batch_size=self.bs,
                    transform=False,
                )
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

        rmse = (y_mean - targets).pow(2).mean().sqrt()
        nll = -torch.distributions.Normal(y_mean, y_std).log_prob(targets).mean()
        cal_metrics = quantile_calibration(y_mean, y_std, targets)

        spearman_rho = 0.0
        for idx in range(targets.size(-1)):
            corr = spearmanr(targets[..., idx], y_mean[..., idx]).correlation
            spearman_rho += float(corr) / targets.size(-1)

        metrics = {
            f"{split}_nll": float(nll.item()),
            f"{split}_rmse": float(rmse.item()),
            f"{split}_s_rho": float(spearman_rho),
            f"{split}_ece": float(cal_metrics["ece"]),
            f"{split}_occ_diff": float(cal_metrics["occ_diff"]),
            f"{split}_post_var": float((f_std**2).mean().item()),
        }
        if hasattr(self.likelihood, "task_noises"):
            metrics["noise"] = float(self.likelihood.task_noises.mean().item())
        elif hasattr(self.likelihood, "noise"):
            metrics["noise"] = float(self.likelihood.noise.mean().item())
        return metrics
