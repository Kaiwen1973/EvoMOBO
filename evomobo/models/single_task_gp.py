from __future__ import annotations

from typing import Any

import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.memoize import clear_cache_hook

from evomobo.models.gp_models import ArrayLike, BaseGPSurrogate
from evomobo.models.gp_utils import fit_gp_surrogate


class SingleTaskExactGP(BaseGPSurrogate, SingleTaskGP):
    """Deep-kernel exact GP with one scalar protein objective."""

    def __init__(
        self,
        feature_dim: int,
        encoder: torch.nn.Module,
        likelihood: Any = None,
        covar_module: Any = None,
        outcome_transform: Any = None,
        input_transform: Any = None,
        *args,
        **kwargs,
    ):
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)

        dummy_x = torch.randn(2, feature_dim, device=self.device, dtype=self.dtype)
        dummy_y = torch.randn(2, 1, device=self.device, dtype=self.dtype)
        if covar_module is not None:
            covar_module = covar_module.to(self.device, self.dtype)

        SingleTaskGP.__init__(
            self,
            dummy_x,
            dummy_y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        self.to(self.device, self.dtype)
        if self.likelihood is not None:
            self.likelihood.to(self.device, self.dtype)
            if hasattr(self.likelihood, "initialize"):
                self.likelihood.initialize(noise=self.task_noise_init)
        self.encoder = encoder.to(self.device, self.dtype)

    def forward(self, x: ArrayLike):
        features = self.get_features(x, self.bs) if isinstance(x, np.ndarray) else x
        return SingleTaskGP.forward(self, features)

    def posterior(
        self,
        x: ArrayLike,
        output_indices=None,
        observation_noise: bool = False,
        **kwargs,
    ):
        features = self.get_features(x, self.bs) if isinstance(x, np.ndarray) else x
        return SingleTaskGP.posterior(
            self,
            features,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs,
        )

    def clear_cache(self):
        clear_cache_hook(self)
        self.prediction_strategy = None

    def set_train_data(self, x=None, targets=None, strict: bool = True):
        self.clear_cache()
        train_features = self.get_features(x, self.bs) if isinstance(x, np.ndarray) else x
        train_targets = self.reshape_targets(targets).to(train_features)
        SingleTaskGP.set_train_data(
            self,
            inputs=train_features,
            targets=train_targets,
            strict=strict,
        )

    def fit(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        reset: bool = False,
        log_prefix: str = "single_task_gp",
        **kwargs,
    ):
        if reset:
            raise NotImplementedError
        fit_kwargs = dict(
            surrogate=self,
            mll=ExactMarginalLogLikelihood(self.likelihood, self),
            X_train=x_train,
            Y_train=y_train,
            X_val=x_val,
            Y_val=y_val,
            X_test=x_test,
            Y_test=y_test,
            train_bs=None,
            eval_bs=self.bs,
            shuffle_train=True,
            log_prefix=log_prefix,
        )
        fit_kwargs.update(kwargs)
        return fit_gp_surrogate(**fit_kwargs)

    @staticmethod
    def reshape_targets(targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim > 1 and targets.shape[-1] == 1:
            return targets.squeeze(-1)
        return targets
