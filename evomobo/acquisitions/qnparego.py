from __future__ import annotations

import logging

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex
from torch import Tensor

from evomobo.utils import batched_call


logger = logging.getLogger(__name__)


class qDiscreteNParEGO(AcquisitionFunction):
    """Parallel noisy EI with one Chebyshev scalarization per candidate."""

    def __init__(
        self,
        model,
        X_baseline,
        X_baseline_string,
        sampler,
        weights,
        lower_bounds,
        outcome_span,
        alpha: float,
    ):
        super().__init__(model=model)
        self.sampler = sampler
        self.register_buffer("X_baseline", X_baseline)
        self.register_buffer("weights", weights)
        self.register_buffer("lower_bounds", lower_bounds)
        self.register_buffer("outcome_span", outcome_span)
        self.alpha = float(alpha)
        self.X_baseline_string = np.asarray(
            X_baseline_string,
            dtype=object,
        ).copy()

    def forward(self, X):
        if isinstance(X, Tensor):
            baseline_X = self.X_baseline.expand(
                *X.shape[:-2],
                -1,
                -1,
            )
            X_full = torch.cat([X, baseline_X], dim=-2)
            q = X.shape[-2]
        else:
            baseline_X = np.broadcast_to(
                self.X_baseline_string,
                X.shape[:-1] + self.X_baseline_string.shape,
            )
            X_full = np.concatenate([X, baseline_X], axis=-1)
            q = X.shape[-1]

        if q > self.weights.shape[0]:
            raise ValueError(
                f"Received q={q} candidates, but only "
                f"{self.weights.shape[0]} qNParEGO scalarizations are available."
            )

        posterior = self.model.posterior(X_full)
        samples = self.sampler(posterior)

        lower_bounds = self.lower_bounds.to(samples)
        outcome_span = self.outcome_span.to(samples)
        weights = self.weights[:q].to(samples)
        normalized = (samples - lower_bounds) / outcome_span

        candidate_values = normalized[..., :q, :] * weights
        candidate_values = candidate_values.min(dim=-1).values + self.alpha * (
            candidate_values.sum(dim=-1)
        )

        baseline_values = normalized[..., q:, :].unsqueeze(-3)
        baseline_values = baseline_values * weights.unsqueeze(-2)
        baseline_values = baseline_values.min(dim=-1).values + self.alpha * (
            baseline_values.sum(dim=-1)
        )
        baseline_best = baseline_values.max(dim=-1).values

        improvement = (candidate_values - baseline_best).clamp_min(0)
        return improvement.mean(dim=0).mean(dim=-1)


class QNoisyParEGO:
    """qNParEGO adapter for the joint latent optimizer used by LaMBO."""

    def __init__(
        self,
        surrogate,
        X_baseline,
        known_targets,
        num_samples,
        batch_size,
        obj_dim=None,
        alpha: float = 0.05,
        qmc_weights: bool = True,
        weight_seed=None,
        **kwargs,
    ):
        inferred_dim = int(known_targets.shape[-1])
        if obj_dim is not None and int(obj_dim) != inferred_dim:
            raise ValueError(
                "obj_dim does not match known_targets: "
                f"obj_dim={obj_dim}, targets={inferred_dim}."
            )

        seed = None if weight_seed is None else int(weight_seed)
        weights = sample_simplex(
            d=inferred_dim,
            n=int(batch_size),
            qmc=bool(qmc_weights),
            seed=seed,
            device=known_targets.device,
            dtype=known_targets.dtype,
        )

        lower_bounds = known_targets.min(dim=0).values
        outcome_span = known_targets.max(dim=0).values - lower_bounds
        span_eps = 16 * torch.finfo(known_targets.dtype).eps
        outcome_span = torch.where(
            outcome_span.abs() <= span_eps,
            torch.ones_like(outcome_span),
            outcome_span,
        )
        sampler = SobolQMCNormalSampler(num_samples=int(num_samples))

        surrogate.eval()
        with torch.no_grad():
            baseline_features = surrogate.get_features(
                X_baseline,
                surrogate.bs,
            ).detach()

        self.acq_fn = qDiscreteNParEGO(
            model=surrogate,
            X_baseline=baseline_features,
            X_baseline_string=X_baseline,
            sampler=sampler,
            weights=weights,
            lower_bounds=lower_bounds,
            outcome_span=outcome_span,
            alpha=float(alpha),
        )
        self.weights = weights.detach().clone()
        self.out_dim = 1
        self.batch_size = int(batch_size)
        logger.info(
            "qNParEGO objective weights (one row per candidate): %s",
            self.weights.cpu().tolist(),
        )

    def __call__(self, candidates):
        return torch.cat(
            batched_call(self.acq_fn, candidates, batch_size=1)
        )
