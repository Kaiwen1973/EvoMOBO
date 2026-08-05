from __future__ import annotations

import logging
import math

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex

from evomobo.utils import batched_call


logger = logging.getLogger(__name__)


class qMultiObjectiveUCB(AcquisitionFunction):
    """Batch UCB averaged over random augmented-Chebyshev objectives."""

    def __init__(
        self,
        model,
        sampler,
        weights,
        lower_bounds,
        outcome_span,
        beta: float,
        alpha: float,
    ):
        super().__init__(model=model)
        self.sampler = sampler
        self.register_buffer("weights", weights)
        self.register_buffer("lower_bounds", lower_bounds)
        self.register_buffer("outcome_span", outcome_span)
        self.beta_prime = math.sqrt(float(beta) * math.pi / 2.0)
        self.alpha = float(alpha)

    def forward(self, X):
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        q = int(samples.shape[-2])
        if q > self.weights.shape[0]:
            raise ValueError(
                f"Received q={q} candidates, but only "
                f"{self.weights.shape[0]} qUCB scalarizations are available."
            )

        lower_bounds = self.lower_bounds.to(samples)
        outcome_span = self.outcome_span.to(samples)
        weights = self.weights[:q].to(samples)
        normalized = (samples - lower_bounds) / outcome_span

        # Evaluate each scalarization across the full q-batch.
        weighted = normalized.unsqueeze(-3) * weights.unsqueeze(-2)
        scalarized = weighted.min(dim=-1).values + self.alpha * weighted.sum(
            dim=-1
        )
        scalarized_mean = scalarized.mean(dim=0, keepdim=True)
        ucb_samples = scalarized_mean + self.beta_prime * (
            scalarized - scalarized_mean
        ).abs()
        batch_ucb = ucb_samples.max(dim=-1).values
        return batch_ucb.mean(dim=0).mean(dim=-1)


class QMultiObjectiveUpperConfidenceBound:
    """qUCB adapter for multi-objective sequence and latent optimization."""

    def __init__(
        self,
        surrogate,
        known_targets,
        num_samples,
        batch_size,
        obj_dim=None,
        beta: float = 0.2,
        alpha: float = 0.05,
        qmc_weights: bool = True,
        weight_seed=None,
        **kwargs,
    ):
        beta = float(beta)
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}.")

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

        surrogate.eval()
        sampler = SobolQMCNormalSampler(num_samples=int(num_samples))
        self.acq_fn = qMultiObjectiveUCB(
            model=surrogate,
            sampler=sampler,
            weights=weights,
            lower_bounds=lower_bounds,
            outcome_span=outcome_span,
            beta=beta,
            alpha=float(alpha),
        )
        self.weights = weights.detach().clone()
        self.out_dim = 1
        self.batch_size = int(batch_size)
        logger.info(
            "qUCB objective weights (one row per trade-off direction): %s",
            self.weights.cpu().tolist(),
        )

    def __call__(self, candidates):
        return torch.cat(
            batched_call(self.acq_fn, candidates, batch_size=1)
        )
