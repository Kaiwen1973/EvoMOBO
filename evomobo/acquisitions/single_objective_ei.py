from __future__ import annotations

import torch
from botorch.sampling import SobolQMCNormalSampler


class BatchExpectedImprovement:
    """Joint q-EI for string candidates or differentiable latent features."""

    def __init__(
        self,
        surrogate,
        known_targets,
        num_samples: int,
        batch_size: int,
        best_f=None,
        **kwargs,
    ):
        targets = torch.as_tensor(
            known_targets,
            dtype=torch.float32,
            device=surrogate.device,
        ).reshape(-1)
        resolved_best_f = targets.max() if best_f is None else torch.as_tensor(
            best_f,
            dtype=targets.dtype,
            device=targets.device,
        )
        self.best_f = resolved_best_f
        self.out_dim = 1
        self.batch_size = int(batch_size)
        self.surrogate = surrogate
        self.sampler = SobolQMCNormalSampler(num_samples=int(num_samples))

    def __call__(self, candidates):
        posterior = self.surrogate.posterior(candidates)
        samples = self.sampler(posterior).squeeze(-1)
        sample_best = samples.max(dim=-1).values
        return (sample_best - self.best_f).clamp_min(0.0).mean(dim=0)
