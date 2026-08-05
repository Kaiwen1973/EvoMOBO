from __future__ import annotations

import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective import infer_reference_point
from botorch.utils.multi_objective.box_decompositions import (
    NondominatedPartitioning,
)

from evomobo.acquisitions.monte_carlo import qDiscreteEHVI
from evomobo.utils import batched_call


class QExpectedHypervolumeImprovement:
    """qEHVI adapter for sequence and latent-feature candidate batches."""

    def __init__(
        self,
        surrogate,
        known_targets,
        num_samples,
        batch_size,
        ref_point=None,
        **kwargs,
    ):
        self.ref_point = (
            infer_reference_point(known_targets)
            if ref_point is None
            else ref_point
        )
        sampler = SobolQMCNormalSampler(num_samples=int(num_samples))
        partitioning = NondominatedPartitioning(
            ref_point=self.ref_point,
            Y=known_targets,
        )
        self.acq_fn = qDiscreteEHVI(
            model=surrogate,
            ref_point=self.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        self.out_dim = 1
        self.batch_size = int(batch_size)

    def __call__(self, candidates):
        return torch.cat(
            batched_call(self.acq_fn, candidates, batch_size=1)
        )
