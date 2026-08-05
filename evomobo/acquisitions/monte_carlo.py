from numpy import array, copy, concatenate
from torch import Tensor
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
)
from botorch.posteriors import GPyTorchPosterior, Posterior, DeterministicPosterior
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import BlockDiagLazyTensor
import torch



class qDiscreteEHVI(qExpectedHypervolumeImprovement):
    def forward(self, X: array) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        return self._compute_qehvi(samples=samples)

    
class qDiscreteNEHVI(qNoisyExpectedHypervolumeImprovement):
    
    def __init__(
        self,
        model,
        ref_point,
        X_baseline,
        sampler = None,
        objective = None,
        constraints = None,
        X_pending = None,
        eta: float = 1e-3,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        **kwargs,
    ):
        model.eval()
        mocked_features = model.get_features(X_baseline, model.bs)
        ref_point = ref_point.to(mocked_features)
        # Collapse extra string-kernel dimensions.
        if mocked_features.ndim > 2:
            mocked_features = mocked_features[..., 0]

        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=mocked_features,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            **kwargs
        )
        self.X_baseline_string = X_baseline

    def forward(self, X: array) -> Tensor:
        if isinstance(X, Tensor):
            baseline_X = self._X_baseline
            baseline_X = baseline_X.expand(*X.shape[:-2], -1, -1)
            X_full = torch.cat([baseline_X, X], dim=-2)
            q = X.shape[-2]
        else:
            baseline_X = copy(self.X_baseline_string)  # Copy before resize.
            baseline_X.resize(
                baseline_X.shape[:-(X.ndim)] + X.shape[:-1] + baseline_X.shape[-1:]
            )
            X_full = concatenate([baseline_X, X], axis=-1)
            q = X.shape[-1]
        posterior = self.model.posterior(X_full)
        self._set_sampler(q=q, posterior=posterior)
        samples = self.sampler(posterior)[..., -q:, :]
        return self._compute_qehvi(samples=samples) + self._prev_nehvi
    
    def _cache_root_decomposition(self, posterior: GPyTorchPosterior) -> None:
        if posterior.mvn._interleaved:
            if hasattr(posterior.mvn.lazy_covariance_matrix, 'base_lazy_tensor'):
                posterior_lc_base = posterior.mvn.lazy_covariance_matrix.base_lazy_tensor
            else:
                posterior_lc_base = posterior.mvn.lazy_covariance_matrix

            new_lazy_covariance = BlockDiagLazyTensor(posterior_lc_base)
            posterior.mvn = MultitaskMultivariateNormal(posterior.mvn.mean, new_lazy_covariance, interleaved=False)
        return super()._cache_root_decomposition(posterior=posterior)
    
    
class qMTGPDiscreteNEHVI(qDiscreteNEHVI):
    
    _uses_matheron = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(cache_root = False, *args, **kwargs)
    
    def _set_sampler(
        self,
        q: int,
        posterior: Posterior,
    ) -> None:
        if self.q != q:
            base_sample_shape = self.sampler._get_base_sample_shape(posterior=posterior)
            self.sampler._construct_base_samples(
                posterior=posterior, shape=base_sample_shape
            )
            if (
                self.X_baseline.shape[0] > 0
                and self.base_sampler.base_samples is not None
                and not isinstance(posterior, DeterministicPosterior)
            ):
                current_base_samples = self.base_sampler.base_samples.detach().clone()
                base_ndims = current_base_samples.dim() - 1
                view_shape = (
                    self.sampler.sample_shape
                    + torch.Size(
                        [1] * (len(base_sample_shape) - current_base_samples.dim())
                    )
                    + current_base_samples.shape[-base_ndims:]
                )
                expanded_shape = (
                    base_sample_shape[:-base_ndims]
                    + current_base_samples.shape[-base_ndims:]
                )
                end_idx = current_base_samples.shape[-1 if self._uses_matheron else -2]
                expanded_samples = current_base_samples.view(view_shape).expand(
                    expanded_shape
                )
                if self._uses_matheron:
                    self.sampler.base_samples[..., :end_idx] = expanded_samples
                else:
                    self.sampler.base_samples[..., :end_idx, :] = expanded_samples
                self._cache_q_subset_indices(q=q)
