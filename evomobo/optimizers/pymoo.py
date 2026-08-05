import numpy as np
import torch

from botorch.utils.multi_objective import pareto


def pareto_frontier(candidate_pool, obj_vals):
    candidate_pool = np.asarray(candidate_pool, dtype=object)
    obj_vals = np.asarray(obj_vals, dtype=float)

    assert len(candidate_pool) == obj_vals.shape[0]
    if len(candidate_pool) == 1:
        return candidate_pool, obj_vals

    # Convert stored minimization values for BoTorch.
    pareto_mask = pareto.is_non_dominated(
        -torch.as_tensor(obj_vals, dtype=torch.float32)
    ).cpu().numpy().astype(bool)
    return candidate_pool[pareto_mask], obj_vals[pareto_mask]


class Normalizer(object):
    def __init__(self, loc=0., scale=1.):
        self.loc = loc
        self.scale = np.where(scale != 0, scale, 1.)

    def __call__(self, arr):
        min_val = self.loc - 4 * self.scale
        max_val = self.loc + 4 * self.scale
        clipped_arr = np.clip(arr, a_min=min_val, a_max=max_val)
        norm_arr = (clipped_arr - self.loc) / self.scale

        return norm_arr

    def inv_transform(self, arr):
        return self.scale * arr + self.loc
