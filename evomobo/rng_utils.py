from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def isolated_seed(seed: int):
    """Temporarily seed global RNGs without advancing the caller's RNG stream.

    Zero intentionally follows the original execution path without inspecting,
    saving, seeding, or restoring any RNG state.
    """
    seed = int(seed)
    if seed == 0:
        yield
        return

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
