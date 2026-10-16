"""Deterministic seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG the training stack relies on.

    Seeds Python ``random``, NumPy, and torch (CPU and all CUDA devices), and
    sets ``PYTHONHASHSEED`` for worker processes. Call once at run start and
    again after any dataset fork.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
