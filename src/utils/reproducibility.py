"""Utilidades para que los experimentos sean reproducibles."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch (CPU y GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
