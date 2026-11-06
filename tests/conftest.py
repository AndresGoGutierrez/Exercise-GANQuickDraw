"""Fixtures compartidas por las pruebas."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.config import Config


@pytest.fixture
def fake_npy(tmp_path: Path) -> Path:
    """Crea un .npy sintético con el mismo formato que Quick, Draw! (N, 784) uint8."""
    rng = np.random.default_rng(0)
    path = tmp_path / "fake_category.npy"
    np.save(path, rng.integers(0, 256, size=(300, 784), dtype=np.uint8))
    return path


@pytest.fixture
def small_config() -> Config:
    """Configuración pequeña para pruebas rápidas en CPU."""
    return replace(
        Config(), epochs=1, num_samples=256, batch_size=64, g_features=8, d_features=8,
        sample_grid_size=16,
    )  # fmt: skip
