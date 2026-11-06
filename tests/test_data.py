"""Pruebas de la descarga (validación) y del Dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import QuickDrawDataset, create_dataloader
from src.data.downloader import build_url, download_category, validate_npy


def test_build_url_encodes_spaces() -> None:
    url = build_url("smiley face")
    assert url.endswith("/smiley%20face.npy")
    assert "numpy_bitmap" in url


def test_validate_npy_accepts_valid_file(fake_npy: Path) -> None:
    assert validate_npy(fake_npy) == 300


def test_validate_npy_rejects_wrong_shape(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((10, 100), dtype=np.uint8))
    with pytest.raises(ValueError, match="Forma inesperada"):
        validate_npy(bad)


def test_validate_npy_rejects_wrong_dtype(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((10, 784), dtype=np.float32))
    with pytest.raises(ValueError, match="Tipo inesperado"):
        validate_npy(bad)


def test_validate_npy_rejects_corrupt_file(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.npy"
    bad.write_bytes(b"esto no es un npy")
    with pytest.raises(ValueError, match="No se pudo leer"):
        validate_npy(bad)


def test_download_skips_when_file_already_valid(fake_npy: Path) -> None:
    """Si el archivo ya existe y es válido no debe tocar la red (no hay mock: fallaría)."""
    assert download_category("cualquiera", fake_npy) == fake_npy


def test_dataset_shape_and_range(fake_npy: Path) -> None:
    ds = QuickDrawDataset(fake_npy, num_samples=100)
    assert len(ds) == 100
    assert ds[0].shape == (1, 28, 28)
    assert ds.images.min() >= -1.0 and ds.images.max() <= 1.0


def test_dataset_limits_to_available_samples(fake_npy: Path) -> None:
    assert len(QuickDrawDataset(fake_npy, num_samples=10_000)) == 300


def test_dataset_rejects_invalid_num_samples(fake_npy: Path) -> None:
    with pytest.raises(ValueError):
        QuickDrawDataset(fake_npy, num_samples=0)


def test_dataloader_batches_and_drops_last(fake_npy: Path) -> None:
    ds = QuickDrawDataset(fake_npy, num_samples=300)
    loader = create_dataloader(ds, batch_size=128, seed=0)
    batches = list(loader)
    assert len(batches) == 2  # 300 // 128, descartando el lote incompleto
    assert batches[0].shape == (128, 1, 28, 28)
