"""Prueba de humo (smoke test) del entrenamiento, además de inspección y reproducibilidad."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from src.config import Config  # noqa: E402
from src.data.dataset import QuickDrawDataset, create_dataloader  # noqa: E402
from src.models.discriminator import build_discriminator  # noqa: E402
from src.models.generator import build_generator  # noqa: E402
from src.training.history import TrainingHistory  # noqa: E402
from src.training.trainer import GANTrainer  # noqa: E402
from src.utils.inspect_models import count_parameters, layer_summary, weight_stats  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture
def trainer_setup(fake_npy: Path, small_config: Config, tmp_path: Path, monkeypatch):
    # Redirige las salidas (outputs/...) a una carpeta temporal: los tests no ensucian el proyecto.
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
    cfg = small_config
    set_seed(cfg.seed)
    ds = QuickDrawDataset(fake_npy, cfg.num_samples)
    loader = create_dataloader(ds, cfg.batch_size, cfg.seed)
    g, d = build_generator(cfg, CPU), build_discriminator(cfg, CPU)
    return GANTrainer(g, d, cfg, CPU), loader, cfg


def test_training_runs_and_records_finite_metrics(trainer_setup) -> None:
    trainer, loader, cfg = trainer_setup
    history = trainer.fit(loader)
    assert len(history.d_loss) == cfg.epochs
    assert len(history.iter_g_loss) == cfg.epochs * len(loader)
    assert all(math.isfinite(v) for v in history.iter_d_loss + history.iter_g_loss)
    assert all(0.0 <= p <= 1.0 for p in history.d_real_prob + history.d_fake_prob)


def test_training_updates_both_networks(trainer_setup) -> None:
    trainer, loader, _ = trainer_setup
    g_before = [p.clone() for p in trainer.generator.parameters()]
    d_before = [p.clone() for p in trainer.discriminator.parameters()]
    trainer.fit(loader)
    assert any(
        not torch.equal(a, b) for a, b in zip(g_before, trainer.generator.parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(d_before, trainer.discriminator.parameters(), strict=True)
    )


def test_training_saves_samples_and_checkpoint(trainer_setup) -> None:
    trainer, loader, cfg = trainer_setup
    trainer.fit(loader)
    trainer.save_checkpoint()
    assert (cfg.outputs_dir / "samples" / "epoch_000.png").exists()
    assert (cfg.outputs_dir / "samples" / f"epoch_{cfg.epochs:03d}.png").exists()
    ckpt = torch.load(cfg.checkpoint_path, weights_only=True)
    assert {"generator", "discriminator", "config"} <= set(ckpt)


def test_history_roundtrip(tmp_path: Path) -> None:
    h = TrainingHistory(d_loss=[1.0], g_loss=[2.0], d_real_prob=[0.9], d_fake_prob=[0.1])
    h.save(tmp_path / "h.json")
    assert TrainingHistory.load(tmp_path / "h.json") == h


def test_same_seed_gives_same_initial_weights() -> None:
    cfg = Config()
    set_seed(1)
    a = build_generator(cfg, CPU)
    set_seed(1)
    b = build_generator(cfg, CPU)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))


def test_inspection_helpers() -> None:
    cfg = Config()
    g = build_generator(cfg, CPU)
    trainable, total = count_parameters(g)
    assert trainable == total == 1_781_505
    layers = layer_summary(g, torch.zeros(2, cfg.latent_dim))
    assert layers[-1].layer_type == "Tanh"
    assert layers[-1].output_shape == (2, 1, 28, 28)
    assert any(s.name == "net.0.weight" for s in weight_stats(g))
