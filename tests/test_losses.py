"""Pruebas de las funciones de pérdida con valores teóricos conocidos."""

from __future__ import annotations

import math

import torch

from src.training.losses import discriminator_loss, generator_loss


def test_equilibrium_values() -> None:
    """Si D no distingue (logit 0 => prob 0.5): L_D = 2*ln2 y L_G = ln2."""
    logits = torch.zeros(10)
    d = discriminator_loss(logits, logits)
    assert math.isclose(d.total.item(), 2 * math.log(2), rel_tol=1e-5)
    assert math.isclose(generator_loss(logits).item(), math.log(2), rel_tol=1e-5)


def test_perfect_discriminator_has_near_zero_loss() -> None:
    real_logits = torch.full((10,), 20.0)  # D está seguro de que son reales
    fake_logits = torch.full((10,), -20.0)  # y seguro de que las falsas son falsas
    assert discriminator_loss(real_logits, fake_logits).total.item() < 1e-6


def test_generator_loss_is_high_when_discriminator_wins() -> None:
    assert generator_loss(torch.full((10,), -10.0)).item() > 5.0


def test_generator_loss_is_low_when_generator_fools_discriminator() -> None:
    assert generator_loss(torch.full((10,), 10.0)).item() < 1e-3


def test_discriminator_loss_total_is_sum_of_parts() -> None:
    d = discriminator_loss(torch.randn(8), torch.randn(8))
    assert torch.isclose(d.total, d.real + d.fake)
