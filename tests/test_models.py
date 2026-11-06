"""Pruebas de las redes Generadora y Discriminadora."""

from __future__ import annotations

import torch

from src.config import Config
from src.models.discriminator import Discriminator, build_discriminator
from src.models.generator import Generator, build_generator

CPU = torch.device("cpu")


def test_generator_output_shape_and_range() -> None:
    g = Generator(latent_dim=100, features=8)
    out = g(torch.randn(5, 100))
    assert out.shape == (5, 1, 28, 28)
    # La Tanh final debe dejar los píxeles en [-1, 1].
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_discriminator_outputs_one_logit_per_image() -> None:
    d = Discriminator(features=8)
    out = d(torch.randn(7, 1, 28, 28))
    assert out.shape == (7,)


def test_build_functions_return_models_with_gradients() -> None:
    cfg = Config()
    g = build_generator(cfg, CPU)
    d = build_discriminator(cfg, CPU)
    assert all(p.requires_grad for p in g.parameters())
    assert all(p.requires_grad for p in d.parameters())


def test_weights_initialised_with_small_std() -> None:
    """Los pesos convolucionales deben seguir ~Normal(0, 0.02) (paper DCGAN)."""
    g = build_generator(Config(), CPU)
    weight = g.net[0].weight
    assert abs(weight.mean().item()) < 0.005
    assert 0.015 < weight.std().item() < 0.025


def test_gradients_flow_from_discriminator_to_generator() -> None:
    g = Generator(latent_dim=100, features=8)
    d = Discriminator(features=8)
    d(g(torch.randn(4, 100))).sum().backward()
    assert all(p.grad is not None for p in g.parameters())
