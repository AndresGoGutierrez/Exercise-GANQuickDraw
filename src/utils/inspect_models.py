"""Inspección de las redes: capas, parámetros y valores de los pesos.

Sirve para EXPLICAR los parámetros del Generador y del Discriminador:
  * Tabla de capas con la forma (shape) de la salida de cada una.
  * Nº de parámetros entrenables.
  * Estadísticas de los pesos (media, desviación, mínimo y máximo).
  * Valores que producen las redes (imágenes de G y probabilidades de D).

Uso desde la terminal (carga el checkpoint si existe; si no, muestra las redes sin entrenar):

    python -m src.utils.inspect_models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from src.config import Config, get_device
from src.models.discriminator import build_discriminator
from src.models.generator import build_generator
from src.utils.logger import setup_logging
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerInfo:
    name: str
    layer_type: str
    output_shape: tuple[int, ...]
    params: int


@dataclass(frozen=True)
class WeightStat:
    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    min: float
    max: float


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Devuelve (parámetros entrenables, parámetros totales)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


@torch.no_grad()
def layer_summary(model: nn.Module, example_input: torch.Tensor) -> list[LayerInfo]:
    """Recorre el modelo con una entrada de ejemplo y registra la salida de cada capa.

    Usa "hooks" de PyTorch: funciones que se disparan al pasar por cada capa hoja.
    """
    infos: list[LayerInfo] = []
    hooks = []

    def make_hook(name: str):
        def hook(module: nn.Module, _inputs, output: torch.Tensor) -> None:
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            infos.append(LayerInfo(name, module.__class__.__name__, tuple(output.shape), n_params))

        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # solo capas "hoja" (no contenedores)
            hooks.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()  # evita que Dropout/BatchNorm alteren el modelo durante la inspección
    try:
        model(example_input)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return infos


def weight_stats(model: nn.Module) -> list[WeightStat]:
    """Estadísticas de cada tensor de parámetros (pesos y sesgos)."""
    stats = []
    for name, p in model.named_parameters():
        d = p.detach()
        stats.append(
            WeightStat(
                name=name,
                shape=tuple(d.shape),
                mean=d.mean().item(),
                std=d.std().item() if d.numel() > 1 else 0.0,
                min=d.min().item(),
                max=d.max().item(),
            )
        )
    return stats


def format_model_report(title: str, model: nn.Module, example_input: torch.Tensor) -> str:
    """Construye un informe de texto con capas, parámetros y estadísticas de pesos."""
    trainable, total = count_parameters(model)
    lines = [
        "=" * 92,
        f"{title}",
        "=" * 92,
        f"{'Capa':<12}{'Tipo':<18}{'Forma de salida':<24}{'Parámetros':>12}",
        "-" * 92,
    ]
    for info in layer_summary(model, example_input):
        shape = str(list(info.output_shape))
        lines.append(f"{info.name:<12}{info.layer_type:<18}{shape:<24}{info.params:>12,}")
    lines += [
        "-" * 92,
        f"Parámetros entrenables: {trainable:,}   |   Totales: {total:,}",
        "",
        "Valores de los pesos (por tensor):",
        f"{'Tensor':<14}{'Forma':<22}{'Media':>10}{'Desv.':>10}{'Mín':>10}{'Máx':>10}",
        "-" * 92,
    ]
    for s in weight_stats(model):
        lines.append(
            f"{s.name:<14}{str(list(s.shape)):<22}{s.mean:>10.4f}{s.std:>10.4f}"
            f"{s.min:>10.4f}{s.max:>10.4f}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class OutputStats:
    """Valores que producen las redes: estadísticas de píxeles y probabilidades de D."""

    fake_mean: float
    fake_std: float
    fake_min: float
    fake_max: float
    real_mean: float
    real_std: float
    real_min: float
    real_max: float
    d_real: float  # D(x): probabilidad media que D asigna a imágenes REALES
    d_fake: float  # D(G(z)): probabilidad media que D asigna a imágenes FALSAS


@torch.no_grad()
def compute_output_stats(
    generator: nn.Module, discriminator: nn.Module, real: torch.Tensor, config: Config
) -> OutputStats:
    """Calcula qué valores producen las redes: imágenes de G y probabilidades de D."""
    device = real.device
    # G en modo evaluación (como al generar muestras finales).
    # D se evalúa en modo ENTRENAMIENTO a propósito: durante el entrenamiento D recibe lotes
    # solo-reales y lotes solo-falsos, y BatchNorm normaliza con las estadísticas de cada lote.
    # Así los valores D(x) y D(G(z)) de este informe son comparables con los del entrenamiento.
    generator.eval()
    z = torch.randn(real.size(0), config.latent_dim, device=device)
    fake = generator(z)
    p_real = torch.sigmoid(discriminator(real))
    p_fake = torch.sigmoid(discriminator(fake))
    generator.train()

    return OutputStats(
        fake_mean=fake.mean().item(),
        fake_std=fake.std().item(),
        fake_min=fake.min().item(),
        fake_max=fake.max().item(),
        real_mean=real.mean().item(),
        real_std=real.std().item(),
        real_min=real.min().item(),
        real_max=real.max().item(),
        d_real=p_real.mean().item(),
        d_fake=p_fake.mean().item(),
    )


def format_output_report(
    generator: nn.Module, discriminator: nn.Module, real: torch.Tensor, config: Config
) -> str:
    """Informe de texto con los valores que producen las redes."""
    s = compute_output_stats(generator, discriminator, real, config)
    return "\n".join(
        [
            "=" * 92,
            "VALORES DE SALIDA DE LAS REDES",
            "=" * 92,
            f"Generador     -> píxeles generados: media={s.fake_mean:.3f}  desv={s.fake_std:.3f}  "
            f"mín={s.fake_min:.3f}  máx={s.fake_max:.3f}   (rango válido [-1, 1] por la Tanh)",
            f"Imágenes reales -> píxeles:          media={s.real_mean:.3f}  desv={s.real_std:.3f}  "
            f"mín={s.real_min:.3f}  máx={s.real_max:.3f}",
            f"Discriminador -> D(x)    sobre imágenes REALES : media={s.d_real:.3f}",
            f"Discriminador -> D(G(z)) sobre imágenes FALSAS : media={s.d_fake:.3f}",
            "   (1.0 = 'seguro que es real'; 0.0 = 'seguro que es falsa'; 0.5 = no distingue)",
        ]
    )


def main() -> None:
    setup_logging()
    config = Config()
    device = get_device()
    set_seed(config.seed)

    generator = build_generator(config, device)
    discriminator = build_discriminator(config, device)

    if config.checkpoint_path.exists():
        ckpt = torch.load(config.checkpoint_path, map_location=device, weights_only=True)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        logger.info("Checkpoint cargado: redes ENTRENADAS (categoría '%s').", config.category)
    else:
        logger.info("Sin checkpoint: se muestran las redes SIN entrenar (pesos iniciales).")

    z = torch.randn(2, config.latent_dim, device=device)
    img = torch.zeros(2, 1, config.img_size, config.img_size, device=device)
    print(format_model_report("RED GENERADORA (G)", generator, z))
    print()
    print(format_model_report("RED DISCRIMINADORA (D)", discriminator, img))

    if config.data_path.exists():
        from src.data.dataset import QuickDrawDataset

        dataset = QuickDrawDataset(config.data_path, 256, config.img_size)
        real = dataset.images.to(device)
        print()
        print(format_output_report(generator, discriminator, real, config))


if __name__ == "__main__":
    main()
