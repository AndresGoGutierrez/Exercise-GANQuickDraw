"""Funciones de visualización: rejillas de imágenes, curvas de pérdida y GIF de evolución.

Todas las figuras incluyen el nombre de la CATEGORÍA usada en el título.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from src.training.history import TrainingHistory


def _to_grid(images: torch.Tensor, nrow: int) -> np.ndarray:
    """Une un lote (N, 1, H, W) en [-1, 1] en una sola imagen 2D en [0, 1]."""
    imgs = images.detach().cpu().squeeze(1).numpy()  # (N, H, W)
    imgs = (imgs + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    n, h, w = imgs.shape
    ncol = int(np.ceil(n / nrow))
    grid = np.zeros((ncol * h, nrow * w), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, nrow)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = imgs[i]
    return np.clip(grid, 0.0, 1.0)


def to_uint8_images(images: torch.Tensor) -> np.ndarray:
    """Convierte un lote (N, 1, H, W) en [-1, 1] a un array (N, H, W) uint8 en [0, 255].

    En el resultado el trazo vale 255 y el fondo 0 (el mismo convenio que Quick, Draw!).
    """
    imgs = (images.detach().cpu().squeeze(1).numpy() + 1.0) / 2.0
    return np.round(np.clip(imgs, 0.0, 1.0) * 255).astype(np.uint8)


def plot_image_grid(images: torch.Tensor, title: str, nrow: int = 8) -> Figure:
    """Devuelve una figura con una rejilla de imágenes y el título indicado."""
    fig, ax = plt.subplots(figsize=(6, 6.4))
    ax.imshow(_to_grid(images, nrow), cmap="gray_r", vmin=0, vmax=1)  # trazo oscuro, fondo claro
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def save_image_grid(images: torch.Tensor, title: str, path: Path, nrow: int = 8) -> None:
    """Guarda en `path` la rejilla de imágenes (crea la carpeta si no existe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_image_grid(images, title, nrow)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_real_vs_fake(real: torch.Tensor, fake: torch.Tensor, category: str) -> Figure:
    """Compara lado a lado dibujos REALES y dibujos GENERADOS por la GAN."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    for ax, imgs, label in zip(
        axes, (real, fake), ("Reales (Quick, Draw!)", "Generados (GAN)"), strict=True
    ):
        ax.imshow(_to_grid(imgs, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    fig.suptitle(f"Categoría: {category}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _moving_average(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_training_curves(history: TrainingHistory, category: str) -> Figure:
    """Dibuja las pérdidas de G y D y las probabilidades D(x) y D(G(z)) durante el entrenamiento."""
    fig, (ax_loss, ax_prob) = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Panel 1: funciones de pérdida (por iteración, suavizadas)
    window = 50
    ax_loss.plot(_moving_average(history.iter_d_loss, window), label="Pérdida Discriminador (D)")
    ax_loss.plot(_moving_average(history.iter_g_loss, window), label="Pérdida Generador (G)")
    ax_loss.axhline(2 * np.log(2), color="gray", ls=":", lw=1, label="D en equilibrio (2·ln2)")
    ax_loss.set_title("Funciones de pérdida")
    ax_loss.set_xlabel(f"Iteración (media móvil de {window})")
    ax_loss.set_ylabel("Pérdida (BCE)")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # --- Panel 2: lo que "cree" el discriminador, por época
    epochs = np.arange(1, len(history.d_real_prob) + 1)
    ax_prob.plot(epochs, history.d_real_prob, label="D(x): prob. de real en imágenes REALES")
    ax_prob.plot(epochs, history.d_fake_prob, label="D(G(z)): prob. de real en imágenes FALSAS")
    ax_prob.axhline(0.5, color="gray", ls=":", lw=1, label="0.5 = D no distingue")
    ax_prob.set_ylim(0, 1)
    ax_prob.set_title("Salidas del Discriminador")
    ax_prob.set_xlabel("Época")
    ax_prob.set_ylabel("Probabilidad")
    ax_prob.legend(fontsize=8)
    ax_prob.grid(alpha=0.3)

    fig.suptitle(f"Entrenamiento de la GAN - Categoría: {category}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def create_gif(image_paths: Sequence[Path], output_path: Path, frame_ms: int = 250) -> None:
    """Crea un GIF animado con la evolución de las muestras generadas época a época."""
    if not image_paths:
        return
    frames = [Image.open(p).convert("P") for p in image_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
    )
