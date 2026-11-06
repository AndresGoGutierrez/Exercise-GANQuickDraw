"""Configuración central del proyecto.

Todos los hiperparámetros viven aquí, en un único lugar. Así evitamos "números mágicos"
dispersos por el código y es muy fácil cambiar la categoría o los parámetros del taller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# Carpeta raíz del proyecto (dos niveles por encima de este archivo: src/config.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """Hiperparámetros del experimento (inmutable para evitar cambios accidentales)."""

    # ------------------------------------------------------------------ Datos
    category: str = "apple"  # Categoría de Quick, Draw! que se usará (ej.: "cat", "star").
    num_samples: int = 20_000  # Cuántos dibujos de la categoría se usan para entrenar.
    img_size: int = 28  # Las imágenes "numpy bitmap" de Quick, Draw! son de 28x28 píxeles.
    batch_size: int = 128

    # ------------------------------------------------------------ Arquitectura
    latent_dim: int = 100  # Tamaño del vector de ruido z que recibe el Generador.
    g_features: int = 64  # Ancho base (nº de canales) del Generador.
    d_features: int = 64  # Ancho base (nº de canales) del Discriminador.
    d_dropout: float = 0.3  # Dropout del Discriminador (ayuda a que no "gane" demasiado rápido).

    # ----------------------------------------------------------- Entrenamiento
    epochs: int = 30
    lr: float = 2e-4  # Tasa de aprendizaje (valor recomendado por el paper DCGAN).
    beta1: float = 0.5  # Parámetro beta1 de Adam (0.5 estabiliza el entrenamiento de GANs).
    beta2: float = 0.999
    seed: int = 42
    sample_grid_size: int = 64  # Nº de imágenes de la rejilla de muestras (8x8).

    # --------------------------------------------------------------- Rutas
    @property
    def data_dir(self) -> Path:
        """Carpeta donde se guardan los .npy descargados."""
        return PROJECT_ROOT / "data"

    @property
    def data_path(self) -> Path:
        """Ruta del archivo .npy de la categoría elegida."""
        return self.data_dir / f"{self.category.replace(' ', '_')}.npy"

    @property
    def outputs_dir(self) -> Path:
        """Carpeta de resultados (imágenes, gráficas, checkpoints) de esta categoría."""
        return PROJECT_ROOT / "outputs" / self.category.replace(" ", "_")

    @property
    def checkpoint_path(self) -> Path:
        return self.outputs_dir / "checkpoint.pt"


def get_device() -> torch.device:
    """Devuelve la GPU (CUDA) si está disponible; si no, la CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
