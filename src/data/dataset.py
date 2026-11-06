"""Dataset de PyTorch para una categoría de Quick, Draw!."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class QuickDrawDataset(Dataset):
    """Carga los dibujos de UNA categoría y los normaliza al rango [-1, 1].

    ¿Por qué [-1, 1]? Porque la última capa del Generador usa la función Tanh, cuya salida
    está en ese rango. Real y falso deben vivir en la misma escala para que el
    Discriminador compare "peras con peras".
    """

    def __init__(self, npy_path: Path, num_samples: int, img_size: int = 28) -> None:
        raw = np.load(npy_path, mmap_mode="r")  # forma (N, 784), uint8 en [0, 255]
        if raw.ndim != 2 or raw.shape[1] != img_size * img_size:
            raise ValueError(f"Forma inesperada {raw.shape} en '{npy_path}'.")
        if num_samples <= 0:
            raise ValueError("num_samples debe ser mayor que 0.")

        n = min(num_samples, raw.shape[0])  # nos quedamos con las primeras n imágenes
        pixels = np.array(raw[:n], dtype=np.float32)

        # [0, 255] -> [-1, 1]
        pixels = pixels / 127.5 - 1.0
        # (N, 784) -> (N, 1, 28, 28): 1 canal (escala de grises)
        self.images = torch.from_numpy(pixels).view(n, 1, img_size, img_size)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        # En una GAN no necesitamos etiquetas: solo hay una categoría.
        return self.images[index]


def create_dataloader(
    dataset: Dataset, batch_size: int, seed: int, pin_memory: bool = False
) -> DataLoader:
    """Crea el DataLoader.

    - shuffle=True: mezcla los datos en cada época.
    - drop_last=True: descarta el último lote incompleto (BatchNorm es más estable así).
    - num_workers=0: el dataset ya está en memoria, no hace falta paralelizar (y en Windows
      evita problemas con multiprocesamiento).
    """
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )
