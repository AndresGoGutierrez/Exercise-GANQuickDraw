"""Descarga de UNA sola categoría del dataset Quick, Draw! de Google.

Repositorio oficial: https://github.com/googlecreativelab/quickdraw-dataset
Los archivos "numpy bitmap" (imágenes de 28x28 en escala de grises) están publicados en un
bucket público de Google Cloud Storage, uno por categoría:

    https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<categoria>.npy
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
CHUNK_SIZE = 1024 * 256  # 256 KB por bloque de descarga
REQUEST_TIMEOUT = 60  # segundos


def build_url(category: str) -> str:
    """Construye la URL de descarga (codifica espacios, ej.: 'smiley face')."""
    return f"{BASE_URL}/{quote(category)}.npy"


def validate_npy(path: Path, img_size: int = 28) -> int:
    """Comprueba que el .npy tenga el formato esperado (N, img_size*img_size) de uint8.

    Devuelve el número de imágenes. Lanza ValueError si el archivo no es válido.
    """
    try:
        data = np.load(path, mmap_mode="r")
    except Exception as exc:  # archivo truncado, corrupto, etc.
        raise ValueError(f"No se pudo leer '{path}': {exc}") from exc

    expected_cols = img_size * img_size
    if data.ndim != 2 or data.shape[1] != expected_cols:
        raise ValueError(
            f"Forma inesperada {data.shape}; se esperaba (N, {expected_cols}) en '{path}'."
        )
    if data.dtype != np.uint8:
        raise ValueError(f"Tipo inesperado {data.dtype}; se esperaba uint8 en '{path}'.")
    return int(data.shape[0])


def download_category(category: str, dest_path: Path, img_size: int = 28) -> Path:
    """Descarga la categoría indicada en `dest_path` (si no existe ya) y la valida.

    Args:
        category: nombre de la categoría de Quick, Draw! (ej.: "apple").
        dest_path: ruta final del archivo .npy.
        img_size: lado de la imagen, usado para validar el archivo.

    Returns:
        La ruta del archivo descargado.
    """
    if dest_path.exists():
        try:
            n = validate_npy(dest_path, img_size)
            logger.info("La categoría '%s' ya está descargada (%d imágenes).", category, n)
            return dest_path
        except ValueError:
            logger.warning("El archivo existente es inválido; se descargará de nuevo.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = build_url(category)
    logger.info("Descargando '%s' desde %s", category, url)

    # Se descarga a un archivo temporal y se renombra al final: así nunca queda un .npy a medias.
    tmp_path = dest_path.with_suffix(".npy.part")
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            if response.status_code == 404:
                raise ValueError(
                    f"La categoría '{category}' no existe en Quick, Draw!. "
                    "Revisa la lista oficial de categorías del repositorio."
                )
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with (
                open(tmp_path, "wb") as file,
                tqdm(total=total, unit="B", unit_scale=True, desc=category) as bar,
            ):
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    file.write(chunk)
                    bar.update(len(chunk))

        n = validate_npy(tmp_path, img_size)
        tmp_path.replace(dest_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info("Descarga completada: %d imágenes en '%s'.", n, dest_path)
    return dest_path
