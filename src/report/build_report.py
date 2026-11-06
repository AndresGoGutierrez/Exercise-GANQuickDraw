"""Genera un INFORME WEB interactivo (un único archivo .html) con los resultados del taller.

El informe incluye: evolución de la GAN por épocas (deslizante), dibujos reales vs generados,
las dos redes neuronales con sus parámetros, funciones de pérdida, curvas de entrenamiento y los
valores de salida de G y D.  Es autocontenido: se abre con doble clic, sin Python ni Jupyter.

Requisito: haber entrenado antes la categoría (python -m src.train).

Uso:
    python -m src.report.build_report                    # -> docs/informe_apple.html
    python -m src.report.build_report --category star    # otra categoría ya entrenada
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import PROJECT_ROOT, Config
from src.data.dataset import QuickDrawDataset
from src.models.discriminator import build_discriminator
from src.models.generator import build_generator
from src.training.history import TrainingHistory
from src.training.losses import discriminator_loss, generator_loss
from src.utils.inspect_models import (
    compute_output_stats,
    count_parameters,
    layer_summary,
    weight_stats,
)
from src.utils.logger import setup_logging
from src.utils.reproducibility import set_seed
from src.utils.visualization import to_uint8_images

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).with_name("template.html")
SMOOTH_WINDOW = 50  # media móvil de las pérdidas por iteración
CHART_STRIDE = 4  # se envía 1 de cada 4 puntos suavizados (basta para dibujar la curva)
CPU = torch.device("cpu")


# ----------------------------------------------------------------------------- utilidades
def moving_average(values: list[float], window: int) -> list[float]:
    """Media móvil 'valid': devuelve len(values) - window + 1 puntos."""
    if window < 1:
        raise ValueError("window debe ser >= 1")
    if len(values) < window:
        return list(values)
    arr = np.asarray(values, dtype=np.float64)
    return np.convolve(arr, np.ones(window) / window, mode="valid").tolist()


def grid_png_b64(images: np.ndarray, nrow: int = 8, gap: int = 2) -> str:
    """Une imágenes uint8 (N, H, W) en una rejilla PNG (base64): trazo oscuro sobre fondo blanco."""
    n, h, w = images.shape
    ncol = int(np.ceil(n / nrow))
    canvas = np.zeros((ncol * h + (ncol + 1) * gap, nrow * w + (nrow + 1) * gap), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, nrow)
        y, x = gap + r * (h + gap), gap + c * (w + gap)
        canvas[y : y + h, x : x + w] = images[i]
    buffer = io.BytesIO()
    Image.fromarray(255 - canvas).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _round(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def _shape_text(shape: tuple[int, ...]) -> str:
    """(2, 256, 7, 7) -> '256×7×7' (se quita la dimensión del lote)."""
    return "×".join(str(s) for s in shape[1:])


# ------------------------------------------------------------------------------- datos
def _network_data(model: torch.nn.Module, example: torch.Tensor, untrained: torch.nn.Module,
                  source_file: str, factory: str) -> dict:  # fmt: skip
    trainable, _ = count_parameters(model)
    before = {s.name: s for s in weight_stats(untrained)}
    return {
        "params": trainable,
        "file": source_file,
        "factory": factory,
        "layers": [
            {
                "type": info.layer_type,
                "shape": _shape_text(info.output_shape),
                "params": info.params,
            }
            for info in layer_summary(model, example)
        ],
        "weights": [
            {
                "name": s.name,
                "shape": "×".join(str(d) for d in s.shape),
                "mean0": round(before[s.name].mean, 4),
                "std0": round(before[s.name].std, 4),
                "mean": round(s.mean, 4),
                "std": round(s.std, 4),
                "min": round(s.min, 4),
                "max": round(s.max, 4),
            }
            for s in weight_stats(model)
        ],
    }


def _loss_reference_rows() -> list[dict]:
    """Valores de las pérdidas en situaciones conocidas, calculados con las funciones REALES."""
    scenarios = {
        "D no distingue (equilibrio ideal)": (0.0, 0.0),
        "D gana: reales bien detectadas, falsas descubiertas": (5.0, -5.0),
        "G engaña a D: las falsas parecen reales": (5.0, 5.0),
    }
    rows = []
    for label, (real_logit, fake_logit) in scenarios.items():
        real, fake = torch.full((8,), real_logit), torch.full((8,), fake_logit)
        rows.append(
            {
                "label": label,
                "d": round(discriminator_loss(real, fake).total.item(), 3),
                "g": round(generator_loss(fake).item(), 3),
            }
        )
    return rows


def build_report_data(config: Config) -> dict:
    """Reúne TODOS los datos del informe a partir de los resultados guardados en outputs/."""
    out = config.outputs_dir
    history_path = out / "history.json"
    frame_paths = sorted((out / "samples").glob("epoch_*.npy"))
    if not (history_path.exists() and config.checkpoint_path.exists() and frame_paths):
        raise FileNotFoundError(
            f"Faltan resultados de '{config.category}' en {out}. "
            f"Entrena primero con: python -m src.train --category {config.category}"
        )
    if not config.data_path.exists():
        raise FileNotFoundError(f"Falta el dataset {config.data_path}; ejecuta el entrenamiento.")

    history = TrainingHistory.load(history_path)
    n_batches = len(history.iter_d_loss) // len(history.d_loss)

    # --- Imágenes: evolución por época (G) y muestra de dibujos reales
    frames = [grid_png_b64(np.load(p)) for p in frame_paths]
    dataset = QuickDrawDataset(config.data_path, 256, config.img_size)
    real_batch = dataset.images[: config.sample_grid_size]
    real_grid = grid_png_b64(to_uint8_images(real_batch))

    # --- Curvas por iteración: media móvil + submuestreo; el eje X se expresa en ÉPOCAS
    smooth_d = moving_average(history.iter_d_loss, SMOOTH_WINDOW)[::CHART_STRIDE]
    smooth_g = moving_average(history.iter_g_loss, SMOOTH_WINDOW)[::CHART_STRIDE]
    offset = min(SMOOTH_WINDOW, len(history.iter_d_loss))
    xs = [(offset + i * CHART_STRIDE) / n_batches for i in range(len(smooth_d))]

    # --- Redes: las ya entrenadas (checkpoint) y las iniciales (misma semilla que train.py)
    set_seed(config.seed)
    g0, d0 = build_generator(config, CPU), build_discriminator(config, CPU)
    generator, discriminator = build_generator(config, CPU), build_discriminator(config, CPU)
    checkpoint = torch.load(config.checkpoint_path, map_location=CPU, weights_only=True)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    z = torch.zeros(2, config.latent_dim)
    img = torch.zeros(2, 1, config.img_size, config.img_size)
    torch.manual_seed(config.seed)
    stats = compute_output_stats(generator, discriminator, dataset.images, config)

    return {
        "category": config.category,
        "generated_on": datetime.now().strftime("%d/%m/%Y"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "config": {
            **asdict(config),
            "n_batches": n_batches,
            "dataset_total": int(np.load(config.data_path, mmap_mode="r").shape[0]),
        },
        "frames": frames,
        "real_grid": real_grid,
        "epochs": {
            "d_loss": _round(history.d_loss),
            "g_loss": _round(history.g_loss),
            "d_real": _round(history.d_real_prob),
            "d_fake": _round(history.d_fake_prob),
        },
        "iter": {
            "x": _round(xs),
            "d": _round(smooth_d),
            "g": _round(smooth_g),
            "window": SMOOTH_WINDOW,
        },
        "networks": {
            "G": _network_data(generator, z, g0, "src/models/generator.py", "build_generator"),
            "D": _network_data(
                discriminator, img, d0, "src/models/discriminator.py", "build_discriminator"
            ),
        },
        "outputs": asdict(stats),
        "loss_reference": _loss_reference_rows(),
    }


def render_html(data: dict) -> str:
    """Inserta los datos (JSON) en la plantilla HTML."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    # "</" se escapa para que ningún dato pueda cerrar prematuramente la etiqueta <script>.
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("__REPORT_DATA__", payload).replace(
        "__CATEGORY__", data["category"].replace("&", "&amp;").replace("<", "&lt;")
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Genera el informe web de una categoría.")
    parser.add_argument("--category", default=None, help="Categoría entrenada (def.: apple)")
    parser.add_argument("--output", type=Path, default=None, help="Archivo .html de salida")
    args = parser.parse_args()

    config = replace(Config(), category=args.category) if args.category else Config()
    output = (
        args.output or PROJECT_ROOT / "docs" / f"informe_{config.category.replace(' ', '_')}.html"
    )

    html = render_html(build_report_data(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("Informe generado: %s (%.1f MB)", output, output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
