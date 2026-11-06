"""Punto de entrada: descarga la categoría, crea las dos redes, entrena y guarda los resultados.

Uso:
    python -m src.train                          # categoría 'apple', 30 épocas
    python -m src.train --epochs 5               # prueba rápida
    python -m src.train --category star          # otra categoría de Quick, Draw!
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # backend sin ventanas: solo guarda archivos de imagen

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.config import Config, get_device  # noqa: E402
from src.data.dataset import QuickDrawDataset, create_dataloader  # noqa: E402
from src.data.downloader import download_category  # noqa: E402
from src.models.discriminator import build_discriminator  # noqa: E402
from src.models.generator import build_generator  # noqa: E402
from src.training.trainer import GANTrainer  # noqa: E402
from src.utils.inspect_models import (  # noqa: E402
    format_model_report,
    format_output_report,
)
from src.utils.logger import setup_logging  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    create_gif,
    plot_real_vs_fake,
    plot_training_curves,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una GAN con una categoría de Quick, Draw!"
    )
    parser.add_argument("--category", type=str, help="Categoría de Quick, Draw! (def.: apple)")
    parser.add_argument("--epochs", type=int, help="Número de épocas")
    parser.add_argument("--num-samples", type=int, help="Nº de dibujos usados para entrenar")
    parser.add_argument("--batch-size", type=int, help="Tamaño del lote")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Parte de la configuración por defecto y aplica solo los argumentos indicados."""
    overrides = {
        "category": args.category,
        "epochs": args.epochs,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
    }
    return replace(Config(), **{k: v for k, v in overrides.items() if v is not None})


def main() -> None:
    setup_logging()
    config = build_config(parse_args())
    device = get_device()
    set_seed(config.seed)
    logger.info("Categoría: '%s' | Dispositivo: %s", config.category, device)

    # 1) DATOS -----------------------------------------------------------------------
    download_category(config.category, config.data_path, config.img_size)
    dataset = QuickDrawDataset(config.data_path, config.num_samples, config.img_size)
    dataloader = create_dataloader(
        dataset, config.batch_size, config.seed, pin_memory=device.type == "cuda"
    )
    logger.info("Dataset: %d imágenes | %d lotes por época", len(dataset), len(dataloader))

    # 2) REDES NEURONALES ------------------------------------------------------------
    # >>> Aquí se CREAN (instancian) las dos redes. Ver src/models/generator.py y
    # >>> src/models/discriminator.py para la arquitectura y los comentarios de cada capa.
    generator = build_generator(config, device)  # <- RED GENERADORA
    discriminator = build_discriminator(config, device)  # <- RED DISCRIMINADORA

    # Informe de las redes ANTES de entrenar (pesos iniciales aleatorios).
    z = torch.zeros(2, config.latent_dim, device=device)
    img = torch.zeros(2, 1, config.img_size, config.img_size, device=device)
    logger.info("\n%s", format_model_report("GENERADOR (sin entrenar)", generator, z))
    logger.info("\n%s", format_model_report("DISCRIMINADOR (sin entrenar)", discriminator, img))

    # 3) ENTRENAMIENTO ---------------------------------------------------------------
    trainer = GANTrainer(generator, discriminator, config, device)
    history = trainer.fit(dataloader)
    trainer.save_checkpoint()
    history.save(config.outputs_dir / "history.json")

    # 4) RESULTADOS ------------------------------------------------------------------
    out = config.outputs_dir
    real_batch = dataset.images[: config.sample_grid_size].to(device)
    logger.info("\n%s", format_model_report("GENERADOR (entrenado)", generator, z))
    logger.info("\n%s", format_model_report("DISCRIMINADOR (entrenado)", discriminator, img))
    logger.info("\n%s", format_output_report(generator, discriminator, real_batch, config))

    fig = plot_training_curves(history, config.category)
    fig.savefig(out / "training_curves.png", dpi=110)
    plt.close(fig)

    fake_batch = trainer.generate_samples()
    fig = plot_real_vs_fake(real_batch, fake_batch, config.category)
    fig.savefig(out / "real_vs_generated.png", dpi=110)
    plt.close(fig)

    frames = sorted((out / "samples").glob("epoch_*.png"))
    create_gif(frames, out / "evolution.gif")
    logger.info("Listo. Resultados de la categoría '%s' en: %s", config.category, out)


if __name__ == "__main__":
    main()
