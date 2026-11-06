"""Bucle de entrenamiento de la GAN.

En cada lote se alternan DOS pasos:

    1) Paso del DISCRIMINADOR: aprende a separar imágenes reales de falsas.
    2) Paso del GENERADOR:     aprende a fabricar imágenes que engañen al Discriminador.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import Config
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.training.history import TrainingHistory
from src.training.losses import discriminator_loss, generator_loss
from src.utils.visualization import save_image_grid, to_uint8_images

logger = logging.getLogger(__name__)


class GANTrainer:
    """Entrena un Generador y un Discriminador de forma adversaria."""

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        config: Config,
        device: torch.device,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.device = device
        self.history = TrainingHistory()

        # Un optimizador POR red: cada una actualiza solo sus propios parámetros.
        betas = (config.beta1, config.beta2)
        self.opt_g = Adam(generator.parameters(), lr=config.lr, betas=betas)
        self.opt_d = Adam(discriminator.parameters(), lr=config.lr, betas=betas)

        # Ruido FIJO para las muestras: así vemos cómo evoluciona LAS MISMAS "semillas"
        # época tras época (útil para el GIF de evolución).
        self.fixed_noise = torch.randn(config.sample_grid_size, config.latent_dim, device=device)

    # ------------------------------------------------------------------ pasos
    def _train_step(self, real: torch.Tensor) -> tuple[float, float, float, float]:
        """Ejecuta un lote. Devuelve (loss_D, loss_G, D(x), D(G(z)))."""
        batch_size = real.size(0)
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake = self.generator(z)  # imágenes falsas fabricadas por G

        # ---------- 1) Paso del DISCRIMINADOR ----------
        self.opt_d.zero_grad(set_to_none=True)
        real_logits = self.discriminator(real)
        # .detach() corta el gradiente hacia G: aquí SOLO se actualiza D.
        fake_logits = self.discriminator(fake.detach())
        d_loss = discriminator_loss(real_logits, fake_logits)
        d_loss.total.backward()
        self.opt_d.step()

        # ---------- 2) Paso del GENERADOR ----------
        self.opt_g.zero_grad(set_to_none=True)
        # Volvemos a pasar las falsas por D (ya actualizado); el gradiente ahora SÍ llega a G.
        g_logits = self.discriminator(fake)
        g_loss = generator_loss(g_logits)
        g_loss.backward()
        self.opt_g.step()

        with torch.no_grad():  # solo para registrar métricas, sin gradientes
            d_x = torch.sigmoid(real_logits).mean().item()
            d_gz = torch.sigmoid(fake_logits).mean().item()
        return d_loss.total.item(), g_loss.item(), d_x, d_gz

    @torch.no_grad()
    def generate_samples(self) -> torch.Tensor:
        """Genera la rejilla de muestras con el ruido fijo (G en modo evaluación)."""
        self.generator.eval()
        samples = self.generator(self.fixed_noise)
        self.generator.train()
        return samples

    def _save_samples(self, epoch: int) -> None:
        path = self.config.outputs_dir / "samples" / f"epoch_{epoch:03d}.png"
        title = f"Categoría: {self.config.category} | Época {epoch}/{self.config.epochs}"
        samples = self.generate_samples()
        save_image_grid(samples, title, path)
        # Copia cruda (uint8, ~50 KB) de las mismas muestras: la usa el informe web interactivo.
        np.save(path.with_suffix(".npy"), to_uint8_images(samples))

    # --------------------------------------------------------------- principal
    def fit(self, dataloader: DataLoader) -> TrainingHistory:
        """Entrena durante `config.epochs` épocas y devuelve el historial de métricas."""
        cfg = self.config
        self.generator.train()
        self.discriminator.train()
        self._save_samples(0)  # imagen inicial: lo que genera G SIN entrenar (ruido puro)

        for epoch in range(1, cfg.epochs + 1):
            sums = [0.0, 0.0, 0.0, 0.0]
            n_batches = 0
            for real in dataloader:
                real = real.to(self.device, non_blocking=True)
                d_l, g_l, d_x, d_gz = self._train_step(real)

                self.history.iter_d_loss.append(d_l)
                self.history.iter_g_loss.append(g_l)
                for i, value in enumerate((d_l, g_l, d_x, d_gz)):
                    sums[i] += value
                n_batches += 1

            d_l, g_l, d_x, d_gz = (s / n_batches for s in sums)
            self.history.d_loss.append(d_l)
            self.history.g_loss.append(g_l)
            self.history.d_real_prob.append(d_x)
            self.history.d_fake_prob.append(d_gz)

            logger.info(
                "Época %2d/%d | Loss D: %.3f | Loss G: %.3f | D(x): %.3f | D(G(z)): %.3f",
                epoch, cfg.epochs, d_l, g_l, d_x, d_gz,
            )  # fmt: skip
            self._save_samples(epoch)

        return self.history

    # ------------------------------------------------------------- persistencia
    def save_checkpoint(self) -> None:
        """Guarda los pesos de ambas redes y la configuración usada."""
        path = self.config.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )
        logger.info("Checkpoint guardado en '%s'", path)
