"""RED NEURONAL DISCRIMINADORA (D).

Objetivo: recibir una imagen de 28x28 y decidir si es REAL (viene del dataset Quick, Draw!) o
FALSA (la fabricó el Generador).  Es un clasificador binario.

Importante: la red devuelve un LOGIT (un número real sin acotar), NO una probabilidad.
La función de pérdida `BCEWithLogitsLoss` aplica la sigmoide internamente, lo cual es
numéricamente más estable.  Para obtener la probabilidad usa `torch.sigmoid(logit)`.

Recorrido de las dimensiones (con features=64):

    imagen         (B,   1, 28, 28)
    Conv 4x4 s2    (B,  64, 14, 14)
    Conv 4x4 s2    (B, 128,  7,  7)
    Conv 7x7       (B,   1,  1,  1)
    flatten        (B,)             <- un logit por imagen
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import Config
from src.models.generator import init_weights


class Discriminator(nn.Module):
    """Discriminador de la GAN (arquitectura tipo DCGAN, adaptada a imágenes de 28x28)."""

    def __init__(self, features: int = 64, dropout: float = 0.3) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # ---- Bloque 1: (B, 1, 28, 28) -> (B, features, 14, 14)
            # Conv2d con stride=2 REDUCE a la mitad el alto y ancho (sustituye al pooling).
            #   kernel_size=4, stride=2, padding=1
            nn.Conv2d(1, features, 4, 2, 1, bias=False),
            # LeakyReLU deja pasar una pequeña pendiente (0.2) para valores negativos.
            # Evita "neuronas muertas" y es la activación estándar del Discriminador.
            nn.LeakyReLU(0.2, inplace=True),
            # ---- Bloque 2: (B, features, 14, 14) -> (B, features*2, 7, 7)
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Dropout apaga al azar un % de neuronas en entrenamiento. Debilita un poco a D
            # para que no aprenda más rápido que G (si D gana siempre, G deja de aprender).
            nn.Dropout(dropout),
            # ---- Bloque 3 (salida): (B, features*2, 7, 7) -> (B, 1, 1, 1)
            # Un kernel de 7x7 "resume" todo el mapa en un único número: el logit.
            nn.Conv2d(features * 2, 1, 7, 1, 0),
            # NO hay Sigmoid: se aplica dentro de la función de pérdida (ver losses.py).
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Devuelve un logit por imagen, con forma (B,).  Valores altos => 'parece real'."""
        return self.net(image).view(-1)


def build_discriminator(config: Config, device: torch.device) -> Discriminator:
    """Crea (instancia) la red Discriminadora, inicializa sus pesos y la mueve al dispositivo."""
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>  AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA                                   <<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    discriminator = Discriminator(features=config.d_features, dropout=config.d_dropout)
    discriminator.apply(init_weights)
    return discriminator.to(device)
