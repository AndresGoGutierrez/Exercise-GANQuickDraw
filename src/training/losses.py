"""FUNCIONES DE PÉRDIDA de la GAN.

Una GAN es un juego de dos jugadores (minimax):

    min_G max_D  V(D, G) = E_x[ log D(x) ] + E_z[ log(1 - D(G(z))) ]

donde  x ~ datos reales,  z ~ ruido,  D(.) = probabilidad de que la imagen sea real.

Usamos la entropía cruzada binaria (BCE):    BCE(p, y) = -[ y*log(p) + (1-y)*log(1-p) ]
con etiquetas  real = 1  y  falso = 0.

* Pérdida del DISCRIMINADOR  (quiere D(x)->1 y D(G(z))->0):
      L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
          = -log D(x)  -  log(1 - D(G(z)))

* Pérdida del GENERADOR  (quiere engañar a D, o sea D(G(z))->1):
      L_G = BCE(D(G(z)), 1) = -log D(G(z))
  Es la versión "no saturante" propuesta en el paper original: da gradientes fuertes al
  principio, cuando G todavía es malo y D lo detecta fácilmente.

Referencia de valores: si D no distingue nada (D = 0.5 siempre),
    L_D = 2*ln(2) ~ 1.386   y   L_G = ln(2) ~ 0.693   (equilibrio ideal de la GAN).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

# BCEWithLogitsLoss = Sigmoide + BCE en una sola operación (más estable numéricamente).
_bce = nn.BCEWithLogitsLoss()

REAL_LABEL = 1.0
FAKE_LABEL = 0.0


class DiscriminatorLoss(NamedTuple):
    """Desglose de la pérdida del Discriminador (útil para analizar el entrenamiento)."""

    total: torch.Tensor  # pérdida total que se retropropaga
    real: torch.Tensor  # parte de imágenes reales:  -log D(x)
    fake: torch.Tensor  # parte de imágenes falsas:  -log(1 - D(G(z)))


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> DiscriminatorLoss:
    """Pérdida del Discriminador: debe clasificar reales como 1 y falsas como 0."""
    real_loss = _bce(real_logits, torch.full_like(real_logits, REAL_LABEL))
    fake_loss = _bce(fake_logits, torch.full_like(fake_logits, FAKE_LABEL))
    return DiscriminatorLoss(total=real_loss + fake_loss, real=real_loss, fake=fake_loss)


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Pérdida del Generador: quiere que D clasifique sus imágenes falsas como REALES (1)."""
    return _bce(fake_logits, torch.full_like(fake_logits, REAL_LABEL))
