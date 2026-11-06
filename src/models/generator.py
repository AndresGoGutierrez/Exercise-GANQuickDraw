"""RED NEURONAL GENERADORA (G).

Objetivo: transformar un vector de ruido aleatorio `z` en una imagen de 28x28 que parezca un
dibujo real de Quick, Draw!.  G nunca ve los dibujos reales directamente: aprende solo a
partir de la señal que le devuelve el Discriminador.

Recorrido de las dimensiones (con latent_dim=100 y g_features=64):

    z            (B, 100)
    reshape      (B, 100, 1, 1)
    ConvT 7x7    (B, 256, 7, 7)      <- "siembra" un mapa de 7x7 con 256 canales
    ConvT 4x4 s2 (B, 128, 14, 14)    <- duplica la resolución
    ConvT 4x4 s2 (B,   1, 28, 28)    <- duplica otra vez y deja 1 canal (escala de grises)
    Tanh         valores en [-1, 1]
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import Config


class Generator(nn.Module):
    """Generador de la GAN (arquitectura tipo DCGAN, adaptada a imágenes de 28x28)."""

    def __init__(self, latent_dim: int = 100, features: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # nn.Sequential encadena las capas en el orden en que se declaran.
        self.net = nn.Sequential(
            # ---- Bloque 1: (B, latent_dim, 1, 1) -> (B, features*4, 7, 7)
            # ConvTranspose2d es la "convolución inversa": en vez de reducir, AMPLÍA.
            #   in_channels=latent_dim : cada componente de z es un canal de entrada
            #   out_channels=features*4: nº de mapas de características que se crean
            #   kernel_size=7, stride=1, padding=0: convierte 1x1 en 7x7
            #   bias=False: el BatchNorm siguiente ya añade su propio desplazamiento
            nn.ConvTranspose2d(latent_dim, features * 4, 7, 1, 0, bias=False),
            # BatchNorm normaliza las activaciones: estabiliza y acelera el entrenamiento.
            nn.BatchNorm2d(features * 4),
            # ReLU: activación no lineal (deja pasar solo valores positivos).
            nn.ReLU(inplace=True),
            # ---- Bloque 2: (B, features*4, 7, 7) -> (B, features*2, 14, 14)
            #   kernel=4, stride=2, padding=1  =>  alto y ancho se multiplican por 2
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
            # ---- Bloque 3 (salida): (B, features*2, 14, 14) -> (B, 1, 28, 28)
            # Sin BatchNorm en la capa de salida (recomendación del paper DCGAN).
            nn.ConvTranspose2d(features * 2, 1, 4, 2, 1),
            # Tanh fuerza la salida al rango [-1, 1], igual que las imágenes reales normalizadas.
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Convierte ruido `z` de forma (B, latent_dim) en imágenes (B, 1, 28, 28)."""
        # Las capas convolucionales esperan 4 dimensiones: (lote, canales, alto, ancho).
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


def init_weights(module: nn.Module) -> None:
    """Inicialización de pesos recomendada por el paper DCGAN: Normal(media=0, std=0.02).

    Partir de pesos pequeños y centrados en 0 evita que la red arranque saturada.
    """
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def build_generator(config: Config, device: torch.device) -> Generator:
    """Crea (instancia) la red Generadora, inicializa sus pesos y la mueve al dispositivo."""
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>  AQUÍ SE CREA LA RED NEURONAL GENERADORA                                       <<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    generator = Generator(latent_dim=config.latent_dim, features=config.g_features)
    generator.apply(init_weights)  # aplica init_weights a cada capa
    return generator.to(device)
