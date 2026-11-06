"""Registro (historial) de las métricas del entrenamiento."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainingHistory:
    """Guarda las métricas del entrenamiento, por iteración y por época.

    Glosario:
        D(x)     = probabilidad (0-1) que da el Discriminador a una imagen REAL.
        D(G(z))  = probabilidad (0-1) que da el Discriminador a una imagen FALSA.
    """

    # --- Por iteración (cada lote)
    iter_d_loss: list[float] = field(default_factory=list)
    iter_g_loss: list[float] = field(default_factory=list)

    # --- Por época (medias de todos los lotes de la época)
    d_loss: list[float] = field(default_factory=list)
    g_loss: list[float] = field(default_factory=list)
    d_real_prob: list[float] = field(default_factory=list)  # D(x)
    d_fake_prob: list[float] = field(default_factory=list)  # D(G(z))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrainingHistory:
        return cls(**json.loads(path.read_text(encoding="utf-8")))
