"""Configuración de logging común para todo el proyecto."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el logging con un formato legible (se usa en lugar de print)."""
    # En Windows la consola puede no usar UTF-8; lo forzamos para mostrar tildes y eñes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
