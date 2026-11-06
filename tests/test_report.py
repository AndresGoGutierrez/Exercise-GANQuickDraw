"""Pruebas del generador del informe web."""

from __future__ import annotations

import base64
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from src.config import Config
from src.report.build_report import (
    build_report_data,
    grid_png_b64,
    moving_average,
    render_html,
)


def test_moving_average_valid_mode() -> None:
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_moving_average_short_series_is_returned_unchanged() -> None:
    assert moving_average([1.0, 2.0], 5) == [1.0, 2.0]


def test_moving_average_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        moving_average([1.0], 0)


def test_grid_png_has_expected_size_and_dark_strokes() -> None:
    images = np.zeros((64, 28, 28), dtype=np.uint8)
    images[0, 10:12, 10:12] = 255  # un "trazo" (255 = trazo en el convenio de Quick, Draw!)
    png = Image.open(io.BytesIO(base64.b64decode(grid_png_b64(images, nrow=8, gap=2))))
    assert png.size == (8 * 28 + 9 * 2, 8 * 28 + 9 * 2)
    pixels = np.asarray(png)
    assert pixels.min() == 0  # el trazo se dibuja oscuro...
    assert pixels.max() == 255  # ...sobre fondo blanco


def test_render_html_embeds_data_and_escapes_script_end() -> None:
    html = render_html({"category": "a<b", "note": "</script><b>x</b>"})
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None  # los datos no cerraron la etiqueta <script> antes de tiempo
    assert json.loads(match.group(1))["note"] == "</script><b>x</b>"
    assert "a&lt;b" in html  # la categoría se escapa en el HTML


def test_build_report_data_requires_trained_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Entrena primero"):
        build_report_data(Config())
