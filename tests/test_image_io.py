"""Tests for image loading utilities."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest

from src import image_io


def _make_test_image(width: int = 8, height: int = 8) -> np.ndarray:
    grid = np.linspace(0, 255, num=width * height, dtype=np.uint8).reshape((height, width))
    return cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)


def test_load_png_from_path(tmp_path):
    img = _make_test_image()
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)

    loaded = image_io.load_png(path)

    assert loaded.shape == img.shape
    assert loaded.dtype == img.dtype


def test_load_png_from_memory(tmp_path):
    img = _make_test_image()
    success, buffer = cv2.imencode(".png", img)
    assert success

    loaded = image_io.load_png(buffer.tobytes())
    assert np.array_equal(loaded, img)


def test_load_png_from_file_like(tmp_path):
    img = _make_test_image()
    success, buffer = cv2.imencode(".png", img)
    assert success

    handle = BytesIO(buffer.tobytes())
    loaded = image_io.load_png(handle)

    assert np.array_equal(loaded, img)


def test_load_png_invalid_source_type():
    with pytest.raises(TypeError):
        image_io.load_png(object())  # type: ignore[arg-type]


def test_validate_image_metadata():
    img = _make_test_image()
    metadata = image_io.validate_image(img)

    assert metadata.width == img.shape[1]
    assert metadata.height == img.shape[0]
    assert metadata.channels == 3
    assert metadata.dtype == str(img.dtype)
    assert metadata.max_intensity >= metadata.min_intensity


def test_validate_image_rejects_invalid_arrays():
    with pytest.raises(ValueError):
        image_io.validate_image(np.array([]))


def test_convert_color_to_grayscale():
    img = _make_test_image()
    gray = image_io.convert_color(img, "grayscale")

    assert gray.ndim == 2


def test_convert_color_to_rgb():
    img = _make_test_image()
    rgb = image_io.convert_color(img, "rgb")

    assert rgb.shape == img.shape
    # BGR -> RGB should equal channel-reversed representation.
    assert np.array_equal(rgb, img[:, :, ::-1])


def test_convert_color_unsupported_mode():
    img = _make_test_image()
    with pytest.raises(ValueError):
        image_io.convert_color(img, "hsv")
