"""Tests for preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src import preprocessing


def test_normalize_intensity_scales_range():
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    normalized = preprocessing.normalize_intensity(image)

    assert normalized.dtype == np.uint8
    assert normalized.min() == 0
    assert normalized.max() == 255


def test_normalize_intensity_constant_returns_zero():
    image = np.full((4, 4), 5, dtype=np.uint8)
    normalized = preprocessing.normalize_intensity(image)

    assert np.count_nonzero(normalized) == 0


def test_normalize_intensity_rejects_empty():
    with pytest.raises(ValueError):
        preprocessing.normalize_intensity(np.array([]))


def test_apply_smoothing_gaussian_preserves_shape():
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    smoothed = preprocessing.apply_smoothing(image, method="gaussian", kernel_size=5)

    assert smoothed.shape == image.shape


def test_apply_smoothing_requires_odd_kernel():
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocessing.apply_smoothing(image, method="gaussian", kernel_size=4)


def test_apply_smoothing_invalid_method():
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocessing.apply_smoothing(image, method="box")  # type: ignore[arg-type]


def test_compute_difference_returns_abs_diff():
    a = np.array([[0, 10], [50, 200]], dtype=np.uint8)
    b = np.array([[10, 5], [25, 100]], dtype=np.uint8)

    diff = preprocessing.compute_difference(a, b)

    expected = np.array([[10, 5], [25, 100]], dtype=np.uint8)
    assert np.array_equal(diff, expected)


def test_compute_difference_requires_matching_shapes():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError):
        preprocessing.compute_difference(a, b)
