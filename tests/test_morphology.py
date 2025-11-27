"""Tests for morphology module."""
import numpy as np
import pytest
from src import morphology


def test_binarize_simple():
    """Test simple binary thresholding."""
    img = np.array([[100, 150], [200, 50]], dtype=np.uint8)
    binary, thresh = morphology.binarize(img, threshold=127)
    assert thresh == 127
    assert binary[0, 0] == 0    # 100 < 127
    assert binary[0, 1] == 255  # 150 > 127
    assert binary[1, 0] == 255  # 200 > 127
    assert binary[1, 1] == 0    # 50 < 127


def test_get_structuring_element():
    """Test kernel creation."""
    kernel = morphology.get_structuring_element("rect", 3)
    assert kernel.shape == (3, 3)
    assert kernel.dtype == np.uint8


def test_erosion_shrinks_white():
    """Test that erosion shrinks white regions."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[3:7, 3:7] = 255  # 4x4 white square
    
    kernel = morphology.get_structuring_element("rect", 3)
    eroded = morphology.erode(img, kernel)
    
    # White area should be smaller
    assert np.sum(eroded) < np.sum(img)


def test_dilation_expands_white():
    """Test that dilation expands white regions."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[4:6, 4:6] = 255  # 2x2 white square
    
    kernel = morphology.get_structuring_element("rect", 3)
    dilated = morphology.dilate(img, kernel)
    
    # White area should be larger
    assert np.sum(dilated) > np.sum(img)