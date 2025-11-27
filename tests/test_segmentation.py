"""Tests for segmentation module."""
import numpy as np
import pytest
from src import segmentation


def test_otsu_threshold_returns_binary():
    """Test Otsu produces binary output."""
    # Create image with two distinct regions
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:50, :] = 50   # dark region
    img[50:, :] = 200  # bright region
    
    binary, thresh = segmentation.otsu_threshold(img)
    
    assert set(np.unique(binary)) <= {0, 255}
    # Otsu should find a threshold between the two intensity values
    # The threshold can be at the boundary (50) or anywhere between 50-200
    assert 50 <= thresh <= 200


def test_find_contours_filters_small():
    """Test that small contours are filtered out."""
    # Create binary image with small and large objects
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10:15, 10:15] = 255   # Small 5x5 square (area ~25)
    img[50:80, 50:80] = 255   # Large 30x30 square (area ~900)
    
    contours = segmentation.find_contours(img, min_area=100)
    
    # Only the large square should be found
    assert len(contours) == 1
    assert contours[0].area > 100


def test_find_contours_returns_sorted():
    """Test contours are sorted by area (largest first)."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10:20, 10:20] = 255   # 10x10 square
    img[50:80, 50:80] = 255   # 30x30 square
    
    contours = segmentation.find_contours(img, min_area=0)
    
    assert len(contours) == 2
    assert contours[0].area > contours[1].area  # Largest first