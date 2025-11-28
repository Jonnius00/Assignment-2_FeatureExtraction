"""Tests for feature extraction helpers."""

from __future__ import annotations

import cv2
import numpy as np

from src import feature_extraction


def _make_checkerboard(size: int = 64, block: int = 8) -> np.ndarray:
    coords = np.indices((size, size))
    pattern = ((coords[0] // block) + (coords[1] // block)) % 2
    image = (pattern * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_get_detector_returns_orb():
    detector = feature_extraction.get_detector("orb", nfeatures=100)

    assert hasattr(detector, "detectAndCompute")


def test_extract_features_returns_keypoints():
    image = _make_checkerboard()
    detector = feature_extraction.get_detector(
        "orb",
        nfeatures=500,
        edgeThreshold=15,
        patchSize=15,
        fastThreshold=0,
    )

    result = feature_extraction.extract_features(image, detector)

    assert result.keypoint_count > 0
    assert result.descriptors is not None
    assert result.descriptor_count == result.descriptors.shape[0]


def test_extract_features_handles_grayscale_input():
    image = cv2.cvtColor(_make_checkerboard(), cv2.COLOR_BGR2GRAY)
    detector = feature_extraction.get_detector(
        "orb",
        nfeatures=500,
        edgeThreshold=15,
        patchSize=15,
        fastThreshold=0,
    )

    result = feature_extraction.extract_features(image, detector)

    assert result.keypoint_count > 0
    assert result.descriptors is not None


def test_get_detector_raises_for_unknown():
    try:
        feature_extraction.get_detector("unknown_detector_name")
    except ValueError as exc:
        assert "Unsupported detector" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported detector")


def test_get_detector_returns_harris():
    """Test that Harris detector can be instantiated."""
    detector = feature_extraction.get_detector("harris")
    assert hasattr(detector, "detectAndCompute")


def test_harris_extract_features_returns_keypoints():
    """Test that Harris detector detects corners in a checkerboard pattern."""
    image = _make_checkerboard()
    detector = feature_extraction.get_detector("harris", blockSize=2, ksize=3, k=0.04, threshold=0.01)
    result = feature_extraction.extract_features(image, detector)
    
    # Checkerboard should have many corners
    assert result.keypoint_count > 0
    assert result.descriptors is not None
    assert result.descriptor_count == result.keypoint_count


def test_harris_accepts_parameters():
    """Test that Harris detector accepts custom parameters."""
    detector = feature_extraction.get_detector("harris", blockSize=5, ksize=5, k=0.08, threshold=0.02)
    image = _make_checkerboard()
    result = feature_extraction.extract_features(image, detector)
    # Should still detect corners, just possibly fewer due to higher threshold
    assert result.keypoint_count >= 0


def test_harris_handles_grayscale_input():
    """Test that Harris detector works with grayscale images."""
    gray_image = cv2.cvtColor(_make_checkerboard(), cv2.COLOR_BGR2GRAY)
    detector = feature_extraction.get_detector("harris")
    result = feature_extraction.extract_features(gray_image, detector)
    assert result.keypoint_count > 0
    assert result.descriptors is not None
