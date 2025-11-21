"""Tests for feature matching utilities."""

from __future__ import annotations

import cv2
import numpy as np

from src import feature_extraction, feature_matching


def _generate_pairs():
    img_a = np.zeros((128, 128), dtype=np.uint8)
    img_b = np.zeros((128, 128), dtype=np.uint8)

    cv2.circle(img_a, (48, 48), 20, 255, -1)
    cv2.rectangle(img_a, (70, 70), (100, 100), 200, -1)

    # Translate shapes for second image to create matching features.
    cv2.circle(img_b, (54, 54), 20, 255, -1)
    cv2.rectangle(img_b, (76, 76), (106, 106), 200, -1)

    detector = feature_extraction.get_detector(
        "orb",
        nfeatures=750,
        edgeThreshold=15,
        patchSize=15,
        fastThreshold=0,
    )
    features_a = feature_extraction.extract_features(img_a, detector)
    features_b = feature_extraction.extract_features(img_b, detector)

    return features_a, features_b


def test_match_descriptors_returns_matches():
    features_a, features_b = _generate_pairs()
    matches = feature_matching.match_descriptors(features_a.descriptors, features_b.descriptors)

    assert matches


def test_filter_matches_ratio_reduces_matches():
    features_a, features_b = _generate_pairs()
    knn_matches = feature_matching.match_descriptors(
        features_a.descriptors,
        features_b.descriptors,
        use_knn=True,
        k=2,
    )

    filtered = feature_matching.filter_matches_ratio(knn_matches, ratio_threshold=0.9)

    assert len(filtered) <= len(knn_matches)


def test_summarize_matches_returns_stats():
    features_a, features_b = _generate_pairs()
    matches = feature_matching.match_descriptors(features_a.descriptors, features_b.descriptors, use_knn=False)
    flattened = (match[0] for match in matches)
    summary = feature_matching.summarize_matches(flattened)

    assert summary.total_matches == len(matches)
    assert not np.isnan(summary.average_distance)


def test_match_descriptors_handles_missing_descriptors():
    matches = feature_matching.match_descriptors(None, None)  # type: ignore[arg-type]

    assert matches == []
