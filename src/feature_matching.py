"""Feature matching helpers for descriptor sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchSummary:
	"""Summary statistics for a collection of matches."""

	total_matches: int
	average_distance: float


def match_descriptors(
	descriptors_a: Optional[np.ndarray],
	descriptors_b: Optional[np.ndarray],
	use_knn: bool = True,
	k: int = 2,
) -> List[List[cv2.DMatch]]:
	"""Match descriptor sets using a brute-force matcher."""

	if descriptors_a is None or descriptors_b is None:
		return []

	if descriptors_a.dtype != descriptors_b.dtype:
		raise ValueError("Descriptor arrays must share the same dtype.")

	norm_type = cv2.NORM_HAMMING if descriptors_a.dtype == np.uint8 else cv2.NORM_L2
	matcher = cv2.BFMatcher(normType=norm_type, crossCheck=not use_knn)

	if use_knn:
		matches_seq = matcher.knnMatch(descriptors_a, descriptors_b, k=k)
		matches = [list(candidate) for candidate in matches_seq]
	else:
		matches = [[m] for m in matcher.match(descriptors_a, descriptors_b)]
	return matches


def filter_matches_ratio(
	matches: Iterable[List[cv2.DMatch]],
	ratio_threshold: float = 0.75,
) -> List[cv2.DMatch]:
	"""Apply Lowe's ratio test to KNN matches."""

	filtered: List[cv2.DMatch] = []
	for candidate in matches:
		if len(candidate) < 2:
			continue
		best, second_best = candidate[:2]
		if second_best.distance == 0:
			continue
		if best.distance / second_best.distance < ratio_threshold:
			filtered.append(best)
	return filtered


def summarize_matches(matches: Iterable[cv2.DMatch]) -> MatchSummary:
	"""Generate quick statistics about match distances."""

	matches_list = list(matches)
	if not matches_list:
		return MatchSummary(total_matches=0, average_distance=float(np.nan))

	distances = [m.distance for m in matches_list]
	average = float(np.mean(distances))
	return MatchSummary(total_matches=len(matches_list), average_distance=average)
