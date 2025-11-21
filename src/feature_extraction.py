"""Feature detector and descriptor helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import cv2
import numpy as np


DetectorName = str  # extendable list of detector identifiers


@dataclass(frozen=True)
class FeatureResult:
	"""Container for keypoints and descriptors."""

	keypoints: tuple[cv2.KeyPoint, ...]
	descriptors: Optional[np.ndarray]

	@property
	def keypoint_count(self) -> int:
		return len(self.keypoints)

	@property
	def descriptor_count(self) -> int:
		return 0 if self.descriptors is None else self.descriptors.shape[0]


def get_detector(
	name: DetectorName = "orb",
	**params: float | int | bool,
) -> cv2.Feature2D:
	"""Construct and return a configured OpenCV feature detector."""

	name_lower = name.lower()
	if name_lower == "orb":
		default_params = {
			"nfeatures": 500,
			"scaleFactor": 1.2,
			"nlevels": 8,
			"edgeThreshold": 31,
			"firstLevel": 0,
			"WTA_K": 2,
			"scoreType": cv2.ORB_HARRIS_SCORE,
			"patchSize": 31,
			"fastThreshold": 20,
		}
		default_params.update(params)
		return cv2.ORB_create(**default_params)  # type: ignore[attr-defined]

	raise ValueError(f"Unsupported detector: {name}")


def extract_features(
	image: np.ndarray,
	detector: cv2.Feature2D,
) -> FeatureResult:
	"""Detect keypoints and extract descriptors using the provided detector."""

	if image.ndim == 3 and image.shape[2] == 3:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image

	keypoints, descriptors = detector.detectAndCompute(gray, None)  # type: ignore[arg-type]
	if not keypoints:
		keypoints_tuple: tuple[cv2.KeyPoint, ...] = tuple()
	else:
		keypoints_tuple = tuple(keypoints)

	return FeatureResult(keypoints=keypoints_tuple, descriptors=descriptors)
