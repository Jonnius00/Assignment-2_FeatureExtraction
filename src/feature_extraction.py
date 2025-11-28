"""Feature detector and descriptor helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import cv2
import numpy as np


DetectorName = str  # extendable list of detector identifiers


class _HarrisDetector:
	"""Wrapper for Harris corner detection that implements cv2.Feature2D interface."""
	
	def __init__(self, blockSize: int = 2, ksize: int = 3, k: float = 0.04, threshold: float = 0.01):
		"""Initialize Harris corner detector with parameters.
		
		Args:
			blockSize: Neighborhood size (default: 2)
			ksize: Sobel kernel size (default: 3, must be odd)
			k: Harris corner response parameter (default: 0.04)
			threshold: Corner strength threshold (default: 0.01)
		"""
		self.blockSize = blockSize
		self.ksize = ksize
		self.k = k
		self.threshold = threshold
	
	def detectAndCompute(
		self,
		image: np.ndarray,
		mask: Optional[np.ndarray] = None,
	) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray]]:
		"""Detect corners and return as keypoints.
		
		Args:
			image: Input image (grayscale or BGR)
			mask: Optional mask (not used for Harris)
		
		Returns:
			Tuple of (keypoints, descriptors) where descriptors are corner response strengths
		"""
		# Convert to grayscale if needed
		if image.ndim == 3:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			gray = image
		
		# Compute Harris corner response
		corners = cv2.cornerHarris(gray, self.blockSize, self.ksize, self.k)
		
		# Normalize corner response to 0-255 range
		corners_norm = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		
		# Threshold to get corner locations
		threshold_val = int(self.threshold * 255)
		corner_mask = corners_norm > threshold_val
		
		# Get corner coordinates
		corner_coords = np.column_stack(np.where(corner_mask))
		
		# Convert to keypoints
		keypoints = []
		responses = []
		
		for y, x in corner_coords:
			kp = cv2.KeyPoint(float(x), float(y), 5.0)
			kp.response = float(corners[y, x])
			keypoints.append(kp)
			responses.append(corners_norm[y, x])
		
		# Sort by response strength (descending)
		if keypoints:
			sorted_indices = np.argsort(-np.array(responses))
			keypoints = [keypoints[i] for i in sorted_indices]
			descriptors = corners_norm[corner_coords[sorted_indices, 0], corner_coords[sorted_indices, 1]].reshape(-1, 1)
		else:
			descriptors = None
		
		return keypoints, descriptors


@dataclass(frozen=True)
class FeatureResult:
	"""Container for keypoints and descriptors."""

	keypoints: tuple[cv2.KeyPoint, ...]
	descriptors: Optional[np.ndarray]

	@property
	def keypoint_count(self) -> int:
		"""Get the number of keypoints.
		
		Returns:
			int: The count of keypoints stored in this FeatureResult instance.
		"""
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
	
	if name_lower == "sift":
		default_params = {
			"nfeatures": 0,            # 0 = detect all features
			"nOctaveLayers": 3,
			"contrastThreshold": 0.04,
			"edgeThreshold": 10,
			"sigma": 1.6,
		}
		default_params.update(params)
		return cv2.SIFT_create(**default_params)

	if name_lower == "harris":
		default_params = {
			"blockSize": 2,
			"ksize": 3,
			"k": 0.04,
			"threshold": 0.01,
		}
		default_params.update(params)
		return _HarrisDetector(**default_params)  # type: ignore[return-value]

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
