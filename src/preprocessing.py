"""Preprocessing utilities supporting the Streamlit app backend."""

from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np


SmoothingMethod = Literal["gaussian", "median", "bilateral"]


def normalize_intensity(image: np.ndarray) -> np.ndarray:
	"""Normalize an image to cover the 0-255 range as uint8."""

	if image.size == 0:
		raise ValueError("Cannot normalize an empty image.")

	image_float = image.astype(np.float32)
	min_val = float(np.min(image_float))
	max_val = float(np.max(image_float))

	if max_val == min_val:
		return np.zeros_like(image, dtype=np.uint8)

	dst = np.empty_like(image_float)
	normalized = cv2.normalize(
		image_float,
		dst,
		alpha=0,
		beta=255,
		norm_type=cv2.NORM_MINMAX,
	)
	return normalized.astype(np.uint8)


def apply_smoothing(
	image: np.ndarray,
	method: SmoothingMethod = "gaussian",
	kernel_size: int = 5,
	sigma: Optional[float] = None,
) -> np.ndarray:
	"""Apply denoising to the input image.

	Parameters
	----------
	method:
		Smoothing algorithm to use. Supported: gaussian, median, bilateral.
	kernel_size:
		Window size for gaussian and median filters. Must be odd.
	sigma:
		Standard deviation for the gaussian kernel. Defaults to OpenCV heuristic.
	"""

	if kernel_size <= 0 or kernel_size % 2 == 0:
		raise ValueError("kernel_size must be a positive odd integer.")

	if method == "gaussian":
		return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma or 0)
	if method == "median":
		return cv2.medianBlur(image, kernel_size)
	if method == "bilateral":
		sigma_color = 75 if sigma is None else sigma
		sigma_space = kernel_size
		return cv2.bilateralFilter(image, d=kernel_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)

	raise ValueError(f"Unsupported smoothing method: {method}")


def compute_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
	"""Compute the absolute difference between two frames."""

	if frame_a.shape != frame_b.shape:
		raise ValueError("Frames must have identical dimensions for differencing.")

	return cv2.absdiff(frame_a, frame_b)

