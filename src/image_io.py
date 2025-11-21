"""Utility helpers for loading and validating medical image slices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import cv2
import numpy as np


ImageSource = Union[str, Path, bytes, bytearray, np.ndarray, BinaryIO]


@dataclass(frozen=True)
class ImageMetadata:
	"""Describes basic properties of a loaded image."""

	width: int
	height: int
	channels: int
	dtype: str
	min_intensity: float
	max_intensity: float
	is_grayscale: bool


def load_png(source: ImageSource) -> np.ndarray:
	"""Load a PNG image from disk or memory into a NumPy array.

	Parameters
	----------
	source:
		Any supported image source. Strings and paths are treated as file paths.
		Bytes-like objects and file handles are decoded in memory.

	Returns
	-------
	np.ndarray
		Image in BGR order when 3-channel, or single channel for grayscale.

	Raises
	------
	ValueError
		If the source cannot be decoded as a PNG image.
	"""

	if isinstance(source, (str, Path)):
		image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
	elif isinstance(source, (bytes, bytearray, np.ndarray)):
		buffer = np.frombuffer(source, dtype=np.uint8)
		image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
	elif hasattr(source, "read"):
		data = source.read()
		buffer = np.frombuffer(data, dtype=np.uint8)
		image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
	else:
		msg = f"Unsupported source type: {type(source)!r}"
		raise TypeError(msg)

	if image is None:
		raise ValueError("Unable to decode image from provided source.")

	return image


def validate_image(image: np.ndarray) -> ImageMetadata:
	"""Validate the image array and return basic metadata.

	Raises
	------
	ValueError
		When the image array is empty or does not have 2-3 dimensions.
	"""

	if image is None:
		raise ValueError("Image data is None.")

	if image.size == 0:
		raise ValueError("Image array is empty.")

	if image.ndim not in (2, 3):
		raise ValueError("Expected 2D or 3D image array.")

	height, width = image.shape[:2]
	channels = 1 if image.ndim == 2 else image.shape[2]
	min_val = float(image.min())
	max_val = float(image.max())
	is_grayscale = channels == 1

	return ImageMetadata(
		width=width,
		height=height,
		channels=channels,
		dtype=str(image.dtype),
		min_intensity=min_val,
		max_intensity=max_val,
		is_grayscale=is_grayscale,
	)


def convert_color(image: np.ndarray, mode: str) -> np.ndarray:
	"""Convert image color representation.

	Parameters
	----------
	image:
		Input image array.
	mode:
		Target mode. Supports "grayscale", "rgb", and "bgr".
	"""

	if image.ndim not in (2, 3):
		raise ValueError("Expected 2D or 3D image array for conversion.")

	mode_lower = mode.lower()

	# No conversion required for matching representation.
	if mode_lower == "grayscale" and image.ndim == 2:
		return image
	if mode_lower == "bgr" and image.ndim == 3 and image.shape[2] == 3:
		return image
	if mode_lower == "rgb" and image.ndim == 3 and image.shape[2] == 3:
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	if mode_lower == "grayscale":
		conversion = cv2.COLOR_BGR2GRAY if image.ndim == 3 else None
	elif mode_lower == "bgr":
		conversion = cv2.COLOR_GRAY2BGR if image.ndim == 2 else None
	elif mode_lower == "rgb":
		if image.ndim == 2:
			conversion = cv2.COLOR_GRAY2RGB
		else:
			conversion = cv2.COLOR_BGR2RGB
	else:
		raise ValueError(f"Unsupported color mode: {mode}")

	if conversion is None:
		return image

	converted = cv2.cvtColor(image, conversion)
	return converted

