"""Morphological operations for binary image processing."""

import cv2
import numpy as np
from typing import Literal

KernelShape = Literal["rect", "ellipse", "cross"]
ThresholdMethod = Literal["binary", "binary_inv", "otsu"]

def get_structuring_element(shape: KernelShape = "rect", size: int = 5) -> np.ndarray:
    """Create structuring element (kernel) for morphological operations."""
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shapes[shape], (size, size))


def binarize(
    image: np.ndarray, 
    threshold: int = 127, 
    method: ThresholdMethod = "binary"
) -> tuple[np.ndarray, int]:
    """
    Convert grayscale image to binary.
    
    Returns: (binary_image, threshold_used)
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method == "otsu":
        thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, int(thresh_val)
    elif method == "binary_inv":
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        return binary, threshold
    else:  # binary
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary, threshold


def erode(image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply erosion: shrink white regions."""
    return cv2.erode(image, kernel, iterations=iterations)


def dilate(image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply dilation: expand white regions."""
    return cv2.dilate(image, kernel, iterations=iterations)


def opening(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply opening: erosion followed by dilation (removes small noise)."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply closing: dilation followed by erosion (fills small holes)."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def gradient(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply morphological gradient: dilation - erosion (edge detection)."""
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


# Convenience function to apply any operation by name
def apply_morphology(
    image: np.ndarray,
    operation: str,
    kernel_shape: KernelShape = "rect",
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply a morphological operation by name.
    
    Operations: "erode", "dilate", "opening", "closing", "gradient"
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    
    operations = {
        "erode": lambda img: erode(img, kernel, iterations),
        "dilate": lambda img: dilate(img, kernel, iterations),
        "opening": lambda img: opening(img, kernel),
        "closing": lambda img: closing(img, kernel),
        "gradient": lambda img: gradient(img, kernel),
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}. Use: {list(operations.keys())}")
    
    return operations[operation](image)