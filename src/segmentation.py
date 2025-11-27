"""Segmentation algorithms for ROI detection."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ContourInfo:
    """Information about a detected contour (ROI)."""
    contour: np.ndarray
    area: float
    bounding_rect: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]


def otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Apply Otsu's automatic thresholding.
    
    Returns: (binary_image, computed_threshold)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, int(thresh_val)


def find_contours(
    binary_image: np.ndarray, 
    min_area: int = 100,
    max_area: int = None,
) -> List[ContourInfo]:
    """
    Find contours (object boundaries) in a binary image.
    
    Parameters:
        binary_image: Binary (thresholded) image
        min_area: Minimum contour area to keep (filters noise)
        max_area: Maximum contour area (None = no limit)
    
    Returns: List of ContourInfo objects
    """
    contours, _ = cv2.findContours(
        binary_image, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    results = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        
        # Compute bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Compute center (centroid)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        results.append(ContourInfo(
            contour=contour,
            area=area,
            bounding_rect=(x, y, w, h),
            center=(cx, cy),
        ))
    
    # Sort by area (largest first)
    results.sort(key=lambda c: c.area, reverse=True)
    return results


def draw_contours(
    image: np.ndarray, 
    contour_infos: List[ContourInfo],
    draw_contour: bool = True,
    draw_bounding_rect: bool = False,
    draw_center: bool = False,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
    rect_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detected contours on an image.
    
    Returns: Image with overlays drawn
    """
    # Ensure we have a color image to draw on
    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
    
    for info in contour_infos:
        if draw_contour:
            cv2.drawContours(output, [info.contour], -1, contour_color, thickness)
        
        if draw_bounding_rect:
            x, y, w, h = info.bounding_rect
            cv2.rectangle(output, (x, y), (x + w, y + h), rect_color, thickness)
        
        if draw_center:
            cv2.circle(output, info.center, 5, rect_color, -1)
    
    return output