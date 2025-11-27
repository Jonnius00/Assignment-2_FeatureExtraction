# ROI Detection Enhancement — Implementation Plan

*Created: November 27, 2025*
*Purpose: Extend the existing project to meet the assignment requirements*

---

## Assignment Requirements Recap

> Create a GUI application where **morphological operations**, **segmentation algorithms**, and **feature extraction and matching** is used to detect **region of interests**.

| Requirement | Current State | Action Needed |
|-------------|---------------|---------------|
| GUI application | ✅ Streamlit exists | Minor extensions |
| Morphological operations | ❌ Missing | **Add new module** |
| Segmentation algorithms | ❌ Missing | **Add new module** |
| Feature extraction | ✅ ORB exists | Add SIFT |
| Feature matching | ✅ Exists | No changes |
| Visualization | ✅ Exists | Extend for new features |

---

## Final Scope (Minimal Viable)

### Morphological Operations (5 operations + prerequisite)
| Operation | OpenCV Function | Purpose |
|-----------|-----------------|---------|
| **Binarization** | `cv2.threshold()` | Convert grayscale → binary (prerequisite) |
| **Erosion** | `cv2.erode()` | Shrink white regions, remove small noise |
| **Dilation** | `cv2.dilate()` | Expand white regions, fill small holes |
| **Opening** | `cv2.morphologyEx(MORPH_OPEN)` | Erosion then Dilation (remove noise) |
| **Closing** | `cv2.morphologyEx(MORPH_CLOSE)` | Dilation then Erosion (fill gaps) |
| **Gradient** | `cv2.morphologyEx(MORPH_GRADIENT)` | Edge detection (Dilation - Erosion) |

### Segmentation Algorithms (2 methods)
| Algorithm | OpenCV Function | Purpose |
|-----------|-----------------|---------|
| **Otsu's Thresholding** | `cv2.threshold(..., THRESH_OTSU)` | Automatic optimal threshold |
| **Contour Detection** | `cv2.findContours()` | Find object boundaries |

### Feature Detectors (2 total)
| Detector | Status |
|----------|--------|
| **ORB** | ✅ Already implemented |
| **SIFT** | ➕ To be added |

---

## Dependency Change Required

```txt
# In requirements.txt, REPLACE:
opencv-python==4.x.x

# WITH:
opencv-contrib-python==4.x.x

# Reason: SIFT is in the contrib package (was patented until 2020)
```

---

## New Files to Create

### 1. `src/morphology.py`

```python
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
```

---

### 2. `src/segmentation.py`

```python
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
```

---

## Files to Modify

### 3. `src/feature_extraction.py` — Add SIFT

Add this block inside `get_detector()` function (after the ORB block):

```python
    # Add after the ORB block:
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
```

---

### 4. `src/ui.py` — Add Settings Dataclasses

Add these new dataclasses for morphology and segmentation settings:

```python
@dataclass
class MorphologySettings:
    """Settings for morphological operations."""
    threshold: int = 127
    threshold_method: str = "binary"  # binary, binary_inv, otsu
    operation: str = "opening"        # erode, dilate, opening, closing, gradient
    kernel_shape: str = "rect"        # rect, ellipse, cross
    kernel_size: int = 5
    iterations: int = 1


@dataclass
class SegmentationSettings:
    """Settings for segmentation."""
    method: str = "otsu"              # otsu
    min_contour_area: int = 100
    show_contours: bool = True
    show_bounding_rects: bool = False
```

---

### 5. `src/main.py` — Add New Tabs

Add imports at top:
```python
from src import morphology, segmentation
```

Add new tabs to the UI (extend existing tab structure):
- **Morphology Tab**: Apply and visualize morphological operations
- **Segmentation Tab**: Apply Otsu + show detected contours

---

## New Test Files

### 6. `tests/test_morphology.py`

```python
"""Tests for morphology module."""
import numpy as np
import pytest
from src import morphology


def test_binarize_simple():
    """Test simple binary thresholding."""
    img = np.array([[100, 150], [200, 50]], dtype=np.uint8)
    binary, thresh = morphology.binarize(img, threshold=127)
    assert thresh == 127
    assert binary[0, 0] == 0    # 100 < 127
    assert binary[0, 1] == 255  # 150 > 127
    assert binary[1, 0] == 255  # 200 > 127
    assert binary[1, 1] == 0    # 50 < 127


def test_get_structuring_element():
    """Test kernel creation."""
    kernel = morphology.get_structuring_element("rect", 3)
    assert kernel.shape == (3, 3)
    assert kernel.dtype == np.uint8


def test_erosion_shrinks_white():
    """Test that erosion shrinks white regions."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[3:7, 3:7] = 255  # 4x4 white square
    
    kernel = morphology.get_structuring_element("rect", 3)
    eroded = morphology.erode(img, kernel)
    
    # White area should be smaller
    assert np.sum(eroded) < np.sum(img)


def test_dilation_expands_white():
    """Test that dilation expands white regions."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[4:6, 4:6] = 255  # 2x2 white square
    
    kernel = morphology.get_structuring_element("rect", 3)
    dilated = morphology.dilate(img, kernel)
    
    # White area should be larger
    assert np.sum(dilated) > np.sum(img)
```

---

### 7. `tests/test_segmentation.py`

```python
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
    assert 50 < thresh < 200


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
```

---

## UI Tab Structure (After Enhancement)

```
┌─────────────────────────────────────────────────────────────────┐
│ Medical Image ROI Detection Demo                                │
├─────────────────────────────────────────────────────────────────┤
│ [Preview] [Preprocessing] [Morphology] [Segmentation]           │
│ [Feature Extraction] [Differential Comparison]                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Content changes based on selected tab...                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Morphology Tab Sidebar Controls:
- Threshold slider (0-255)
- Threshold method dropdown (binary / binary_inv / otsu)
- Operation dropdown (erode / dilate / opening / closing / gradient)
- Kernel shape dropdown (rect / ellipse / cross)
- Kernel size dropdown (3 / 5 / 7 / 9 / 11)
- Iterations slider (1-5, for erode/dilate only)

### Segmentation Tab Sidebar Controls:
- Method dropdown (otsu)
- Min contour area slider (0-1000)
- Show contours checkbox
- Show bounding rectangles checkbox

### Feature Extraction Tab Updates:
- Detector dropdown: ORB / **SIFT** (new)
- Parameters update based on selected detector

---

## Implementation Order

| Step | Task | Est. Time | Files |
|------|------|-----------|-------|
| 1 | Update `requirements.txt` | 5 min | `requirements.txt` |
| 2 | Create `morphology.py` | 30 min | `src/morphology.py` |
| 3 | Create `segmentation.py` | 30 min | `src/segmentation.py` |
| 4 | Add SIFT to `feature_extraction.py` | 15 min | `src/feature_extraction.py` |
| 5 | Add settings dataclasses to `ui.py` | 20 min | `src/ui.py` |
| 6 | Add sidebar controls for new features | 30 min | `src/ui.py` |
| 7 | Add Morphology tab to `main.py` | 30 min | `src/main.py` |
| 8 | Add Segmentation tab to `main.py` | 30 min | `src/main.py` |
| 9 | Update Feature Extraction tab for SIFT | 15 min | `src/main.py` |
| 10 | Create `test_morphology.py` | 15 min | `tests/test_morphology.py` |
| 11 | Create `test_segmentation.py` | 15 min | `tests/test_segmentation.py` |
| 12 | Integration testing & bug fixes | 30 min | All |
| **Total** | | **~4-5 hours** | |

---

## Quick Start Commands

```powershell
# 1. Activate virtual environment
.\\MedImaggeFeatureExtractor.venv\\Scripts\\Activate.ps1

# 2. Update dependencies (after changing requirements.txt)
pip install -r requirements.txt

# 3. Run tests
pytest

# 4. Launch the app
streamlit run src/main.py
```

---

## Visual Workflow Summary

```
┌──────────────┐
│  Load Image  │
└──────┬───────┘
       ↓
┌──────────────┐
│  Preprocess  │ (normalize, smooth)
└──────┬───────┘
       ↓
┌──────────────┐
│  Binarize    │ (threshold)
└──────┬───────┘
       ↓
┌──────────────┐
│  Morphology  │ (opening → closing to clean up)
└──────┬───────┘
       ↓
┌──────────────┐
│ Segmentation │ (find contours = ROIs)
└──────┬───────┘
       ↓
┌──────────────┐
│  Features    │ (SIFT/ORB on ROIs)
└──────┬───────┘
       ↓
┌──────────────┐
│  Visualize   │
└──────────────┘
```

---

## Notes

- **AKAZE detector**: Postponed. Can be added later if teacher requires (same pattern as SIFT).
- **Top-hat / Black-hat**: Postponed. Add to morphology.py if needed.
- **Watershed segmentation**: Postponed. More complex, add if required.

---

*Ready to start implementation when you confirm!*
