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

### 5.1 `src/morphology.py` — Morphological Operations
Core functions for binary image processing and morphological transformations:
- `binarize(img, method, threshold)` supporting manual, binary_inv, and Otsu thresholding.
- `erode(img, kernel_shape, kernel_size, iterations)` to shrink white regions.
- `dilate(img, kernel_shape, kernel_size, iterations)` to expand white regions.
- `opening(img, kernel_shape, kernel_size)` applying erosion then dilation.
- `closing(img, kernel_shape, kernel_size)` applying dilation then erosion.
- `gradient(img, kernel_shape, kernel_size)` for edge detection via dilation minus erosion.
- `create_kernel(shape, size)` factory for rectangular, elliptical, and cross-shaped kernels.

### 5.2 `src/segmentation.py` — Segmentation Algorithms
Image segmentation methods for ROI boundary detection:
- `otsu_threshold(img)` computing automatic optimal threshold value.
- `find_contours(img, min_area)` extracting contour boundaries from binary image.
- `get_contour_bounding_boxes(contours)` computing axis-aligned rectangles for detected regions.
- `filter_contours_by_area(contours, min_area, max_area)` removing spurious small/large regions.

## Files to Modify

### 5.3 `src/feature_extraction.py` — Add SIFT Detector
Extend the existing `get_detector()` function to support SIFT:
- Add SIFT detector initialization block alongside existing ORB block.
- `extract_features(img, detector)` already handles both ORB and SIFT descriptors.
- Return keypoints and descriptors in standardized format for downstream matching.

### 5.4 `src/ui.py` — UI Layout Enhancement
Extend UI components to support morphology and segmentation workflows:
- Add sidebar configuration for morphology (threshold method, operation type, kernel parameters).
- Add sidebar configuration for segmentation (method selection, contour filtering thresholds).
- Add dropdown for feature detector selection (ORB / SIFT).
- Extend tab structure to include **Morphology Tab** and **Segmentation Tab**.
- Maintain layout consistency with existing **Preview**, **Preprocessing**, and **Feature Extraction** tabs.

### 5.5 `src/main.py` — New Tab Orchestration
Integrate new modules into Streamlit application flow:
- Add **Morphology Tab** handling: load binary image, apply morphological operation, display result with processing metrics.
- Add **Segmentation Tab** handling: apply Otsu thresholding, detect contours, filter by area, visualize bounding boxes.
- Route user selections to `morphology.py` and `segmentation.py` functions.
- Manage session state for cached binary images and detected contours.

## New Test Files

### 5.6 `tests/test_morphology.py` — Morphology Tests
Unit tests for morphological operations module:
- Test binarization with different threshold methods and values.
- Test erosion, dilation, opening, closing, and gradient operations with various kernel shapes/sizes.
- Verify kernel creation for rectangular, elliptical, and cross shapes.
- Validate edge cases (empty image, single-pixel image, fully white/black images).

### 5.7 `tests/test_segmentation.py` — Segmentation Tests
Unit tests for segmentation algorithms module:
- Test Otsu thresholding on synthetic and real image samples.
- Test contour detection and filtering by area.
- Verify bounding box computation for detected contours.
- Validate robustness with images containing multiple ROIs of varying sizes.

## UI Tab Structure (After Enhancement)

```
┌───────────────────────────────────────────────────────┐
│ Medical Image ROI Detection Demo                      │
├───────────────────────────────────────────────────────┤
│ [Preview] [Preprocessing] [Morphology] [Segmentation] │
│ [Feature Extraction] [Differential Comparison]        │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Content changes based on selected tab...             │
│                                                       │
└───────────────────────────────────────────────────────┘
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

