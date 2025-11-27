# 🏥 Medical Image ROI Detection Application

> A comprehensive Streamlit-based GUI for **morphological analysis**, **segmentation**, and **feature extraction** on medical imaging data. Designed for detecting regions of interest (ROIs) through advanced image processing techniques.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Educational-orange)](LICENSE)

---

## 📋 Project Overview

This application combines **morphological operations**, **segmentation algorithms**, and **feature extraction & matching** to enable researchers, students, and medical professionals to interactively explore and analyze medical image data. Built with modern Python tools for **ease of use** and **rapid prototyping**.

## 🎯 Key Capabilities

| Feature | Description |
|---------|-------------|
| **📂 Image Management** | Load and preview PNG medical slices with metadata inspection |
| **🔧 Preprocessing Pipeline** | Normalization, smoothing (Gaussian, Median, Bilateral) |
| **🔲 Morphological Operations** | Binarization, erosion, dilation, opening, closing, gradient edge detection |
| **📊 Segmentation** | Otsu's automatic thresholding + contour detection for ROI extraction |
| **🔍 Feature Detection** | ORB (fast, free) and SIFT (scale-invariant) detector support |
| **🔗 Feature Matching** | Descriptor matching with Lowe's ratio test for differential analysis |
| **📈 Visualization** | Real-time overlays, histograms, contour maps, and match diagrams |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+** (tested on Python 3.12.7)
- **pip** package manager
- Virtual environment (recommended: `venv` or `conda`)

### Installation (5 minutes)

```powershell
# Clone the repository
git clone https://github.com/Jonnius00/Assignment-2_FeatureExtraction.git
cd "Assignment 2_FeatureExtraction"

# Create virtual environment (optional but recommended)
python -m venv MedImaggeFeatureExtractor.venv
.\MedImaggeFeatureExtractor.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run src/main.py
```

**✅ App opens automatically at:** `http://localhost:8501`

---

## 📖 User Guide

### 🎮 The Six Application Tabs

#### 1️⃣ **Preview Tab**
- View uploaded image with metadata (dimensions, data type, value range)
- Inspect intensity distribution via histogram
- Display image statistics (mean, std, min, max)
- Compare original vs. preprocessed versions

#### 2️⃣ **Preprocessing Tab**
- **Normalize Intensity**: Scale pixel values to consistent range
- **Apply Smoothing**: Choose filter type (Gaussian, Median, Bilateral)
- **Real-time Comparison**: See before/after side-by-side
- Useful for noise reduction and preparing images for feature detection

#### 3️⃣ **Morphology Tab**
**Step 1 — Binarization:**
- Convert grayscale image to binary
- Choose threshold method: manual, binary inverse, or Otsu's automatic
- Inspect computed threshold value

**Step 2 — Morphological Operations:**
- Apply: **erosion**, **dilation**, **opening**, **closing**, or **gradient**
- Configure kernel shape (rectangular, elliptical, cross)
- Adjust kernel size (3×3 to 11×11)
- Control iterations (especially for erosion/dilation)
- Use for noise removal, gap filling, and edge detection

#### 4️⃣ **Segmentation Tab**
- **Otsu's Thresholding**: Automatic optimal threshold computation
- **Contour Detection**: Extract ROI boundaries from binary image
- **Area Filtering**: Remove spurious small/large contours via min-area slider
- **Visualization**: Display detected contours and bounding rectangles
- Shows ROI count and area statistics (min, max, mean)

#### 5️⃣ **Feature Extraction Tab**
- Select detector: **ORB** (fast, royalty-free) or **SIFT** (scale-invariant)
- Configure detector parameters via sidebar sliders
- View detected keypoints overlaid on image
- Monitor keypoint count and descriptor dimensions
- Useful for identifying distinctive image features

#### 6️⃣ **Differential Comparison Tab** *(requires 2+ images)*
- Select secondary image from dropdown
- Compute absolute difference image
- Extract features from both images using same detector
- Perform feature matching with BFMatcher or FLANN
- Apply Lowe's ratio test for match quality filtering
- Visualize matches as connecting lines between keypoints
- Display match statistics (count, average distance)

---

## ⚙️ Sidebar Configuration

The **Sidebar** centralizes all settings for reproducible workflows:

### Detector Settings
- **Feature Detector**: ORB (fast, recommended for quick exploration) or SIFT (scale-invariant, better for detailed analysis)
- **ORB Parameters**: 
  - Number of features (100–5000)
  - Scale factor (1.0–2.0)
  - Edge threshold (5–31)
- **SIFT Parameters**:
  - Number of features (100–5000)
  - Scale factor (1.0–2.0)

### Preprocessing Settings
- **Normalize Intensity**: Toggle intensity normalization (0–255 scale)
- **Apply Smoothing**: Choose filter (Gaussian, Median, Bilateral) or disable
- **Kernel Size**: 3×3, 5×5, 7×7, etc.

### Morphology Settings
- **Threshold**: Slider 0–255 for manual binarization
- **Threshold Method**: Binary, Binary Inverse, Otsu Automatic
- **Operation**: Erode, Dilate, Opening, Closing, Gradient
- **Kernel Shape**: Rectangular, Elliptical, Cross
- **Kernel Size**: 3×3 to 11×11
- **Iterations**: 1–5 (for erode/dilate operations)

### Segmentation Settings
- **Min Contour Area**: Filter contours smaller than specified area
- **Show Contours**: Toggle contour visualization
- **Show Bounding Boxes**: Toggle bounding rectangle visualization

### Matching Settings
- **Ratio Threshold**: Adjust Lowe's ratio test strictness (0.5–1.0)
  - Lower = stricter filtering, fewer false matches
  - Higher = relaxed filtering, more matches (potentially false positives)

---

## 🔬 Technical Architecture

### Module Organization

```
src/
├── main.py                   # Streamlit app entry point & tab orchestration
├── ui.py                     # Sidebar controls & metadata display
├── image_io.py              # PNG loading, validation, color conversion
├── preprocessing.py         # Normalization, smoothing, difference frames
├── morphology.py            # Binarization, erosion, dilation, opening, closing, gradient
├── segmentation.py          # Otsu thresholding, contour detection, ROI extraction
├── feature_extraction.py    # ORB & SIFT detector wrappers
├── feature_matching.py      # Descriptor matching & statistics
└── visualization.py         # Image overlays, histograms, contours, matches
```

### Data Flow Pipeline

```
Input Image → Preprocess → Morphology → Segmentation → Feature Detection → Feature Matching → Visualization
```

**Key Design Patterns:**
- ✅ **Modular architecture** — each module has single responsibility
- ✅ **Type hints & dataclasses** — improved code clarity and IDE support
- ✅ **Caching** — Streamlit `@st.cache_data` for performance
- ✅ **Comprehensive testing** — 6 test files covering all modules
- ✅ **Error handling** — graceful fallbacks for edge cases

---

## 🧪 Testing

Comprehensive test suite using **pytest**:

```powershell
# Run all tests
pytest -v

# Run specific test module
pytest tests/test_morphology.py -v

# Generate coverage report
pytest --cov=src tests/
```

**Test Coverage:**
- ✅ Image I/O (PNG loading, validation)
- ✅ Preprocessing (normalization, smoothing, differencing)
- ✅ Morphology (all operations, kernel creation)
- ✅ Segmentation (Otsu, contour detection, filtering)
- ✅ Feature extraction (ORB & SIFT)
- ✅ Feature matching (descriptor matching, ratio filtering)

---

## 📚 Workflow Examples

### Example 1: Detect ROIs in Medical Image
1. **Upload** image → **Preview** tab
2. **Preprocess** (normalize + smooth) → **Preprocessing** tab
3. **Binarize** → **Morphology** tab (Step 1)
4. **Apply Opening** to remove noise → **Morphology** tab (Step 2)
5. **Detect Contours** → **Segmentation** tab
6. Filter small contours via **Min Contour Area** slider
7. ✅ Detected ROIs displayed with statistics

### Example 2: Compare Two Medical Scans
1. **Upload** first image
2. **Configure detector** (ORB for speed, SIFT for robustness)
3. **Extract features** → **Feature Extraction** tab
4. **Upload second image** to enable **Differential Comparison** tab
5. View **difference image** and **matched keypoints**
6. Adjust **ratio threshold** to control match quality
7. ✅ Matched features visualized with connection lines

### Example 3: Explore Morphological Effects
1. **Upload** image → **Morphology** tab
2. **Adjust threshold slider** → see binarization change in real-time
3. **Select operation** (e.g., "opening") → immediate result
4. **Vary kernel size** and **shape** → observe morphological impact
5. ✅ Understand how parameters affect image processing

---

## 🔧 Configuration & Dependencies

### requirements.txt
```
streamlit>=1.28.0
opencv-contrib-python>=4.8.0   # opencv-contrib needed for SIFT
numpy>=1.24.0
matplotlib>=3.7.0
pytest>=7.4.0
```

### Python Version
- **Minimum**: Python 3.11
- **Tested on**: Python 3.12.7
- **Recommended**: Python 3.12+ for best performance

---

## 📖 Documentation

- **[Developer Manual](developer_manual.md)** — In-depth guide for developers:
  - Module architecture
  - Component responsibilities
  - Extensibility patterns
  - Troubleshooting tips

- **[Implementation Plan](implementation_plan.md)** — Original project scope:
  - Objectives and design decisions
  - Feature workflow breakdown
  - Implementation phases

- **[ROI Implementation Plan](roi_implementation_plan.md)** — Assignment requirements:
  - Morphological operations specification
  - Segmentation algorithms detail
  - Feature enhancement roadmap

---

## 🎓 Educational Value

This project demonstrates:
- **Image Processing Fundamentals**: Morphological operations, segmentation, thresholding
- **Feature Detection**: Scale-invariant (SIFT) vs. fast (ORB) approaches
- **Descriptor Matching**: Ratio tests, robust correspondence filtering
- **GUI Development**: Modern Streamlit patterns for interactive data exploration
- **Software Architecture**: Modular design, type hints, comprehensive testing
- **Medical Imaging**: Real-world preprocessing and ROI detection workflows

Perfect for:
- 👨‍🎓 Computer Vision coursework
- 👨‍💼 Portfolio projects demonstrating image processing expertise
- 🏥 Quick prototyping for medical imaging applications
- 📊 Educational demonstrations of CV concepts

---

## 🚀 Future Enhancements

**Planned Extensions:**
- [ ] **AKAZE detector** — Patent-free alternative to SIFT
- [ ] **Video support** — Process video frames instead of static images
- [ ] **Watershed segmentation** — Advanced method for overlapping objects
- [ ] **Result export** — CSV descriptors, annotated images, ROI coordinates
- [ ] **GPU acceleration** — CUDA-enabled processing for large images
- [ ] **Batch processing** — Apply settings to multiple images automatically

**Potential Integrations:**
- [ ] DICOM file support (medical image standard)
- [ ] SimpleITK for advanced medical imaging
- [ ] 3D volumetric visualization

---

## 📝 License

This project is provided for **educational purposes**. 

⚠️ **Patent Notice**: SIFT was patented until March 2020. Ensure compliance with local regulations if using in commercial applications. ORB is royalty-free and recommended for production use.

---

## 👤 Author

**Jonnius00** — Computer Vision Assignment Project  
Created: November 2025

---

## 🙏 Acknowledgments

Built with:
- 🎨 [Streamlit](https://streamlit.io/) — Interactive web framework
- 🖼️ [OpenCV](https://opencv.org/) — Computer vision library
- 🔢 [NumPy](https://numpy.org/) — Numerical computing
- 📊 [Matplotlib](https://matplotlib.org/) — Data visualization
- ✅ [pytest](https://pytest.org/) — Testing framework

---

## 📞 Support & Questions

For implementation details, see the **developer_manual.md**.  
For assignment requirements, see the **roi_implementation_plan.md**.  
For questions, refer to inline code comments or the official documentation links provided.

**Happy analyzing! 🔬**
