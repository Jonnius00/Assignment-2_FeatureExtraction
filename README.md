# Medical Image Feature Extraction Demo

This project provides a Streamlit web app that demonstrates feature extraction, preprocessing, and basic feature matching on medical PNG slices. It is designed to accompany coursework on computer vision and highlight how feature descriptors work on still images and frame differences.

## Key Features
- **Image Upload & Inspection**: Drag-and-drop multiple PNG slices, view metadata, intensity statistics, and histograms.
- **Preprocessing Controls**: Toggle normalization, select smoothing filters (Gaussian, Median, Bilateral), and examine the resulting image.
- **Feature Extraction**: Configure ORB detector parameters, overlay detected keypoints, and monitor descriptor counts.
- **Differential Comparison**: Choose a secondary slice, compute an absolute difference image, and visualize feature matches with Lowe ratio filtering.

## Prerequisites
- Python 3.11 or newer (project tested on Python 3.12.7).
- `pip` for dependency management.
- Optional: virtual environment tooling (`python -m venv`).

## Setup
1. **Clone or copy the project** onto your machine.
2. **Create and activate a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Launch the Streamlit app** from the repository root:
   ```powershell
   streamlit run src/main.py
   ```
5. Open the provided local URL (default `http://localhost:8501`) in your browser.

## Usage Walkthrough
1. Upload one or more medical PNG slices using the sidebar uploader.
2. Inspect the "Preview" tab for metadata, statistics, and histograms.
3. Use the sidebar controls to tweak preprocessing and detector settings; the "Feature Extraction" tab updates live.
4. Upload a second slice to enable the "Differential Comparison" tab and inspect frame differences plus feature matches.

## Testing
Run unit tests with:
```powershell
pytest
```
Tests cover image I/O, preprocessing utilities, feature extraction, and matching logic.

## Future Enhancements
- Add true video ingestion using OpenCV `VideoCapture`.
- Support alternative descriptors (e.g., SIFT via `opencv-contrib-python`, HOG).
- Provide result exports (CSV descriptors, annotated images).

## License
This project is provided for educational purposes. Review licensing requirements before enabling detectors with patent restrictions (e.g., SIFT/SURF).
