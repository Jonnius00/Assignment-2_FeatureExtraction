# Medical Image Feature Extraction App — Implementation Plan

## 1. Objectives & Scope
- Demonstrate feature extraction concepts on medical PNG slices via a simple interactive interface.
- Support single-image analysis first, then differential comparisons between two slices to hint at video-based matching.
- Provide a future-ready outline for extending to true video processing without committing to that work now.

## 2. Target Stack & Tooling
- **Language:** Python 3.11 (adjust to local availability if needed).
- **Framework:** Streamlit for a lightweight web UI (fast prototyping, built-in widgets).
- **Core Libraries:**
  - `opencv-python` for image loading, preprocessing, feature detectors/descriptors, and matching.
  - `numpy` for array manipulation.
  - `scikit-image` (optional) for supplemental metrics (texture, structural similarity) if time permits.
  - `matplotlib` or `plotly` (Streamlit-native) for histograms and charts.
- **Environment:** Virtual environment managed with `venv`.
- **Version Control:** Git (ensure existing worktree changes are preserved).
- **Testing:** `pytest` for unit tests, `tox` optional if time allows.

## 3. Workspace Layout
```
Assignment 2_FeatureExtraction/
├─ dicom_images_png/          # dicom images 
├─ assignment.txt
├─ implementation_plan.md     # fixes progress and changes
├─ requirements.txt           # dependency pins
├─ README.md                  # high-level overview + instructions
├─ src/
│  ├─ __init__.py
│  ├─ main.py                 # Streamlit entry point
│  ├─ ui.py                   # UI layout and widget orchestration
│  ├─ image_io.py             # file validation, loading, metadata extraction
│  ├─ preprocessing.py        # grayscale conversion, normalization, diff frames
│  ├─ feature_extraction.py   # detector/descriptor wrappers (ORB first)
│  ├─ feature_matching.py     # pairwise feature matching utilities
│  └─ visualization.py        # helper plots and overlays
└─ tests/
   ├─ __init__.py
   ├─ test_image_io.py
   ├─ test_feature_extraction.py
   └─ test_feature_matching.py
```

## 4. Data Handling Strategy
- Accept PNG uploads through Streamlit file uploader; store in memory first, optionally cache temporary files.
- Validate image type, dimensions, and bit depth; convert to consistent grayscale or BGR structure.
- Initially load and analyze only the first supplied PNG (per current requirement) to avoid bulk processing.
- Extract metadata: shape, dtype, min/max intensity, histogram bins.
- For differential comparisons, allow a second PNG selection and compute `abs(frame_a - frame_b)`.

## 5. Core Functional Modules
### 5.1 `image_io.py`
- `load_png(file_like) -> np.ndarray`
- `validate_image(img) -> dict` returning quality checks and metadata.
- `convert_color(img, mode)` to handle grayscale/BGR conversions.

### 5.2 `preprocessing.py`
- `normalize_intensity(img)` to scale values to 0-255.
- `apply_smoothing(img, method)` for optional denoising.
- `compute_difference(img_a, img_b)` for differential frames.

### 5.3 `feature_extraction.py`
- `get_detector(name, **params)` returning configured ORB/SIFT/SURF (start with ORB — no extra packages).
- `extract_features(img, detector)` returning keypoints and descriptors.
- Future: integrate texture descriptors (HOG, Haralick) if time remains.

### 5.4 `feature_matching.py`
- `match_descriptors(desc_a, desc_b, method="bf")` using BFMatcher / FLANN.
- `filter_matches(matches, ratio_thresh)` to keep best correspondences.

### 5.5 `visualization.py`
- `render_keypoints(img, keypoints)` overlay keypoints via `cv2.drawKeypoints`.
- `render_matches(img_a, keypoints_a, img_b, keypoints_b, matches)` for side-by-side diagrams.
- `plot_histogram(img)` returning a Streamlit-compatible figure.

### 5.6 `ui.py`
- Layout for sidebar configuration (file upload, detector selection, parameter sliders).
- Tabs/sections: Image preview, Feature extraction, Differential comparison, Diagnostics.
- Display of textual metrics (mean, std, keypoint counts) and figures.

### 5.7 `main.py`
- Initialize Streamlit page config.
- Orchestrate calls between UI widgets and backend modules.
- Manage session state (uploaded files, selected detector, cached results).

## 6. Feature Workflow Breakdown
1. **File Upload & Inspection**
   - Single PNG upload path; display preview and metadata table.
   - Provide histogram chart and optional intensity statistics.
2. **Preprocessing Options**
   - Toggle grayscale conversion.
   - Optional smoothing and normalization controls (checkbox + sliders).
3. **Feature Extraction**
   - Allow selection of ORB parameters (nfeatures, scaleFactor, edgeThreshold).
   - Compute keypoints/descriptors and visualize overlays.
   - Show descriptor size, keypoint count, processing time.
4. **Descriptor Export** (nice-to-have)
   - Provide download button for descriptors as CSV/JSON.
5. **Differential Frame Comparison**
   - Enable upload/selection of a second PNG.
   - Compute absolute difference image and display.
   - Run ORB on both frames, perform BFMatcher ratio test, visualize matches.
   - Report match counts and average distances.
6. **Session Logging**
   - Maintain debug/info logs (e.g., via `logging` module) displayed in an expandable section.

## 7. Implementation Phases
### Phase 0 — Preparation (0.5 day)
- Confirm Python version; set up virtual environment (`python -m venv .venv`, activate in PowerShell).
- Create `requirements.txt` with pinned versions: `streamlit`, `opencv-python`, `numpy`, `matplotlib`, `pytest` (add `scikit-image` later if needed).
- Install dependencies, run a quick script to load the first provided PNG and print metadata.

### Phase 1 — Project Scaffolding (0.5 day)
- Create directory structure (`src`, `tests`).
- Add initial `README.md` with project summary and run instructions.
- Draft `.streamlit/config.toml` if custom theme desired (optional).

### Phase 2 — Data IO & Preprocessing (1 day)
- Implement `image_io.py` with loading/validation conversions.
- Write unit tests covering typical and edge input scenarios (bad file type, grayscale vs color).
- Add `preprocessing.py` helpers; test normalization and difference computation.

### Phase 3 — Feature Extraction Backend (1 day)
- Implement ORB detector creation, feature extraction, and descriptor handling.
- Add feature matching utilities with BFMatcher and ratio test.
- Write tests using sample arrays or synthetic images to verify keypoint counts and match filtering.

### Phase 4 — Streamlit UI (1.5 days)
- Build base layout in `main.py` and `ui.py` with file uploader and preview components.
- Link preprocessing controls and display visual outputs via `visualization.py`.
- Integrate feature extraction controls, results, and visualizations.
- Add differential comparison tab with dual uploads and match visualization.
- Ensure responsive behavior (avoid re-computation by caching results with `st.cache_data` where appropriate).

### Phase 5 — Polish & Documentation (0.5 day)
- Improve error handling and user prompts (e.g., warnings when no keypoints found).
- Finalize README: installation, usage walkthrough, screenshots, feature explanation, future work.
- Capture demo screenshots/gifs showing extraction and matching results for inclusion in docs or presentation.

### Phase 6 — Testing & Handover (0.5 day)
- Run unit tests; add Streamlit smoke test instructions (`streamlit run src/main.py`).
- Validate on sample medical images; log observations (contrast, noise) for future improvements.
- Prepare submission package: source code, README, tests, sample outputs, and this plan.

## 8. Testing Strategy
- **Unit Tests:** Focus on deterministic logic (loading, preprocessing, descriptor extraction) using synthetic images.
- **Integration Tests:** Manual testing of Streamlit workflows; document checklist (upload, preprocess, extract, compare).
- **Performance Checks:** Measure time per extraction/match to ensure interface remains responsive for high-resolution slices.
- **Regression Safeguards:** Keep baseline hashes/keypoint counts for sample images to detect unexpected changes.

## 9. Documentation & Deliverables
- `README.md` with project overview, prerequisites, setup, usage, screenshots, future extensions.
- `implementation_plan.md` (this file) for instructor reference.
- Optional `architecture.md` with module diagrams, only if requested.
- Optional short video or GIF demonstrating app interactions, only if requested.

## 10. Extension Roadmap (Post-Submission)
- Integrate real video ingestion via `cv2.VideoCapture` with frame sampling and difference visualization.
- Experiment with additional descriptors (SIFT via `opencv-contrib-python`, HOG, deep features via pretrained models).
- Implement automatic ROI selection or segmentation to focus on medical structures.
- Deploy Streamlit app via Streamlit Cloud or local network for class demo.

## 11. Risks & Mitigations
- **Large image sizes** → Downsample preview while keeping full resolution for computation.
- **Performance bottlenecks** → Use caching, limit selectable features, provide progress indicators.
- **Licensing concerns (SIFT/SURF)** → Default to ORB; mention licensing in README if enabling others.
- **Limited time** → Prioritize Phases 0–4; treat phases 5–6 as stretch goals if deadlines tighten.

## 12. Success Criteria Checklist
- [ ] Streamlit app runs locally and loads first PNG slice without error.
- [ ] User can visualize original image, histogram, and summary stats.
- [ ] ORB features extract successfully with adjustable parameters.
- [ ] Keypoint overlay and descriptor summaries display in UI.
- [ ] Differential comparison between two slices shows difference image and match visualization.
- [ ] README documents setup, usage, and theory at a beginner-friendly level.
- [ ] Unit tests cover core utility functions and pass consistently.
