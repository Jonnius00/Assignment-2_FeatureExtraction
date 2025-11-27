"""UI helpers for Streamlit layout components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

@dataclass(frozen=True)
class ORBSettings:
	"""ORB detector parameters."""
	nfeatures: int
	scale_factor: float
	nlevels: int
	edge_threshold: int
	fast_threshold: int
	patch_size: int


@dataclass(frozen=True)
class SIFTSettings:
	"""SIFT detector parameters."""
	nfeatures: int
	n_octave_layers: int
	contrast_threshold: float
	edge_threshold: float
	sigma: float


@dataclass(frozen=True)
class DetectorSettings:
	"""Combined detector settings with detector type selection."""
	detector_type: str  # "orb" or "sift"
	orb: ORBSettings
	sift: SIFTSettings


@dataclass(frozen=True)
class PreprocessingSettings:
	normalize: bool
	smoothing_method: Optional[str]
	kernel_size: int
	sigma: Optional[float]


@dataclass(frozen=True)
class MatchingSettings:
	ratio_threshold: float

@dataclass
class MorphologySettings:
    """Settings for morphological operations."""
    threshold: int = 127
    threshold_method: str = "binary"  # binary, binary_inv, otsu
    operation: str = "opening"        # erode, dilate, open, close, gradient
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

def render_sidebar() -> tuple[DetectorSettings, PreprocessingSettings, MorphologySettings, SegmentationSettings, MatchingSettings]:
	"""Render sidebar controls and return the chosen configuration."""
	st.sidebar.header("Processing Controls")
	
	st.sidebar.subheader("Preprocessing")
	normalize = st.sidebar.checkbox("Normalize to 0-255", value=True)
	smoothing_choice = st.sidebar.selectbox("Smoothing", ("None", "Gaussian", "Median", "Bilateral"))
	kernel_size = st.sidebar.slider("Kernel size", min_value=3, max_value=15, value=5, step=2)
	sigma = None
	if smoothing_choice == "Gaussian":
		sigma = st.sidebar.slider("Gaussian sigma", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
	elif smoothing_choice == "Bilateral":
		sigma = st.sidebar.slider("Bilateral sigma", min_value=10.0, max_value=150.0, value=75.0, step=5.0)

	smoothing_method = None if smoothing_choice == "None" else smoothing_choice.lower()
	preprocessing_settings = PreprocessingSettings(
		normalize=normalize,
		smoothing_method=smoothing_method,
		kernel_size=kernel_size,
		sigma=sigma,
	)
	
	st.sidebar.subheader("Morphology Settings")
	threshold = st.sidebar.slider("Binarization Threshold", min_value=0, max_value=255, value=127)
	threshold_method = st.sidebar.selectbox("Threshold Method", ("binary", "binary_inv", "otsu"))
	operation = st.sidebar.selectbox("Morphological Operation", ("erode", "dilate", "opening", "closing", "gradient"))
	kernel_shape = st.sidebar.selectbox("Kernel Shape", ("rect", "ellipse", "cross"))
	kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=21, value=5, step=2)
	iterations = st.sidebar.slider("Iterations", min_value=1, max_value=10, value=1)
	
	morphology_settings = MorphologySettings(
	  threshold=threshold,
	  threshold_method=threshold_method,
	  operation=operation,
	  kernel_shape=kernel_shape,
	  kernel_size=kernel_size,
	  iterations=iterations,
	)
	
	st.sidebar.subheader("Segmentation Settings")
	method = st.sidebar.selectbox("Segmentation Method", ("otsu",))
	min_contour_area = st.sidebar.slider("Min Contour Area", min_value=10, max_value=1000, value=100, step=10)
	show_contours = st.sidebar.checkbox("Show Contours", value=True)
	show_bounding_rects = st.sidebar.checkbox("Show Bounding Rects", value=False)
	
	segmentation_settings = SegmentationSettings(
      method=method,
      min_contour_area=min_contour_area,
      show_contours=show_contours,
      show_bounding_rects=show_bounding_rects,
  )

	st.sidebar.subheader("Feature Detector")
	
	# Detector type selection
	detector_type = st.sidebar.selectbox("Detector Type", ("ORB", "SIFT"))
	
	# ORB parameters (always collected, shown only when ORB selected)
	if detector_type == "ORB":
		st.sidebar.markdown("**ORB Parameters**")
		orb_nfeatures = st.sidebar.slider("Max keypoints", min_value=100, max_value=2000, value=750, step=50, key="orb_nfeatures")
		orb_scale_factor = st.sidebar.slider("Scale factor", min_value=1.1, max_value=1.9, value=1.2, step=0.05, key="orb_scale")
		orb_nlevels = st.sidebar.slider("Pyramid levels", min_value=1, max_value=12, value=8, key="orb_nlevels")
		orb_edge_threshold = st.sidebar.slider("Edge threshold", min_value=5, max_value=60, value=31, key="orb_edge")
		orb_fast_threshold = st.sidebar.slider("FAST threshold", min_value=0, max_value=50, value=20, key="orb_fast")
		orb_patch_size = st.sidebar.slider("Patch size", min_value=15, max_value=64, value=31, step=2, key="orb_patch")
	else:
		# Default ORB values when SIFT is selected
		orb_nfeatures, orb_scale_factor, orb_nlevels = 750, 1.2, 8
		orb_edge_threshold, orb_fast_threshold, orb_patch_size = 31, 20, 31
	
	orb_settings = ORBSettings(
		nfeatures=orb_nfeatures,
		scale_factor=orb_scale_factor,
		nlevels=orb_nlevels,
		edge_threshold=orb_edge_threshold,
		fast_threshold=orb_fast_threshold,
		patch_size=orb_patch_size,
	)
	
	# SIFT parameters (shown only when SIFT selected)
	if detector_type == "SIFT":
		st.sidebar.markdown("**SIFT Parameters**")
		sift_nfeatures = st.sidebar.slider("Max keypoints (0=all)", min_value=0, max_value=2000, value=0, step=50, key="sift_nfeatures")
		sift_n_octave_layers = st.sidebar.slider("Octave layers", min_value=1, max_value=6, value=3, key="sift_octave")
		sift_contrast_threshold = st.sidebar.slider("Contrast threshold", min_value=0.01, max_value=0.1, value=0.04, step=0.01, key="sift_contrast")
		sift_edge_threshold = st.sidebar.slider("Edge threshold", min_value=5.0, max_value=20.0, value=10.0, step=1.0, key="sift_edge")
		sift_sigma = st.sidebar.slider("Sigma", min_value=0.5, max_value=3.0, value=1.6, step=0.1, key="sift_sigma")
	else:
		# Default SIFT values when ORB is selected
		sift_nfeatures, sift_n_octave_layers = 0, 3
		sift_contrast_threshold, sift_edge_threshold, sift_sigma = 0.04, 10.0, 1.6
	
	sift_settings = SIFTSettings(
		nfeatures=sift_nfeatures,
		n_octave_layers=sift_n_octave_layers,
		contrast_threshold=sift_contrast_threshold,
		edge_threshold=sift_edge_threshold,
		sigma=sift_sigma,
	)
	
	detector_settings = DetectorSettings(
		detector_type=detector_type.lower(),
		orb=orb_settings,
		sift=sift_settings,
	)

	st.sidebar.subheader("Matching")
	ratio_threshold = st.sidebar.slider("Lowe ratio threshold", min_value=0.1, max_value=1.0, value=0.75, step=0.05)
	matching_settings = MatchingSettings(ratio_threshold=ratio_threshold)

	return detector_settings, preprocessing_settings, morphology_settings, segmentation_settings, matching_settings


def display_metadata(metadata: dict) -> None:
	"""Show image metadata within the main panel."""
	st.markdown("**Image Metadata**")
	st.json(metadata)
