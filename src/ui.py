"""UI helpers for Streamlit layout components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


@dataclass(frozen=True)
class DetectorSettings:
	nfeatures: int
	scale_factor: float
	nlevels: int
	edge_threshold: int
	fast_threshold: int
	patch_size: int


@dataclass(frozen=True)
class PreprocessingSettings:
	normalize: bool
	smoothing_method: Optional[str]
	kernel_size: int
	sigma: Optional[float]


@dataclass(frozen=True)
class MatchingSettings:
	ratio_threshold: float


def render_sidebar() -> tuple[DetectorSettings, PreprocessingSettings, MatchingSettings]:
	"""Render sidebar controls and return the chosen configuration."""
	st.sidebar.header("Processing Controls")

	st.sidebar.subheader("Feature Detector")
	nfeatures = st.sidebar.slider("Max keypoints", min_value=100, max_value=2000, value=750, step=50)
	scale_factor = st.sidebar.slider("Scale factor", min_value=1.1, max_value=1.9, value=1.2, step=0.05)
	nlevels = st.sidebar.slider("Pyramid levels", min_value=1, max_value=12, value=8)
	edge_threshold = st.sidebar.slider("Edge threshold", min_value=5, max_value=60, value=31)
	fast_threshold = st.sidebar.slider("FAST threshold", min_value=0, max_value=50, value=20)
	patch_size = st.sidebar.slider("Patch size", min_value=15, max_value=64, value=31, step=2)

	detector_settings = DetectorSettings(
		nfeatures=nfeatures,
		scale_factor=scale_factor,
		nlevels=nlevels,
		edge_threshold=edge_threshold,
		fast_threshold=fast_threshold,
		patch_size=patch_size,
	)

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

	st.sidebar.subheader("Matching")
	ratio_threshold = st.sidebar.slider("Lowe ratio threshold", min_value=0.1, max_value=1.0, value=0.75, step=0.05)
	matching_settings = MatchingSettings(ratio_threshold=ratio_threshold)

	return detector_settings, preprocessing_settings, matching_settings


def display_metadata(metadata: dict) -> None:
	"""Show image metadata within the main panel."""
	st.markdown("**Image Metadata**")
	st.json(metadata)
