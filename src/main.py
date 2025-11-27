"""Streamlit entry point for the medical image feature extraction demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sys

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src import feature_extraction, feature_matching, image_io, preprocessing, ui, visualization, morphology, segmentation


@dataclass
class LoadedImage:
	key: str
	name: str
	image: np.ndarray
	metadata: image_io.ImageMetadata


@st.cache_data(show_spinner=False)
def _load_image(bytes_data: bytes, name: str) -> LoadedImage:
	"""Decode a PNG payload and cache its metadata."""
	image = image_io.load_png(bytes_data)
	metadata = image_io.validate_image(image)
	return LoadedImage(key=name, name=name, image=image, metadata=metadata)


def _preprocess_image(image: np.ndarray, settings: ui.PreprocessingSettings) -> np.ndarray:
	"""Apply optional normalization and smoothing according to sidebar settings."""
	working = image
	if working.dtype != np.uint8:
		working = preprocessing.normalize_intensity(working)
	if settings.normalize:
		working = preprocessing.normalize_intensity(working)
	if settings.smoothing_method:
		working = preprocessing.apply_smoothing(
			working,
			method=settings.smoothing_method,  # type: ignore[arg-type]
			kernel_size=settings.kernel_size,
			sigma=settings.sigma,
		)
	return working


def _build_metadata_dict(meta: image_io.ImageMetadata) -> dict:
	"""Convert metadata dataclass to a Streamlit-friendly dictionary."""
	return {
		"width": meta.width,
		"height": meta.height,
		"channels": meta.channels,
		"dtype": meta.dtype,
		"min_intensity": meta.min_intensity,
		"max_intensity": meta.max_intensity,
		"is_grayscale": meta.is_grayscale,
	}


def _select_image(images: List[LoadedImage], label: str, index: int = 0) -> Optional[LoadedImage]:
	"""Render a select box for choosing among uploaded images."""
	if not images:
		return None
	options = [
		(f"{i + 1}: {img.name}" if img.name else f"Image {i + 1}", img)
		for i, img in enumerate(images)
	]
	labels = [label_text for label_text, _ in options]
	selection = st.selectbox(label, labels, index=min(index, len(labels) - 1))
	for label_text, img in options:
		if label_text == selection:
			return img
	return None


def _extract_features(image: np.ndarray, settings: ui.DetectorSettings) -> feature_extraction.FeatureResult:
	"""Run feature extraction with the selected detector configuration."""
	if settings.detector_type == "orb":
		detector = feature_extraction.get_detector("orb",
			nfeatures=settings.orb.nfeatures,
			scaleFactor=settings.orb.scale_factor,
			nlevels=settings.orb.nlevels,
			edgeThreshold=settings.orb.edge_threshold,
			fastThreshold=settings.orb.fast_threshold,
			patchSize=settings.orb.patch_size,
		)
	else:  # sift
		detector = feature_extraction.get_detector("sift",
			nfeatures=settings.sift.nfeatures,
			nOctaveLayers=settings.sift.n_octave_layers,
			contrastThreshold=settings.sift.contrast_threshold,
			edgeThreshold=settings.sift.edge_threshold,
			sigma=settings.sift.sigma,
		)
	return feature_extraction.extract_features(image, detector)


def main() -> None:
	"""Launch the Streamlit interface for the demo application."""
	st.set_page_config(page_title="Medical Feature Extraction", layout="wide")
	st.title("Medical Image Feature Extraction Demo")
	st.write("Upload medical PNG slices to explore feature detection, matching, and differential analysis.")

	detector_settings, preprocessing_settings, morphology_settings, segmentation_settings, matching_settings = ui.render_sidebar()

	uploaded_files = st.file_uploader("Upload PNG slices", type=["png"], accept_multiple_files=True)

	loaded_images: List[LoadedImage] = []
	if uploaded_files:
		for uploaded in uploaded_files:
			contents = uploaded.getvalue()
			loaded = _load_image(contents, uploaded.name)
			loaded_images.append(loaded)

	if not loaded_images:
		st.info("Upload at least one PNG image to begin.")
		return

	primary = _select_image(loaded_images, "Primary image", index=0)
	secondary = None
	if len(loaded_images) > 1:
		secondary = _select_image(loaded_images, "Comparison image", index=1)

	if primary is None:
		st.warning("Select a primary image from the uploader.")
		return

	processed_primary = _preprocess_image(primary.image, preprocessing_settings)

	tabs = st.tabs(["Preview", "Feature Extraction", "Differential Comparison",  "Preprocessing", "Morphology", "Segmentation"])

	with tabs[0]: # Preview
		st.subheader("Original Slice")
		st.image(visualization.to_rgb(primary.image), caption=primary.name, width="stretch")

		ui.display_metadata(_build_metadata_dict(primary.metadata))

		mean_val = float(np.mean(primary.image))
		std_val = float(np.std(primary.image))
		st.metric("Mean Intensity", f"{mean_val:.2f}")
		st.metric("Std. Dev.", f"{std_val:.2f}")

		st.subheader("Histogram")
		hist_fig = visualization.plot_histogram(primary.image)
		st.pyplot(hist_fig)

		if preprocessing_settings.normalize or preprocessing_settings.smoothing_method or primary.image.dtype != np.uint8:
			st.subheader("Preprocessed Preview")
			st.image(visualization.to_rgb(processed_primary), caption="Preprocessed", width="stretch")

	with tabs[1]: # Feature Extraction
		st.subheader("Keypoint Detection")
		features_primary = _extract_features(processed_primary, detector_settings)

		st.metric("Keypoints", features_primary.keypoint_count)
		st.metric("Descriptor Count", features_primary.descriptor_count)

		overlay = visualization.draw_keypoints(processed_primary, features_primary.keypoints)
		st.image(overlay, caption="Keypoints overlay", width="stretch")

		if features_primary.descriptors is None or features_primary.descriptor_count == 0:
			st.warning("No descriptors detected. Consider adjusting detector parameters or preprocessing.")

	with tabs[2]: # Differential Comparison
		if secondary is None:
			st.info("Upload a second image to enable comparison.")
		else:
			st.subheader("Differential Frame")
			st.write(
				"The differential frame shows the absolute pixel-wise difference between the primary and comparison slices. "
				"It highlights regions where intensity changed, helping motivate frame-to-frame analysis for video data."
			)
			processed_secondary = _preprocess_image(secondary.image, preprocessing_settings)

			if processed_primary.shape != processed_secondary.shape:
				st.warning("Images must share identical dimensions for differencing and matching.")
				st.image(visualization.to_rgb(secondary.image), caption=secondary.name, width="stretch")
			else:
				diff_image = preprocessing.compute_difference(processed_primary, processed_secondary)
				st.image(visualization.to_rgb(diff_image), caption="Absolute difference", width="stretch")

				st.subheader("Feature Matching")
				st.info(
					"Detected features from both slices are compared using ORB descriptors and Lowe's ratio test. "
					"Lines connect keypoints that remain similar between the slices, illustrating potential temporal correspondences."
				)
				features_primary = _extract_features(processed_primary, detector_settings)
				features_secondary = _extract_features(processed_secondary, detector_settings)

				matches = feature_matching.match_descriptors(features_primary.descriptors, features_secondary.descriptors)
				filtered_matches = feature_matching.filter_matches_ratio(matches, matching_settings.ratio_threshold)
				summary = feature_matching.summarize_matches(filtered_matches)

				st.metric("Initial matches", sum(len(bucket) for bucket in matches))
				st.metric("Filtered matches", summary.total_matches)
				st.metric("Average distance", f"{summary.average_distance:.2f}" if not np.isnan(summary.average_distance) else "nan")

				if summary.total_matches == 0:
					st.warning("No matches found after filtering. Try easing the ratio threshold or detector settings.")
				else:
					matched_viz = visualization.draw_matches(processed_primary, features_primary.keypoints, processed_secondary, features_secondary.keypoints, filtered_matches)
					st.image(matched_viz, caption="Feature matches", width="stretch")

	with tabs[3]: # Preprocessing
		# Preprocessing tab - placeholder for future enhancement
		st.info("Preprocessing tab - reserved for future enhancement. Use sidebar controls for preprocessing options.")

	with tabs[4]: # Morphology
		st.subheader("Morphological Operations")
		st.write(
			"Morphological operations work on binary images to clean up noise, fill holes, "
			"and detect edges. First, the image is binarized (thresholded), then the selected "
			"morphological operation is applied."
		)
		
		# Step 1: Binarization
		st.markdown("### Step 1: Binarization")
		binary_image, thresh_used = morphology.binarize(
			processed_primary,
			threshold=morphology_settings.threshold,
			method=morphology_settings.threshold_method,
		)
		
		col1, col2 = st.columns(2)
		with col1:
			st.image(visualization.to_rgb(processed_primary), caption="Preprocessed Input", use_container_width=True)
		with col2:
			st.image(visualization.to_rgb(binary_image), caption=f"Binary (threshold={thresh_used})", use_container_width=True)
		
		# Step 2: Morphological Operation
		st.markdown(f"### Step 2: {morphology_settings.operation.capitalize()}")
		
		morphed_image = morphology.apply_morphology(
			binary_image,
			operation=morphology_settings.operation,
			kernel_shape=morphology_settings.kernel_shape,
			kernel_size=morphology_settings.kernel_size,
			iterations=morphology_settings.iterations,
		)
		
		col1, col2 = st.columns(2)
		with col1:
			st.image(visualization.to_rgb(binary_image), caption="Binary Input", use_container_width=True)
		with col2:
			st.image(visualization.to_rgb(morphed_image), caption=f"After {morphology_settings.operation}", use_container_width=True)
		
		# Show settings used
		st.markdown("**Settings Used:**")
		st.json({
			"threshold_method": morphology_settings.threshold_method,
			"threshold": thresh_used,
			"operation": morphology_settings.operation,
			"kernel_shape": morphology_settings.kernel_shape,
			"kernel_size": morphology_settings.kernel_size,
			"iterations": morphology_settings.iterations,
		})

	with tabs[5]: # Segmentation
		st.subheader("Segmentation & ROI Detection")
		st.write(
			"Segmentation separates the image into meaningful regions. "
			"Otsu's method automatically finds the optimal threshold, then contour detection "
			"identifies the boundaries of each region of interest (ROI)."
		)
		
		# Step 1: Otsu Thresholding
		st.markdown("### Step 1: Otsu's Automatic Thresholding")
		binary_seg, otsu_thresh = segmentation.otsu_threshold(processed_primary)
		
		col1, col2 = st.columns(2)
		with col1:
			st.image(visualization.to_rgb(processed_primary), caption="Preprocessed Input", use_container_width=True)
		with col2:
			st.image(visualization.to_rgb(binary_seg), caption=f"Otsu Binary (threshold={otsu_thresh})", use_container_width=True)
		
		st.metric("Otsu Threshold", otsu_thresh)
		
		# Step 2: Contour Detection
		st.markdown("### Step 2: Contour Detection (ROIs)")
		contours = segmentation.find_contours(
			binary_seg,
			min_area=segmentation_settings.min_contour_area,
		)
		
		st.metric("Detected ROIs", len(contours))
		
		if contours:
			# Draw contours on original image
			contour_overlay = segmentation.draw_contours(
				processed_primary,
				contours,
				draw_contour=segmentation_settings.show_contours,
				draw_bounding_rect=segmentation_settings.show_bounding_rects,
				draw_center=segmentation_settings.show_bounding_rects,
			)
			st.image(visualization.to_rgb(contour_overlay), caption="Detected ROIs", use_container_width=True)
			
			# Show ROI details
			st.markdown("**ROI Details (sorted by area):**")
			roi_data = []
			for i, c in enumerate(contours[:10]):  # Show top 10
				roi_data.append({
					"ROI #": i + 1,
					"Area (px)": int(c.area),
					"Center": f"({c.center[0]}, {c.center[1]})",
					"Bounding Box": f"{c.bounding_rect[2]}x{c.bounding_rect[3]}",
				})
			st.table(roi_data)
		else:
			st.warning("No ROIs detected. Try lowering the 'Min Contour Area' in the sidebar.")


if __name__ == "__main__":
	main()
