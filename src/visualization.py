"""Visualization helpers for the Streamlit front-end."""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure the image is RGB for display."""

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_histogram(image: np.ndarray, bins: int = 256) -> plt.Figure:
    """Return a histogram figure for intensity distribution."""

    fig, ax = plt.subplots()
    if image.ndim == 2:
        ax.hist(image.ravel(), bins=bins, color="steelblue")
        ax.set_title("Intensity Histogram")
    else:
        colors = ("blue", "green", "red")
        channels = cv2.split(image)
        for channel, color in zip(channels, colors):
            ax.hist(channel.ravel(), bins=bins, color=color, alpha=0.5, label=color.upper())
        ax.legend()
        ax.set_title("Channel Histograms")

    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _hstack_images(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    height = max(image_a.shape[0], image_b.shape[0])
    width = image_a.shape[1] + image_b.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: image_a.shape[0], : image_a.shape[1]] = image_a
    canvas[: image_b.shape[0], image_a.shape[1] : width] = image_b
    return canvas


def draw_keypoints(image: np.ndarray, keypoints: Sequence[cv2.KeyPoint]) -> np.ndarray:
    """Overlay keypoints on the image."""

    base = _ensure_bgr(image)
    if not keypoints:
        return to_rgb(base)

    canvas = base.copy()
    cv2.drawKeypoints(
        base,
        list(keypoints),
        canvas,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return to_rgb(canvas)


def draw_matches(
    image_a: np.ndarray,
    keypoints_a: Sequence[cv2.KeyPoint],
    image_b: np.ndarray,
    keypoints_b: Sequence[cv2.KeyPoint],
    matches: Iterable[cv2.DMatch],
) -> np.ndarray:
    """Visualize matches between two images."""

    base_a = _ensure_bgr(image_a)
    base_b = _ensure_bgr(image_b)
    matches_list = list(matches)

    if not matches_list:
        return to_rgb(_hstack_images(base_a, base_b))

    canvas = cv2.drawMatches(
        base_a,
        list(keypoints_a),
        base_b,
        list(keypoints_b),
        matches_list,
        None,  # type: ignore[arg-type]
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return to_rgb(canvas)
