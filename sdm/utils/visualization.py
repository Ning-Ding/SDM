"""Visualization utilities for face landmarks.

This module provides functions to visualize facial landmarks on images.
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def draw_landmarks(
    image: NDArray[np.uint8],
    landmarks: NDArray[np.int32],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = -1,
) -> NDArray[np.uint8]:
    """Draw landmarks on image.

    Args:
        image: Input image (grayscale or RGB)
        landmarks: Facial landmarks (N, 2) in (x, y) format
        color: Circle color in RGB
        radius: Circle radius
        thickness: Circle thickness (-1 for filled)

    Returns:
        Image with landmarks drawn
    """
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = image.copy()

    # Draw each landmark
    for x, y in landmarks:
        cv2.circle(vis_image, (int(x), int(y)), radius, color, thickness)

    return vis_image


def plot_landmarks(
    image: NDArray[np.uint8],
    landmarks_list: list[NDArray[np.int32]],
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
) -> None:
    """Plot image with multiple sets of landmarks.

    Useful for comparing predicted vs ground truth landmarks.

    Args:
        image: Input image
        landmarks_list: List of landmark arrays to plot
        labels: Labels for each landmark set
        colors: Colors for each landmark set
        figsize: Figure size
        title: Plot title
    """
    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(landmarks_list))]

    if colors is None:
        colors = ["red", "blue", "green", "yellow", "purple"]

    plt.figure(figsize=figsize)

    # Display image
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    # Plot each landmark set
    for landmarks, label, color in zip(landmarks_list, labels, colors):
        plt.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            c=color,
            s=50,
            alpha=0.7,
            label=label,
        )

    plt.legend()
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_hog_cell(
    image: NDArray[np.uint8],
    landmarks: NDArray[np.int32],
    cell_size: int = 10,
) -> NDArray[np.uint8]:
    """Visualize HOG cells around landmarks.

    Draws rectangles showing HOG cell regions around each landmark.

    Args:
        image: Input image
        landmarks: Facial landmarks
        cell_size: Size of HOG cells

    Returns:
        Image with cell visualizations
    """
    vis_image = image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)

    for x, y in landmarks:
        # Draw cell rectangle
        top_left = (int(x - cell_size), int(y - cell_size))
        bottom_right = (int(x + cell_size), int(y + cell_size))
        cv2.rectangle(vis_image, top_left, bottom_right, (0, 255, 0), 1)

        # Draw center point
        cv2.circle(vis_image, (int(x), int(y)), 2, (255, 0, 0), -1)

    return vis_image


def plot_training_curve(
    mse_history: list[list[float]],
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot MSE across iterations for all samples.

    Args:
        mse_history: List of MSE values per sample per iteration
        figsize: Figure size
    """
    mse_array = np.array(mse_history)
    mean_mse = mse_array.mean(axis=0)
    std_mse = mse_array.std(axis=0)

    iterations = np.arange(1, len(mean_mse) + 1)

    plt.figure(figsize=figsize)
    plt.plot(iterations, mean_mse, "b-", linewidth=2, label="Mean MSE")
    plt.fill_between(
        iterations,
        mean_mse - std_mse,
        mean_mse + std_mse,
        alpha=0.3,
        color="blue",
        label="±1 std",
    )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Square Error", fontsize=12)
    plt.title("SDM Training Progress", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
