"""Histogram of Oriented Gradients (HOG) feature extraction.

This module implements HOG feature extraction for face alignment.
HOG is a classical computer vision descriptor that captures local gradient
information, making it robust to illumination changes.

Reference:
    Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for
    human detection. CVPR.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from sdm.core.model import SDMConfig


class HOGExtractor:
    """HOG feature extractor for face landmark localization.

    This class extracts HOG features around specified landmark locations.
    Features are computed in a local window around each landmark point.

    Attributes:
        config: SDM configuration containing HOG parameters
    """

    def __init__(self, config: SDMConfig):
        """Initialize HOG extractor.

        Args:
            config: SDM configuration
        """
        self.config = config
        self.eps = 1e-5

    def extract(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """Extract HOG features around landmarks.

        Args:
            image: Grayscale image (H, W)
            landmarks: Landmark coordinates (N, 2) in (x, y) format

        Returns:
            Flattened HOG feature vector

        Raises:
            ValueError: If image is not 2D grayscale
        """
        image = np.atleast_2d(image)
        if image.ndim > 2:
            raise ValueError("HOG extraction requires grayscale image")

        # Normalize image
        image_norm = np.sqrt(image.astype(np.float32))

        # Compute gradients
        gx, gy = self._compute_gradients(image_norm)

        # Compute magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx + 1e-15) * (180 / np.pi) + 180

        # Extract orientation histograms
        orientation_hist = self._compute_orientation_histogram(
            magnitude,
            orientation,
            landmarks,
        )

        # Apply normalization
        if self.config.hog_no_block:
            # Cell-level normalization
            features = self._normalize_cells(orientation_hist)
        else:
            # Block-level normalization
            features = self._normalize_blocks(orientation_hist)

        return features.ravel()

    def _compute_gradients(
        self,
        image: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute image gradients using finite differences.

        Args:
            image: Normalized grayscale image

        Returns:
            Tuple of (gradient_x, gradient_y)
        """
        gx = np.zeros_like(image)
        gy = np.zeros_like(image)

        # Compute gradients using simple differences
        gx[:, :-1] = np.diff(image, n=1, axis=1)
        gy[:-1, :] = np.diff(image, n=1, axis=0)

        return gx, gy

    def _compute_orientation_histogram(
        self,
        magnitude: NDArray[np.float32],
        orientation: NDArray[np.float32],
        landmarks: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """Compute orientation histograms around landmarks.

        Args:
            magnitude: Gradient magnitudes
            orientation: Gradient orientations (in degrees, 0-360)
            landmarks: Landmark positions

        Returns:
            Orientation histograms (n_landmarks, cells, cells, orientations)
        """
        n_landmarks = len(landmarks)
        cells_per_side = self.config.cells_per_side
        cells_total = cells_per_side * 2  # Both sides
        n_orientations = self.config.orientations
        pixels_per_cell = self.config.pixels_per_cell

        # Radius of the window around each landmark
        radius = pixels_per_cell * cells_per_side

        # Initialize histogram
        hist = np.zeros(
            (n_landmarks, cells_total, cells_total, n_orientations),
            dtype=np.float32,
        )

        # Bin size for orientations (360 degrees / n_orientations)
        bin_size = 360.0 / n_orientations

        for landmark_idx, (x, y) in enumerate(landmarks):
            x, y = int(x), int(y)

            for ori_idx in range(n_orientations):
                # Define orientation bin range
                ori_start = bin_size * ori_idx
                ori_end = bin_size * (ori_idx + 1)

                # Create mask for this orientation bin
                in_bin = (orientation > ori_start) & (orientation <= ori_end)

                # Get magnitudes for this bin
                bin_magnitude = np.where(in_bin, magnitude, 0)

                # Apply uniform filter (average pooling) for cell aggregation
                filtered = uniform_filter(bin_magnitude, size=pixels_per_cell)

                # Extract cells around landmark
                # Note: Need to handle boundaries carefully
                x_start = max(0, x - radius + pixels_per_cell // 2)
                x_end = min(filtered.shape[0], x + radius + pixels_per_cell // 2)
                y_start = max(0, y - radius + pixels_per_cell // 2)
                y_end = min(filtered.shape[1], y + radius + pixels_per_cell // 2)

                # Sample at pixel_per_cell intervals
                try:
                    sampled = filtered[
                        x_start:x_end:pixels_per_cell,
                        y_start:y_end:pixels_per_cell,
                    ].T

                    # Place in histogram (handle size mismatches)
                    h, w = sampled.shape
                    h_out, w_out = min(h, cells_total), min(w, cells_total)
                    hist[landmark_idx, :h_out, :w_out, ori_idx] = sampled[:h_out, :w_out]

                except (IndexError, ValueError):
                    # Landmark too close to boundary
                    continue

        return hist

    def _normalize_cells(self, hist: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize histogram at cell level.

        Args:
            hist: Orientation histograms

        Returns:
            Normalized histograms
        """
        normalized = hist.copy()

        for i in range(len(hist)):
            norm = np.sqrt(hist[i].sum() ** 2 + self.eps)
            normalized[i] = hist[i] / norm

        return normalized

    def _normalize_blocks(self, hist: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize histogram at block level.

        Args:
            hist: Orientation histograms (n_landmarks, cells, cells, orientations)

        Returns:
            Block-normalized features
        """
        n_landmarks = hist.shape[0]
        cells_total = self.config.cells_per_side * 2
        cells_per_block = self.config.cells_per_block
        n_blocks = cells_total - cells_per_block + 1
        n_orientations = self.config.orientations

        normalized_blocks = np.zeros(
            (
                n_landmarks,
                n_blocks,
                n_blocks,
                cells_per_block,
                cells_per_block,
                n_orientations,
            ),
            dtype=np.float32,
        )

        for landmark_idx in range(n_landmarks):
            for bx in range(n_blocks):
                for by in range(n_blocks):
                    # Extract block
                    block = hist[
                        landmark_idx,
                        bx : bx + cells_per_block,
                        by : by + cells_per_block,
                        :,
                    ]

                    # Normalize block
                    norm = np.sqrt(block.sum() ** 2 + self.eps)
                    normalized_blocks[landmark_idx, bx, by, :] = block / norm

        return normalized_blocks


def extract_hog_features(
    image: NDArray[np.uint8],
    landmarks: NDArray[np.int32],
    config: SDMConfig,
) -> NDArray[np.float32]:
    """Convenience function to extract HOG features.

    Args:
        image: Grayscale image
        landmarks: Landmark coordinates
        config: SDM configuration

    Returns:
        HOG feature vector
    """
    extractor = HOGExtractor(config)
    return extractor.extract(image, landmarks)


def visualize_hog(
    image: NDArray[np.uint8],
    landmarks: NDArray[np.int32],
    config: SDMConfig,
    landmark_idx: int = 0,
) -> NDArray[np.float32]:
    """Visualize HOG features for a specific landmark.

    Args:
        image: Grayscale image
        landmarks: Landmark coordinates
        config: SDM configuration
        landmark_idx: Index of landmark to visualize

    Returns:
        HOG visualization as 2D array
    """
    extractor = HOGExtractor(config)

    # Compute gradients
    image_norm = np.sqrt(image.astype(np.float32))
    gx, gy = extractor._compute_gradients(image_norm)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx + 1e-15) * (180 / np.pi) + 180

    # Get histogram for specific landmark
    hist = extractor._compute_orientation_histogram(
        magnitude,
        orientation,
        landmarks[landmark_idx : landmark_idx + 1],
    )

    return hist[0]  # Return 3D array (cells, cells, orientations)
