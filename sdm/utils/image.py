"""Image processing utilities for face alignment.

This module provides image loading, preprocessing, and transformation functions.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from sdm.core.model import SDMConfig
from sdm.utils.bbox import expand_bbox


def load_image(image_path: str | Path, grayscale: bool = False) -> NDArray[np.uint8]:
    """Load image from file path.

    Args:
        image_path: Path to image file
        grayscale: Whether to convert to grayscale

    Returns:
        Image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if grayscale:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def crop_and_resize(
    image: NDArray[np.uint8],
    bbox: NDArray[np.float32],
    landmarks: NDArray[np.float32],
    config: SDMConfig,
) -> Tuple[NDArray[np.uint8], NDArray[np.int32]]:
    """Crop and resize image with corresponding landmark transformation.

    This function performs the following steps:
    1. Expands bounding box
    2. Crops image
    3. Adds padding
    4. Resizes to target size
    5. Transforms landmarks accordingly
    6. Converts to grayscale

    Args:
        image: Input image (H, W, 3) or (H, W)
        bbox: Bounding box [x0, y0, x1, y1]
        landmarks: Facial landmarks (N, 2)
        config: SDM configuration

    Returns:
        Tuple of (processed_gray_image, transformed_landmarks)

    Example:
        >>> image = load_image("face.jpg")
        >>> bbox = np.array([100, 100, 300, 300])
        >>> landmarks = np.array([[150, 150], [200, 200]])
        >>> config = SDMConfig()
        >>> gray, new_landmarks = crop_and_resize(image, bbox, landmarks, config)
    """
    # Expand bounding box
    image_size = image.shape[:2]  # (height, width)
    expanded_bbox = expand_bbox(image_size, bbox, config.expand_rate)

    # Convert to PIL for easier manipulation
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(image)

    # Crop image
    # PIL uses (left, upper, right, lower) = (x0, y0, x1, y1)
    # bbox is [x0, y0, x1, y1] where x is horizontal, y is vertical
    x0, y0, x1, y1 = expanded_bbox
    cropped = pil_image.crop((x0, y0, x1, y1))

    # Add padding
    expand = config.expand_pixels
    padded = ImageOps.expand(cropped, border=expand, fill="black")

    # Resize to target size
    resized = padded.resize(config.image_size)

    # Convert to grayscale
    gray = resized.convert("L")
    gray_array = np.array(gray)

    # Transform landmarks
    # 1. Adjust for crop (landmarks are in (x, y) format)
    new_landmarks = landmarks.copy() - np.array([x0, y0])
    # 2. Adjust for padding
    new_landmarks = new_landmarks + expand
    # 3. Adjust for resize
    scale_x = config.image_size[0] / padded.size[0]
    scale_y = config.image_size[1] / padded.size[1]
    new_landmarks = new_landmarks * np.array([scale_x, scale_y])

    return gray_array, new_landmarks.astype(np.int32)


def normalize_image(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Normalize image to [0, 1] range and apply sqrt normalization.

    This is the normalization used in HOG feature extraction.

    Args:
        image: Input grayscale image (uint8)

    Returns:
        Normalized image (float32)
    """
    # Convert to float and normalize to [0, 1]
    normalized = image.astype(np.float32) / 255.0

    # Apply sqrt normalization (common in HOG)
    normalized = np.sqrt(normalized)

    return normalized


def resize_with_aspect_ratio(
    image: NDArray[np.uint8],
    target_size: int,
) -> NDArray[np.uint8]:
    """Resize image maintaining aspect ratio.

    Args:
        image: Input image
        target_size: Target size for longer dimension

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    if height > width:
        new_height = target_size
        new_width = int(width * target_size / height)
    else:
        new_width = target_size
        new_height = int(height * target_size / width)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized
