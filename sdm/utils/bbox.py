"""Bounding box utilities for face detection and cropping.

This module provides robust bounding box manipulation functions.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def expand_bbox(
    image_size: Tuple[int, int],
    bbox: NDArray[np.float32],
    expand_rate: float = 0.2,
) -> Tuple[int, int, int, int]:
    """Expand bounding box while handling image boundaries.

    This function expands a bounding box by a given rate, ensuring the expanded
    box stays within image boundaries. It maintains aspect ratio by expanding
    the shorter dimension to match the longer one after rate expansion.

    Args:
        image_size: Image dimensions (height, width)
        bbox: Bounding box coordinates [x0, y0, x1, y1]
        expand_rate: Expansion rate (0.2 = 20% expansion on each side)

    Returns:
        Tuple of (x0, y0, x1, y1) for expanded bounding box

    Example:
        >>> image_size = (480, 640)
        >>> bbox = np.array([100, 100, 200, 200])
        >>> expanded = expand_bbox(image_size, bbox, expand_rate=0.2)
        >>> print(expanded)
        (80, 80, 220, 220)
    """
    height, width = image_size
    x0, y0, x1, y1 = bbox

    # Calculate current dimensions
    bbox_width = y1 - y0
    bbox_height = x1 - x0

    # Expand based on larger dimension to maintain aspect ratio
    if bbox_width > bbox_height:
        # Width is larger - expand width first, then match height
        delta = expand_rate * bbox_width

        # Expand width with boundary checking
        new_y1 = min(width, int(y1 + delta))
        new_y0 = max(0, int(y0 - delta))
        new_width = new_y1 - new_y0

        # Expand height to match
        delta_h = (new_width - bbox_height) / 2
        new_x0 = max(0, int(x0 - delta_h))
        new_x1 = min(height, int(x1 + delta_h))

    else:
        # Height is larger - expand height first, then match width
        delta = expand_rate * bbox_height

        # Expand height with boundary checking
        new_x1 = min(height, int(x1 + delta))
        new_x0 = max(0, int(x0 - delta))
        new_height = new_x1 - new_x0

        # Expand width to match
        delta_w = (new_height - bbox_width) / 2
        new_y0 = max(0, int(y0 - delta_w))
        new_y1 = min(width, int(y1 + delta_w))

    return new_x0, new_y0, new_x1, new_y1


def bbox_to_square(bbox: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert bounding box to square by expanding shorter dimension.

    Args:
        bbox: Bounding box [x0, y0, x1, y1]

    Returns:
        Square bounding box
    """
    x0, y0, x1, y1 = bbox
    width = y1 - y0
    height = x1 - x0

    if width > height:
        delta = (width - height) / 2
        x0 -= delta
        x1 += delta
    else:
        delta = (height - width) / 2
        y0 -= delta
        y1 += delta

    return np.array([x0, y0, x1, y1], dtype=np.float32)


def clip_bbox(
    bbox: NDArray[np.float32],
    image_size: Tuple[int, int],
) -> NDArray[np.int32]:
    """Clip bounding box to image boundaries.

    Args:
        bbox: Bounding box [x0, y0, x1, y1]
        image_size: Image dimensions (height, width)

    Returns:
        Clipped bounding box as integers
    """
    height, width = image_size
    x0, y0, x1, y1 = bbox

    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(height, int(x1))
    y1 = min(width, int(y1))

    return np.array([x0, y0, x1, y1], dtype=np.int32)
