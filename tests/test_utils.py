"""Tests for utility functions."""

import numpy as np
import pytest

from sdm.utils.bbox import expand_bbox, bbox_to_square, clip_bbox


def test_expand_bbox():
    """Test bounding box expansion."""
    image_size = (480, 640)  # (height, width)
    bbox = np.array([100, 100, 200, 200], dtype=np.float32)

    expanded = expand_bbox(image_size, bbox, expand_rate=0.2)

    # Check that bbox is expanded
    assert expanded[0] < bbox[0]  # x0 decreased
    assert expanded[1] < bbox[1]  # y0 decreased
    assert expanded[2] > bbox[2]  # x1 increased
    assert expanded[3] > bbox[3]  # y1 increased

    # Check bounds
    assert expanded[0] >= 0
    assert expanded[1] >= 0
    assert expanded[2] <= image_size[0]
    assert expanded[3] <= image_size[1]


def test_bbox_to_square():
    """Test converting bbox to square."""
    # Wider bbox
    bbox = np.array([100, 100, 200, 300], dtype=np.float32)
    square = bbox_to_square(bbox)

    width = square[3] - square[1]
    height = square[2] - square[0]
    assert abs(width - height) < 1.0

    # Taller bbox
    bbox = np.array([100, 100, 300, 200], dtype=np.float32)
    square = bbox_to_square(bbox)

    width = square[3] - square[1]
    height = square[2] - square[0]
    assert abs(width - height) < 1.0


def test_clip_bbox():
    """Test bbox clipping."""
    image_size = (480, 640)

    # Bbox within bounds
    bbox = np.array([100, 100, 200, 200], dtype=np.float32)
    clipped = clip_bbox(bbox, image_size)
    np.testing.assert_array_equal(clipped, [100, 100, 200, 200])

    # Bbox exceeding bounds
    bbox = np.array([-10, -10, 500, 700], dtype=np.float32)
    clipped = clip_bbox(bbox, image_size)
    assert clipped[0] == 0
    assert clipped[1] == 0
    assert clipped[2] == image_size[0]
    assert clipped[3] == image_size[1]
