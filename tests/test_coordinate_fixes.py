"""Tests for coordinate system fixes.

These tests verify the critical bug fixes for coordinate axis confusion:
1. Bounding box crop and landmark transformation
2. HOG feature extraction coordinate indexing
"""

import numpy as np
import pytest
from PIL import Image

from sdm.core.model import SDMConfig
from sdm.features.hog import HOGExtractor
from sdm.utils.bbox import clip_bbox, expand_bbox
from sdm.utils.image import crop_and_resize


def test_bbox_coordinate_system():
    """Test that bbox uses correct coordinate system [x0, y0, x1, y1]."""
    # Image: 480 height x 640 width
    image_size = (480, 640)

    # Bbox: [x0, y0, x1, y1] where x is horizontal (0-640), y is vertical (0-480)
    bbox = np.array([100, 100, 200, 200], dtype=np.float32)

    expanded = expand_bbox(image_size, bbox, expand_rate=0.2)

    # Expanded x coords should be within [0, 640]
    assert 0 <= expanded[0] < expanded[2] <= 640, f"x coords out of range: {expanded}"

    # Expanded y coords should be within [0, 480]
    assert 0 <= expanded[1] < expanded[3] <= 480, f"y coords out of range: {expanded}"

    # Width and height should be roughly equal (square-ish)
    width = expanded[2] - expanded[0]
    height = expanded[3] - expanded[1]
    assert abs(width - height) < 10, f"Not square: width={width}, height={height}"


def test_clip_bbox_boundaries():
    """Test bbox clipping respects correct coordinate boundaries."""
    image_size = (480, 640)  # height x width

    # Bbox extending beyond boundaries
    bbox = np.array([-10, -10, 700, 500], dtype=np.float32)

    clipped = clip_bbox(bbox, image_size)

    # x should be clipped to [0, width=640]
    assert clipped[0] == 0
    assert clipped[2] == 640

    # y should be clipped to [0, height=480]
    assert clipped[1] == 0
    assert clipped[3] == 480


def test_crop_and_landmark_transformation():
    """Test that crop and landmark transformation use consistent coordinates."""
    # Create test image: 400x600 (height x width)
    image = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)

    # Bbox in center: [x0=200, y0=100, x1=400, y1=300]
    # This should crop a 200x200 region from center of image
    bbox = np.array([200, 100, 400, 300], dtype=np.float32)

    # Landmarks at known positions within bbox
    landmarks = np.array(
        [
            [250, 150],  # (x, y) - center-left of bbox
            [350, 250],  # (x, y) - bottom-right of bbox
        ],
        dtype=np.float32,
    )

    config = SDMConfig(
        image_size=(200, 200), expand_pixels=0, expand_rate=0.0  # No expansion for simple test
    )

    cropped_img, transformed_landmarks = crop_and_resize(image, bbox, landmarks, config)

    # Check image shape
    assert cropped_img.shape == (200, 200), f"Wrong shape: {cropped_img.shape}"

    # Check landmarks are transformed correctly
    # Original landmarks relative to bbox:
    # [250, 150] - [200, 100] = [50, 50]
    # [350, 250] - [200, 100] = [150, 150]
    # After resize (no change since bbox is 200x200): same

    expected_landmarks = np.array([[50, 50], [150, 150]], dtype=np.int32)

    np.testing.assert_allclose(
        transformed_landmarks, expected_landmarks, atol=2, err_msg="Landmark transformation incorrect"
    )


def test_hog_coordinate_indexing():
    """Test that HOG extraction uses correct coordinate indexing."""
    config = SDMConfig()
    extractor = HOGExtractor(config)

    # Create simple test image: 100x100
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # Landmarks at known positions (x, y format)
    landmarks = np.array(
        [
            [50, 50],  # Center
            [25, 25],  # Top-left region
            [75, 75],  # Bottom-right region
        ],
        dtype=np.int32,
    )

    # Extract features - should not crash
    features = extractor.extract(image, landmarks)

    # Check output shape is reasonable
    expected_dim = config.get_hog_feature_dim()
    assert features.shape == (expected_dim,), f"Feature shape mismatch: {features.shape}"


def test_pil_crop_coordinates():
    """Test that PIL crop receives correct coordinate order."""
    # Create test image with known pattern
    # Make it identifiable: gradient from left to right
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    for x in range(200):
        image[:, x, :] = x  # Gradient in x direction

    # Convert to PIL
    pil_image = Image.fromarray(image)

    # Crop region: [x0=50, y0=25, x1=150, y1=75]
    # Should extract 100x50 region
    bbox = (50, 25, 150, 75)  # (left, upper, right, lower) = (x0, y0, x1, y1)

    cropped = pil_image.crop(bbox)

    # Check size
    assert cropped.size == (100, 50), f"Wrong crop size: {cropped.size}"

    # Check that gradient is preserved in x direction
    cropped_array = np.array(cropped)

    # Left edge should have value ~50, right edge should have value ~150
    left_value = cropped_array[25, 0, 0]  # Middle row, left column
    right_value = cropped_array[25, 99, 0]  # Middle row, right column

    assert 45 <= left_value <= 55, f"Left edge value wrong: {left_value}"
    assert 145 <= right_value <= 155, f"Right edge value wrong: {right_value}"


def test_consistency_across_pipeline():
    """Integration test: ensure coordinates are consistent through pipeline."""
    # Create synthetic data
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Bbox and landmarks in consistent coordinate system
    bbox = np.array([100, 100, 300, 300], dtype=np.float32)  # [x0, y0, x1, y1]
    landmarks = np.array(
        [
            [150, 150],  # Inside bbox
            [250, 250],  # Inside bbox
        ],
        dtype=np.float32,
    )

    config = SDMConfig()

    # Preprocess
    processed_img, processed_landmarks = crop_and_resize(image, bbox, landmarks, config)

    # Extract HOG features - should work without errors
    extractor = HOGExtractor(config)
    features = extractor.extract(processed_img, processed_landmarks)

    # Verify features were extracted
    assert features.size > 0
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()
