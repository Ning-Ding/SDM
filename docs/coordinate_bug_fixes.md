# Critical Coordinate System Bug Fixes

**Date**: 2025-11-08
**Severity**: CRITICAL
**Impact**: All training and inference affected
**Status**: ✅ Fixed in commit `978bf00`

---

## Executive Summary

Two **critical bugs** were discovered involving systematic confusion of coordinate axes throughout the codebase. These bugs caused:

1. **Wrong image regions to be cropped** (transposed)
2. **Landmarks to be incorrectly transformed**
3. **HOG features extracted from wrong pixel locations**
4. **All training to learn on corrupted/misaligned data**

**Any models trained before this fix learned incorrect feature-to-landmark mappings.**

---

## Bug #1: Bounding Box Coordinate Confusion

### The Problem

The bounding box format is documented as `[x0, y0, x1, y1]` where:
- `x` = horizontal coordinate (column)
- `y` = vertical coordinate (row)

However, the code **systematically confused** these axes in multiple functions.

### Affected Functions

1. **`expand_bbox()`** (`sdm/utils/bbox.py`)
2. **`bbox_to_square()`** (`sdm/utils/bbox.py`)
3. **`clip_bbox()`** (`sdm/utils/bbox.py`)
4. **`crop_and_resize()`** (`sdm/utils/image.py`)

### Specific Errors

#### Error in `expand_bbox()`:

```python
# ❌ WRONG
bbox_width = y1 - y0    # Treating y as width!
bbox_height = x1 - x0   # Treating x as height!

new_y1 = min(width, int(y1 + delta))    # y should be limited by height
new_x1 = min(height, int(x1 + delta))   # x should be limited by width

# ✅ CORRECT
bbox_width = x1 - x0    # x is horizontal (width)
bbox_height = y1 - y0   # y is vertical (height)

new_x1 = min(width, int(x1 + delta))    # x limited by width
new_y1 = min(height, int(y1 + delta))   # y limited by height
```

#### Error in `crop_and_resize()`:

```python
# ❌ WRONG
# Comment was incorrect!
# PIL uses (left, upper, right, lower) = (y0, x0, y1, x1)  # WRONG!
x0, y0, x1, y1 = expanded_bbox
cropped = pil_image.crop((y0, x0, y1, x1))  # Swapped coordinates!

# Landmark adjustment also wrong
new_landmarks = landmarks.copy() - np.array([y0, x0])  # Swapped!

# ✅ CORRECT
# PIL uses (left, upper, right, lower) = (x0, y0, x1, y1)
x0, y0, x1, y1 = expanded_bbox
cropped = pil_image.crop((x0, y0, x1, y1))  # Correct order

# Landmarks are (x, y), so subtract [x0, y0]
new_landmarks = landmarks.copy() - np.array([x0, y0])  # Correct!
```

### Impact

- **Cropped wrong image regions**: The crop was effectively transposed
- **Landmarks misaligned**: Transformation used wrong offset
- **Training data corrupted**: Model learned on wrong face regions

---

## Bug #2: HOG Feature Extraction Coordinate Confusion

### The Problem

**Landmarks** are stored as `(x, y)` coordinates where:
- `x` = horizontal position (column)
- `y` = vertical position (row)

**NumPy arrays** are indexed as `[row, col]` which equals `[y, x]`:
- `array.shape[0]` = height (number of rows)
- `array.shape[1]` = width (number of columns)
- `array[row, col]` = `array[y, x]`

The code used `array[x, y]` instead of `array[y, x]`, extracting features from the **wrong pixel neighborhoods**.

### Affected Function

**`HOGExtractor._compute_orientation_histogram()`** (`sdm/features/hog.py`)

### Specific Error

```python
# ❌ WRONG
for landmark_idx, (x, y) in enumerate(landmarks):
    x, y = int(x), int(y)

    # Boundary checks were wrong
    x_start = max(0, x - radius + ...)
    x_end = min(filtered.shape[0], x + radius + ...)  # shape[0] is height!
    y_start = max(0, y - radius + ...)
    y_end = min(filtered.shape[1], y + radius + ...)  # shape[1] is width!

    # Indexing was backwards!
    sampled = filtered[
        x_start:x_end:pixels_per_cell,  # Using x for first index (rows)
        y_start:y_end:pixels_per_cell,  # Using y for second index (cols)
    ].T

# ✅ CORRECT
for landmark_idx, (x, y) in enumerate(landmarks):
    x, y = int(x), int(y)

    # Correct boundary checks
    x_start = max(0, x - radius + ...)
    x_end = min(filtered.shape[1], x + radius + ...)  # x limits by width
    y_start = max(0, y - radius + ...)
    y_end = min(filtered.shape[0], y + radius + ...)  # y limits by height

    # Correct indexing: [row, col] = [y, x]
    sampled = filtered[
        y_start:y_end:pixels_per_cell,  # y for rows
        x_start:x_end:pixels_per_cell,  # x for columns
    ].T
```

### Impact

- **Features extracted from wrong locations**: The pixel neighborhood was transposed
- **Feature-landmark mismatch**: Features didn't correspond to intended facial points
- **Regression learned wrong patterns**: Training associated features with wrong landmarks

---

## Coordinate System Conventions

To prevent future confusion, here are the **clear conventions**:

### Bounding Box Format

```python
bbox = [x0, y0, x1, y1]

where:
  x0, x1: horizontal coordinates (columns), range [0, width]
  y0, y1: vertical coordinates (rows), range [0, height]

Example:
  image.shape = (480, 640)  # height=480, width=640
  bbox = [100, 50, 300, 250]
  # x0=100, y0=50, x1=300, y1=250
  # Width = x1 - x0 = 200
  # Height = y1 - y0 = 200
```

### Landmark Format

```python
landmarks = [(x, y), (x, y), ...]

where:
  x: horizontal position (column)
  y: vertical position (row)
```

### NumPy Array Indexing

```python
image.shape = (height, width) = (num_rows, num_cols)
image[row, col] = image[y, x]

# Access pixel at landmark (x, y):
pixel_value = image[y, x]  # NOT image[x, y]!

# Slice using x, y coordinates:
region = image[y_start:y_end, x_start:x_end]  # [rows, cols] = [y, x]
```

### PIL Image Coordinates

```python
from PIL import Image

# Image.crop(box) where box = (left, upper, right, lower)
bbox = [x0, y0, x1, y1]  # Our format
pil_box = (x0, y0, x1, y1)  # PIL format - SAME!

cropped = image.crop((x0, y0, x1, y1))

# Image.size = (width, height)
width, height = image.size
```

---

## Fixes Applied

### Files Modified

1. **`sdm/utils/bbox.py`** (49 lines changed)
   - Fixed `expand_bbox()`: Correct width/height calculation and boundary checks
   - Fixed `bbox_to_square()`: Correct dimension expansion
   - Fixed `clip_bbox()`: Correct x/y limits

2. **`sdm/utils/image.py`** (9 lines changed)
   - Fixed `crop_and_resize()`: Correct PIL.crop() parameter order
   - Fixed `crop_and_resize()`: Correct landmark transformation offset

3. **`sdm/features/hog.py`** (10 lines changed)
   - Fixed `_compute_orientation_histogram()`: Correct boundary checks
   - Fixed `_compute_orientation_histogram()`: Correct numpy indexing

4. **`tests/test_coordinate_fixes.py`** (182 lines added)
   - Comprehensive coordinate system tests
   - Boundary checking tests
   - Integration tests

---

## Testing

### New Tests Added

```python
# Test bbox coordinate system
def test_bbox_coordinate_system():
    """Verify bbox uses [x0, y0, x1, y1] correctly."""

# Test boundary clipping
def test_clip_bbox_boundaries():
    """Verify x limited by width, y by height."""

# Test crop and transformation
def test_crop_and_landmark_transformation():
    """Verify crop and landmark transform are consistent."""

# Test HOG indexing
def test_hog_coordinate_indexing():
    """Verify HOG uses correct array indexing."""

# Integration test
def test_consistency_across_pipeline():
    """Verify coordinates consistent end-to-end."""
```

### Running Tests

```bash
# Run coordinate fix tests
pytest tests/test_coordinate_fixes.py -v

# Run all tests
pytest tests/ -v
```

---

## Migration Guide

### For Existing Models

⚠️ **IMPORTANT**: Any models trained **before this fix** learned on incorrectly aligned data.

**Recommendation**:
1. **Retrain all models** with the fixed code
2. **Do not use** models trained before commit `978bf00`
3. Previous models have incorrect feature-to-landmark mappings

### For Existing Code

If you have custom code using these functions:

```python
# Before (WRONG)
bbox = [x0, y0, x1, y1]
width = y1 - y0   # Wrong!
height = x1 - x0  # Wrong!

# After (CORRECT)
bbox = [x0, y0, x1, y1]
width = x1 - x0   # Correct
height = y1 - y0  # Correct
```

---

## Performance Impact

**Runtime**: No change
**Memory**: No change
**Accuracy**: ✅ **Significantly improved** (now learns correct patterns)

---

## Acknowledgments

Special thanks to the code reviewer who identified these critical issues!

These bugs demonstrate the importance of:
- Clear coordinate system documentation
- Comprehensive testing
- Code review

---

## References

- Commit: `978bf00`
- CHANGELOG: [Unreleased] section
- Related Tests: `tests/test_coordinate_fixes.py`

---

*Last updated: 2025-11-08*
