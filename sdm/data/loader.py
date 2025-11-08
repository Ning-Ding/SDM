"""Data loading utilities for face alignment datasets.

This module provides functions to load bounding boxes, landmarks,
and images from various face alignment datasets.
"""

from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy import io


def load_bounding_boxes(
    mat_file_path: str | Path,
) -> Dict[str, NDArray[np.float32]]:
    """Load bounding boxes from MATLAB .mat file.

    Args:
        mat_file_path: Path to .mat file containing bounding boxes

    Returns:
        Dictionary mapping image filename to bounding box [x0, y0, x1, y1]

    Raises:
        FileNotFoundError: If mat file doesn't exist
        KeyError: If mat file doesn't contain 'bounding_boxes' key

    Example:
        >>> bbox_dict = load_bounding_boxes('bounding_boxes_lfpw_trainset.mat')
        >>> print(bbox_dict['image_0001.png'])
        [100.5 120.3 250.8 270.9]
    """
    mat_file_path = Path(mat_file_path)

    if not mat_file_path.exists():
        raise FileNotFoundError(f"Bounding box file not found: {mat_file_path}")

    # Load .mat file
    mat_data = io.loadmat(str(mat_file_path))

    if "bounding_boxes" not in mat_data:
        raise KeyError("Mat file doesn't contain 'bounding_boxes' key")

    # Parse bounding boxes structure
    bbox_data = mat_data["bounding_boxes"][0]
    bbox_dict = {}

    for item in bbox_data:
        filename = item[0][0]  # Image filename
        bbox = item[1][0]  # Bounding box coordinates
        bbox_dict[filename] = bbox.astype(np.float32)

    return bbox_dict


def load_landmarks_from_pts(
    pts_file_path: str | Path,
) -> NDArray[np.float32]:
    """Load facial landmarks from .pts file.

    The .pts format is commonly used in face alignment datasets.
    Format example:
        version: 1
        n_points: 68
        {
        x1 y1
        x2 y2
        ...
        }

    Args:
        pts_file_path: Path to .pts file

    Returns:
        Landmarks array of shape (N, 2) with (x, y) coordinates

    Raises:
        FileNotFoundError: If pts file doesn't exist

    Example:
        >>> landmarks = load_landmarks_from_pts('image_0001.pts')
        >>> print(landmarks.shape)
        (68, 2)
    """
    pts_file_path = Path(pts_file_path)

    if not pts_file_path.exists():
        raise FileNotFoundError(f"Landmarks file not found: {pts_file_path}")

    with open(pts_file_path, "r") as f:
        lines = [line.strip() for line in f]

    # Find content between { and }
    try:
        start_idx = lines.index("{") + 1
        end_idx = lines.index("}")
    except ValueError:
        raise ValueError(f"Invalid .pts file format: {pts_file_path}")

    # Parse coordinates
    landmarks = []
    for line in lines[start_idx:end_idx]:
        if line:
            coords = line.split()
            if len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                landmarks.append([x, y])

    return np.array(landmarks, dtype=np.float32)


def load_landmarks_from_txt(
    txt_file_path: str | Path,
) -> NDArray[np.float32]:
    """Load landmarks from simple text file (one point per line).

    Args:
        txt_file_path: Path to text file

    Returns:
        Landmarks array of shape (N, 2)
    """
    txt_file_path = Path(txt_file_path)

    if not txt_file_path.exists():
        raise FileNotFoundError(f"Landmarks file not found: {txt_file_path}")

    landmarks = np.loadtxt(txt_file_path, dtype=np.float32)

    # Ensure shape is (N, 2)
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(-1, 2)

    return landmarks


def get_image_list(
    image_dir: str | Path,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> list[Path]:
    """Get list of image files in directory.

    Args:
        image_dir: Directory containing images
        extensions: Tuple of valid image extensions

    Returns:
        List of image file paths

    Raises:
        NotADirectoryError: If image_dir is not a directory
    """
    image_dir = Path(image_dir)

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))

    return sorted(image_files)


def save_landmarks(
    landmarks: NDArray[np.float32],
    output_path: str | Path,
    format: str = "pts",
) -> None:
    """Save landmarks to file.

    Args:
        landmarks: Landmarks array (N, 2)
        output_path: Output file path
        format: Output format ('pts' or 'txt')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pts":
        with open(output_path, "w") as f:
            f.write("version: 1\n")
            f.write(f"n_points: {len(landmarks)}\n")
            f.write("{\n")
            for x, y in landmarks:
                f.write(f"{x:.6f} {y:.6f}\n")
            f.write("}\n")

    elif format == "txt":
        np.savetxt(output_path, landmarks, fmt="%.6f")

    else:
        raise ValueError(f"Unsupported format: {format}")
