"""Face alignment dataset classes.

This module provides PyTorch-style dataset classes for face alignment tasks.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from sdm.core.model import SDMConfig
from sdm.data.loader import (
    get_image_list,
    load_bounding_boxes,
    load_landmarks_from_pts,
)
from sdm.utils.image import crop_and_resize, load_image


class FaceAlignmentDataset:
    """Base class for face alignment datasets.

    This class provides a common interface for loading face images,
    bounding boxes, and landmarks.
    """

    def __init__(
        self,
        image_dir: str | Path,
        bbox_dict: Dict[str, NDArray[np.float32]],
        landmarks_dir: str | Path,
        config: SDMConfig,
        transform: Optional[callable] = None,
    ):
        """Initialize dataset.

        Args:
            image_dir: Directory containing images
            bbox_dict: Dictionary mapping filenames to bounding boxes
            landmarks_dir: Directory containing landmark files
            config: SDM configuration
            transform: Optional transform function
        """
        self.image_dir = Path(image_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.bbox_dict = bbox_dict
        self.config = config
        self.transform = transform

        # Get list of images that have both bbox and landmarks
        self.image_files = self._get_valid_images()

    def _get_valid_images(self) -> list[Path]:
        """Get list of images with valid bbox and landmarks."""
        all_images = get_image_list(self.image_dir)
        valid_images = []

        for img_path in all_images:
            filename = img_path.name
            stem = img_path.stem

            # Check if bbox exists
            if filename not in self.bbox_dict:
                continue

            # Check if landmarks file exists
            pts_file = self.landmarks_dir / f"{stem}.pts"
            if not pts_file.exists():
                continue

            valid_images.append(img_path)

        return valid_images

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32]]:
        """Get dataset item.

        Args:
            idx: Index

        Returns:
            Tuple of (processed_image, landmarks, original_landmarks)
        """
        img_path = self.image_files[idx]
        filename = img_path.name
        stem = img_path.stem

        # Load image
        image = load_image(img_path, grayscale=False)

        # Load bounding box
        bbox = self.bbox_dict[filename]

        # Load landmarks
        pts_file = self.landmarks_dir / f"{stem}.pts"
        landmarks = load_landmarks_from_pts(pts_file)

        # Crop and resize
        processed_image, processed_landmarks = crop_and_resize(
            image, bbox, landmarks, self.config
        )

        if self.transform:
            processed_image, processed_landmarks = self.transform(
                processed_image, processed_landmarks
            )

        return processed_image, processed_landmarks, landmarks

    def get_raw(self, idx: int) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """Get raw image and landmarks without preprocessing.

        Args:
            idx: Index

        Returns:
            Tuple of (image, landmarks)
        """
        img_path = self.image_files[idx]
        stem = img_path.stem

        image = load_image(img_path, grayscale=False)
        pts_file = self.landmarks_dir / f"{stem}.pts"
        landmarks = load_landmarks_from_pts(pts_file)

        return image, landmarks


class LFPWDataset(FaceAlignmentDataset):
    """LFPW (Labeled Face Parts in the Wild) dataset.

    This dataset contains images with 68 facial landmark annotations.

    Dataset structure:
        data/
        ├── trainset/
        │   ├── png/          # Images
        │   └── pts/          # Landmark files
        ├── testset/
        │   ├── png/
        │   └── pts/
        └── bounding_boxes/
            ├── bounding_boxes_lfpw_trainset.mat
            └── bounding_boxes_lfpw_testset.mat
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        config: Optional[SDMConfig] = None,
        transform: Optional[callable] = None,
    ):
        """Initialize LFPW dataset.

        Args:
            data_root: Root directory of LFPW dataset
            split: 'train' or 'test'
            config: SDM configuration
            transform: Optional transform function
        """
        if config is None:
            config = SDMConfig()

        data_root = Path(data_root)

        # Set mode
        config.mode = split

        # Construct paths
        if split == "train":
            image_dir = data_root / "trainset" / "png"
            landmarks_dir = data_root / "trainset" / "pts"
            bbox_file = data_root / "bounding_boxes" / "bounding_boxes_lfpw_trainset.mat"
        else:
            image_dir = data_root / "testset" / "png"
            landmarks_dir = data_root / "testset" / "pts"
            bbox_file = data_root / "bounding_boxes" / "bounding_boxes_lfpw_testset.mat"

        # Load bounding boxes
        bbox_dict = load_bounding_boxes(bbox_file)

        super().__init__(
            image_dir=image_dir,
            bbox_dict=bbox_dict,
            landmarks_dir=landmarks_dir,
            config=config,
            transform=transform,
        )

        self.split = split

    def __repr__(self) -> str:
        return f"LFPWDataset(split={self.split}, n_samples={len(self)})"


class Dataset300W(FaceAlignmentDataset):
    """300W dataset wrapper.

    The 300W dataset is a collection of multiple face datasets with
    unified 68-point annotations.

    This is a placeholder for future implementation.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        config: Optional[SDMConfig] = None,
    ):
        raise NotImplementedError("300W dataset support coming soon")
