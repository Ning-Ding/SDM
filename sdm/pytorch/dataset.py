"""PyTorch dataset wrapper for face alignment."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

from sdm.data.dataset import FaceAlignmentDataset


class PyTorchFaceDataset(Dataset):
    """PyTorch wrapper for face alignment datasets.

    Converts numpy arrays to torch tensors and provides
    PyTorch DataLoader compatibility.

    Args:
        base_dataset: Base face alignment dataset
        transform: Optional torchvision transforms
        normalize: Whether to normalize images to [0, 1]
    """

    def __init__(
        self,
        base_dataset: FaceAlignmentDataset,
        transform=None,
        normalize: bool = True,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item as PyTorch tensors.

        Args:
            idx: Index

        Returns:
            Tuple of (image_tensor, landmarks_tensor)
        """
        image, landmarks, _ = self.base_dataset[idx]

        # Convert to tensor
        # Image: (H, W) -> (1, H, W) for grayscale
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        if self.normalize:
            image_tensor = image_tensor / 255.0

        # Landmarks: (N, 2) -> (N*2,) flattened
        landmarks_tensor = torch.from_numpy(landmarks).flatten().float()

        # Apply transforms if any
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, landmarks_tensor


def create_dataloaders(
    train_dataset: FaceAlignmentDataset,
    test_dataset: FaceAlignmentDataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test dataloaders.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_pytorch = PyTorchFaceDataset(train_dataset)
    test_pytorch = PyTorchFaceDataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_pytorch,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_pytorch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
