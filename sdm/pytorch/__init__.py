"""PyTorch-based modern face alignment implementation.

This module provides deep learning alternatives to classical SDM
for educational comparison.
"""

from sdm.pytorch.model import LandmarkCNN, LandmarkResNet
from sdm.pytorch.trainer import PyTorchTrainer
from sdm.pytorch.dataset import PyTorchFaceDataset

__all__ = ["LandmarkCNN", "LandmarkResNet", "PyTorchTrainer", "PyTorchFaceDataset"]
