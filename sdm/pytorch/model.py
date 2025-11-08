"""PyTorch models for face landmark detection.

This module provides CNN-based models as modern alternatives to SDM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LandmarkCNN(nn.Module):
    """Simple CNN for facial landmark detection.

    This is a baseline CNN architecture for educational purposes,
    demonstrating the difference between classical and deep learning approaches.

    Architecture:
        - 4 convolutional blocks (conv + batch_norm + relu + pool)
        - 2 fully connected layers
        - Output: (x, y) coordinates for all landmarks

    Args:
        n_landmarks: Number of landmarks (default: 68 for 68-point face model)
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
    """

    def __init__(self, n_landmarks: int = 68, input_channels: int = 1):
        super().__init__()
        self.n_landmarks = n_landmarks

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # For 400x400 input: after 4 pooling layers -> 25x25
        # Feature dimension: 256 * 25 * 25 = 156,800
        self.fc1 = nn.Linear(256 * 25 * 25, 1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)

        # Output layer: n_landmarks * 2 (x, y coordinates)
        self.fc3 = nn.Linear(512, n_landmarks * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Predicted landmarks (B, n_landmarks * 2)
        """
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LandmarkResNet(nn.Module):
    """ResNet-based model for facial landmark detection.

    A more advanced architecture using residual connections,
    typically achieving better performance than simple CNN.

    Args:
        n_landmarks: Number of landmarks
        input_channels: Number of input channels
        num_blocks: Number of residual blocks per stage
    """

    def __init__(
        self,
        n_landmarks: int = 68,
        input_channels: int = 1,
        num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.n_landmarks = n_landmarks

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.stage1 = self._make_stage(64, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(64, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks[3], stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, n_landmarks * 2)

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a stage of residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Predicted landmarks (B, n_landmarks * 2)
        """
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class WingLoss(nn.Module):
    """Wing Loss for robust landmark detection.

    Wing Loss is designed specifically for face alignment tasks,
    providing better gradient properties near ground truth.

    Reference:
        Feng, Z. H., et al. (2018). Wing loss for robust facial
        landmark localisation with convolutional neural networks. CVPR.

    Args:
        omega: Width parameter
        epsilon: Curvature parameter
    """

    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Wing Loss.

        Args:
            pred: Predicted landmarks (B, n_landmarks * 2)
            target: Ground truth landmarks (B, n_landmarks * 2)

        Returns:
            Loss value
        """
        delta = (target - pred).abs()

        # Wing loss computation
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1.0 + delta / self.epsilon),
            delta - self.C,
        )

        return loss.mean()
