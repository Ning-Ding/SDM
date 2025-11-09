"""SDM configuration and model parameters.

This module defines the configuration class for SDM training and inference.
Uses Pydantic for validation and type safety.
"""

from typing import Literal, Tuple

from pydantic import BaseModel, Field


class SDMConfig(BaseModel):
    """Configuration for Supervised Descent Method.

    This class manages all hyperparameters for SDM training and inference.

    Attributes:
        n_iterations: Number of descent iterations (N in original paper)
        alpha: L1 regularization parameter for Lasso regression (0 = no regularization)
        image_size: Target size for resizing images (width, height)
        expand_pixels: Pixels to expand around crop for padding
        expand_rate: Rate to expand bounding box before cropping

        # HOG Feature Parameters
        orientations: Number of gradient orientation bins
        pixels_per_cell: Pixels per HOG cell
        cells_per_block: Cells per HOG block for normalization
        cells_per_side: Cells on each side of landmark
        hog_no_block: If True, skip block normalization (faster)

        # Runtime
        mode: 'train' or 'test' mode
        verbose: Enable verbose output
        device: Device for PyTorch operations
    """

    # Algorithm parameters
    n_iterations: int = Field(default=3, ge=1, le=10, description="Number of SDM iterations")
    alpha: float = Field(default=0.001, ge=0.0, description="L1 regularization strength")

    # Image preprocessing
    image_size: Tuple[int, int] = Field(default=(400, 400), description="Target image size")
    expand_pixels: int = Field(default=50, ge=0, description="Pixels to pad around crop")
    expand_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Bbox expand rate")

    # HOG feature parameters
    orientations: int = Field(default=4, ge=1, description="Number of orientation bins")
    pixels_per_cell: int = Field(default=3, ge=1, description="Pixels per cell")
    cells_per_block: int = Field(default=2, ge=1, description="Cells per block")
    cells_per_side: int = Field(default=1, ge=1, description="Cells per side of landmark")
    hog_no_block: bool = Field(default=True, description="Skip block normalization")

    # Runtime
    mode: Literal["train", "test"] = Field(default="train", description="Training or testing mode")
    verbose: bool = Field(default=True, description="Verbose output")
    device: str = Field(default="cpu", description="Device for computation")

    # Landmark parameters
    n_landmarks: int = Field(default=68, description="Number of facial landmarks")

    model_config = {
        "frozen": False,
        "validate_assignment": True,
    }

    def display(self) -> None:
        """Display configuration in readable format."""
        print("=" * 60)
        print("SDM Configuration")
        print("=" * 60)
        print(f"Mode: {self.mode}")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"L1 regularization (alpha): {self.alpha}")
        print(f"Target image size: {self.image_size}")
        print(f"Bounding box expansion: {self.expand_rate * 100:.1f}%")
        print(f"Padding pixels: {self.expand_pixels}")
        print("-" * 60)
        print("HOG Feature Configuration:")
        print(f"  Orientations: {self.orientations}")
        print(f"  Pixels per cell: {self.pixels_per_cell}")
        print(f"  Cells per block: {self.cells_per_block}")
        print(f"  Cells per side: {self.cells_per_side}")
        print(f"  Block normalization: {'No' if self.hog_no_block else 'Yes'}")
        print("=" * 60)

    def get_hog_feature_dim(self) -> int:
        """Calculate HOG feature dimension based on parameters."""
        cells_total = self.cells_per_side * 2  # Both sides

        if self.hog_no_block:
            # Without block normalization: n_landmarks × cells × cells × orientations
            return self.n_landmarks * cells_total * cells_total * self.orientations
        else:
            # With block normalization
            n_blocks = cells_total - self.cells_per_block + 1
            return (
                self.n_landmarks
                * n_blocks
                * n_blocks
                * self.cells_per_block
                * self.cells_per_block
                * self.orientations
            )
