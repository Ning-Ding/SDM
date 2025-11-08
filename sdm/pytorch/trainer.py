"""PyTorch training utilities for landmark detection."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sdm.pytorch.model import WingLoss


class PyTorchTrainer:
    """Trainer for PyTorch landmark detection models.

    Provides a unified interface for training, validation, and evaluation
    of deep learning models for face alignment.

    Args:
        model: PyTorch model
        device: Device to train on ('cuda' or 'cpu')
        loss_fn: Loss function (default: MSELoss)
        optimizer: Optimizer (default: Adam with lr=0.001)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        self.model = model.to(device)
        self.device = device

        # Default loss and optimizer
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.optimizer = (
            optimizer
            if optimizer is not None
            else optim.Adam(model.parameters(), lr=0.001)
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader: DataLoader, verbose: bool = True) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            verbose: Whether to show progress bar

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        iterator = train_loader
        if verbose:
            iterator = tqdm(train_loader, desc="Training")

        for images, landmarks in iterator:
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.loss_fn(predictions, landmarks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if verbose:
                iterator.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / n_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, verbose: bool = True) -> float:
        """Validate model.

        Args:
            val_loader: Validation data loader
            verbose: Whether to show progress bar

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        iterator = val_loader
        if verbose:
            iterator = tqdm(val_loader, desc="Validation")

        for images, landmarks in iterator:
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            predictions = self.model(images)
            loss = self.loss_fn(predictions, landmarks)

            total_loss += loss.item()
            n_batches += 1

            if verbose:
                iterator.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / n_batches
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        early_stopping_patience: int = 10,
        save_dir: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> dict:
        """Train model with validation and early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            early_stopping_patience: Epochs to wait before early stopping
            save_dir: Directory to save checkpoints
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        patience_counter = 0

        for epoch in range(n_epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{n_epochs}")
                print("=" * 60)

            # Train
            train_loss = self.train_epoch(train_loader, verbose=verbose)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader, verbose=verbose)
            self.val_losses.append(val_loss)

            if verbose:
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                if save_dir:
                    checkpoint_path = save_dir / "best_model.pth"
                    self.save_checkpoint(checkpoint_path)
                    if verbose:
                        print(f"✓ Saved best model (val_loss: {val_loss:.6f})")

            else:
                patience_counter += 1
                if verbose:
                    print(
                        f"No improvement ({patience_counter}/{early_stopping_patience})"
                    )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "n_epochs": len(self.train_losses),
        }

        return history

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict landmarks for a single image.

        Args:
            image: Input image tensor (1, H, W) or (B, 1, H, W)

        Returns:
            Predicted landmarks (B, n_landmarks * 2)
        """
        self.model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        predictions = self.model(image)

        return predictions.cpu()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))


def create_trainer(
    model_type: str = "cnn",
    n_landmarks: int = 68,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    loss_type: str = "mse",
    learning_rate: float = 0.001,
) -> PyTorchTrainer:
    """Factory function to create trainer with specified configuration.

    Args:
        model_type: 'cnn' or 'resnet'
        n_landmarks: Number of landmarks
        device: Device to train on
        loss_type: 'mse' or 'wing'
        learning_rate: Learning rate

    Returns:
        Configured PyTorchTrainer
    """
    from sdm.pytorch.model import LandmarkCNN, LandmarkResNet

    # Create model
    if model_type == "cnn":
        model = LandmarkCNN(n_landmarks=n_landmarks)
    elif model_type == "resnet":
        model = LandmarkResNet(n_landmarks=n_landmarks)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create loss function
    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type == "wing":
        loss_fn = WingLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create trainer
    trainer = PyTorchTrainer(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    return trainer
