"""Example: Train PyTorch CNN model for face alignment.

This script demonstrates training a deep learning model as an
alternative to classical SDM.

Usage:
    python examples/train_pytorch.py --data-root data/ --output models/pytorch_model.pth
"""

import argparse
from pathlib import Path

import torch

from sdm import SDMConfig
from sdm.data.dataset import LFPWDataset
from sdm.pytorch.dataset import create_dataloaders
from sdm.pytorch.trainer import create_trainer


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch landmark detection model")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/pytorch",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "resnet"],
        default="cnn",
        help="Model architecture",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["mse", "wing"],
        default="mse",
        help="Loss function",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PyTorch Face Alignment Training")
    print("=" * 80)
    print(f"Model: {args.model_type}")
    print(f"Loss: {args.loss_type}")
    print(f"Device: {args.device}")

    # Create configuration
    config = SDMConfig(mode="train")

    # Load datasets
    print(f"\nLoading datasets from: {args.data_root}")
    train_dataset = LFPWDataset(data_root=args.data_root, split="train", config=config)
    test_dataset = LFPWDataset(data_root=args.data_root, split="test", config=config)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Create trainer
    print(f"\nInitializing {args.model_type.upper()} model...")
    trainer = create_trainer(
        model_type=args.model_type,
        n_landmarks=68,
        device=args.device,
        loss_type=args.loss_type,
        learning_rate=args.lr,
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        n_epochs=args.epochs,
        early_stopping_patience=10,
        save_dir=args.output_dir,
        verbose=True,
    )

    # Print summary
    print(f"\n{'=' * 80}")
    print("Training Summary:")
    print(f"{'=' * 80}")
    print(f"Total epochs: {history['n_epochs']}")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Final training loss: {history['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.6f}")
    print(f"{'=' * 80}")

    print(f"\n✓ Model saved to: {Path(args.output_dir) / 'best_model.pth'}")


if __name__ == "__main__":
    main()
