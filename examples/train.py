"""Example: Train SDM model on LFPW dataset.

This script demonstrates how to train the SDM model from scratch.

Usage:
    python examples/train.py --data-root data/ --output models/sdm_model.mat
"""

import argparse
from pathlib import Path

from sdm import SDM, SDMConfig
from sdm.data.dataset import LFPWDataset


def main():
    parser = argparse.ArgumentParser(description="Train SDM model")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/sdm_model.mat",
        help="Output model path",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of SDM iterations",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="L1 regularization strength (0 = no regularization)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=400,
        help="Target image size",
    )

    args = parser.parse_args()

    # Create configuration
    config = SDMConfig(
        n_iterations=args.iterations,
        alpha=args.alpha,
        image_size=(args.image_size, args.image_size),
        mode="train",
        verbose=True,
    )

    print("=" * 80)
    print("SDM Face Alignment Training")
    print("=" * 80)

    # Load training dataset
    print(f"\nLoading dataset from: {args.data_root}")
    train_dataset = LFPWDataset(
        data_root=args.data_root,
        split="train",
        config=config,
    )
    print(f"Training samples: {len(train_dataset)}")

    # Initialize and train model
    print("\nInitializing SDM model...")
    model = SDM(config)

    print("\nStarting training...")
    model.train(train_dataset)

    # Save trained model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print(f"\n{'=' * 80}")
    print(f"✓ Training completed!")
    print(f"✓ Model saved to: {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
