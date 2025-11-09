"""Example: Run inference with trained SDM model.

This script demonstrates how to use a trained SDM model for prediction.

Usage:
    python examples/inference.py --model models/sdm_model.mat --data-root data/ --output-dir results/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sdm import SDM, SDMConfig
from sdm.data.dataset import LFPWDataset
from sdm.utils.visualization import draw_landmarks


def main():
    parser = argparse.ArgumentParser(description="SDM inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.mat file)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (None = all)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save visualization images",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SDM Face Alignment Inference")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {args.model}")
    config = SDMConfig(mode="test", verbose=True)
    model = SDM(config)
    model.load(args.model)

    # Load test dataset
    print(f"\nLoading test dataset from: {args.data_root}")
    test_dataset = LFPWDataset(
        data_root=args.data_root,
        split="test",
        config=config,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_images:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

    # Run inference
    num_samples = args.num_samples if args.num_samples else len(test_dataset)
    errors = []

    print(f"\nRunning inference on {num_samples} samples...")

    for idx in tqdm(range(num_samples)):
        # Get image and ground truth
        image, landmarks_true, _ = test_dataset[idx]

        # Predict
        landmarks_pred, _ = model.predict(image)

        # Compute error
        mse = np.mean((landmarks_pred - landmarks_true) ** 2)
        errors.append(mse)

        # Save visualization
        if args.save_images:
            vis_image = draw_landmarks(image, landmarks_pred, color=(255, 0, 0))
            vis_image = draw_landmarks(vis_image, landmarks_true, color=(0, 255, 0))

            output_path = viz_dir / f"result_{idx:04d}.png"
            cv2.imwrite(str(output_path), vis_image)

    # Compute statistics
    errors = np.array(errors)
    mean_mse = errors.mean()
    std_mse = errors.std()
    median_mse = np.median(errors)

    # Print results
    print(f"\n{'=' * 80}")
    print("Evaluation Results:")
    print(f"{'=' * 80}")
    print(f"Mean MSE: {mean_mse:.4f}")
    print(f"Std MSE: {std_mse:.4f}")
    print(f"Median MSE: {median_mse:.4f}")
    print(f"{'=' * 80}")

    # Save results to file
    results_file = output_dir / "inference_results.txt"
    with open(results_file, "w") as f:
        f.write("SDM Inference Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.data_root}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write("\n")
        f.write(f"Mean MSE: {mean_mse:.4f}\n")
        f.write(f"Std MSE: {std_mse:.4f}\n")
        f.write(f"Median MSE: {median_mse:.4f}\n")

    print(f"\n✓ Results saved to: {results_file}")
    if args.save_images:
        print(f"✓ Visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
