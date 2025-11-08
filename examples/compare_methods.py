"""Example: Compare SDM vs PyTorch methods.

This script compares the performance of classical SDM and deep learning approaches.

Usage:
    python examples/compare_methods.py --data-root data/ --sdm-model models/sdm_model.mat --pytorch-model models/pytorch/best_model.pth
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sdm import SDM, SDMConfig
from sdm.data.dataset import LFPWDataset
from sdm.pytorch.model import LandmarkCNN
from sdm.pytorch.trainer import PyTorchTrainer


def evaluate_sdm(model: SDM, dataset: LFPWDataset) -> dict:
    """Evaluate SDM model."""
    errors = []
    times = []

    for idx in tqdm(range(len(dataset)), desc="SDM Evaluation"):
        image, landmarks_true, _ = dataset[idx]

        start_time = time.time()
        landmarks_pred, _ = model.predict(image)
        inference_time = time.time() - start_time

        mse = np.mean((landmarks_pred - landmarks_true) ** 2)
        errors.append(mse)
        times.append(inference_time)

    return {
        "mean_mse": np.mean(errors),
        "std_mse": np.std(errors),
        "mean_time": np.mean(times),
        "total_time": np.sum(times),
    }


def evaluate_pytorch(trainer: PyTorchTrainer, dataset: LFPWDataset) -> dict:
    """Evaluate PyTorch model."""
    from sdm.pytorch.dataset import PyTorchFaceDataset

    pytorch_dataset = PyTorchFaceDataset(dataset)
    errors = []
    times = []

    for idx in tqdm(range(len(pytorch_dataset)), desc="PyTorch Evaluation"):
        image_tensor, landmarks_true = pytorch_dataset[idx]

        start_time = time.time()
        landmarks_pred = trainer.predict(image_tensor)
        inference_time = time.time() - start_time

        # Convert to numpy
        landmarks_pred = landmarks_pred.numpy().reshape(-1, 2)
        landmarks_true = landmarks_true.numpy().reshape(-1, 2)

        mse = np.mean((landmarks_pred - landmarks_true) ** 2)
        errors.append(mse)
        times.append(inference_time)

    return {
        "mean_mse": np.mean(errors),
        "std_mse": np.std(errors),
        "mean_time": np.mean(times),
        "total_time": np.sum(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare SDM and PyTorch methods")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to dataset root",
    )
    parser.add_argument(
        "--sdm-model",
        type=str,
        required=True,
        help="Path to SDM model (.mat)",
    )
    parser.add_argument(
        "--pytorch-model",
        type=str,
        required=True,
        help="Path to PyTorch model (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results.txt",
        help="Output file for comparison results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Method Comparison: SDM vs PyTorch CNN")
    print("=" * 80)

    # Load test dataset
    config = SDMConfig(mode="test", verbose=False)
    test_dataset = LFPWDataset(data_root=args.data_root, split="test", config=config)
    print(f"\nTest samples: {len(test_dataset)}")

    # Load SDM model
    print(f"\nLoading SDM model from: {args.sdm_model}")
    sdm_model = SDM(config)
    sdm_model.load(args.sdm_model)

    # Load PyTorch model
    print(f"Loading PyTorch model from: {args.pytorch_model}")
    pytorch_model = LandmarkCNN(n_landmarks=68)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pytorch_trainer = PyTorchTrainer(pytorch_model, device=device)
    pytorch_trainer.load_checkpoint(args.pytorch_model)

    # Evaluate both methods
    print("\n" + "=" * 80)
    print("Evaluating SDM...")
    sdm_results = evaluate_sdm(sdm_model, test_dataset)

    print("\nEvaluating PyTorch...")
    pytorch_results = evaluate_pytorch(pytorch_trainer, test_dataset)

    # Print comparison
    print("\n" + "=" * 80)
    print("Comparison Results:")
    print("=" * 80)
    print(f"\n{'Method':<20} {'Mean MSE':<15} {'Std MSE':<15} {'Avg Time (s)':<15}")
    print("-" * 80)
    print(
        f"{'SDM':<20} {sdm_results['mean_mse']:<15.4f} {sdm_results['std_mse']:<15.4f} {sdm_results['mean_time']:<15.6f}"
    )
    print(
        f"{'PyTorch CNN':<20} {pytorch_results['mean_mse']:<15.4f} {pytorch_results['std_mse']:<15.4f} {pytorch_results['mean_time']:<15.6f}"
    )
    print("=" * 80)

    # Calculate relative performance
    mse_improvement = (
        (sdm_results["mean_mse"] - pytorch_results["mean_mse"])
        / sdm_results["mean_mse"]
        * 100
    )
    speed_ratio = sdm_results["mean_time"] / pytorch_results["mean_time"]

    print(f"\nPyTorch MSE improvement over SDM: {mse_improvement:+.2f}%")
    print(f"Speed ratio (SDM time / PyTorch time): {speed_ratio:.2f}x")

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write("Method Comparison: SDM vs PyTorch CNN\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(f"{'Method':<20} {'Mean MSE':<15} {'Std MSE':<15} {'Avg Time (s)':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'SDM':<20} {sdm_results['mean_mse']:<15.4f} {sdm_results['std_mse']:<15.4f} {sdm_results['mean_time']:<15.6f}\n"
        )
        f.write(
            f"{'PyTorch CNN':<20} {pytorch_results['mean_mse']:<15.4f} {pytorch_results['std_mse']:<15.4f} {pytorch_results['mean_time']:<15.6f}\n"
        )
        f.write("\n")
        f.write(f"PyTorch MSE improvement: {mse_improvement:+.2f}%\n")
        f.write(f"Speed ratio (SDM / PyTorch): {speed_ratio:.2f}x\n")

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
