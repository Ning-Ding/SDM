"""Command-line interface for SDM face alignment.

This module provides CLI commands for training, inference, and demo.
"""

import sys
from pathlib import Path

import click
import numpy as np

from sdm import SDM, SDMConfig
from sdm.data.dataset import LFPWDataset
from sdm.utils.image import load_image
from sdm.utils.visualization import draw_landmarks


@click.group()
def cli():
    """SDM Face Alignment CLI."""
    pass


@cli.command()
@click.option("--data-root", type=click.Path(exists=True), required=True, help="Path to dataset")
@click.option("--output", type=click.Path(), default="models/sdm_model.mat", help="Output model path")
@click.option("--iterations", type=int, default=3, help="Number of SDM iterations")
@click.option("--alpha", type=float, default=0.001, help="L1 regularization strength")
def train_cli(data_root: str, output: str, iterations: int, alpha: float):
    """Train SDM model on dataset."""
    click.echo("=" * 60)
    click.echo("SDM Face Alignment - Training")
    click.echo("=" * 60)

    # Create config
    config = SDMConfig(n_iterations=iterations, alpha=alpha)

    # Load dataset
    click.echo(f"\nLoading dataset from: {data_root}")
    dataset = LFPWDataset(data_root, split="train", config=config)
    click.echo(f"Dataset size: {len(dataset)} images")

    # Create and train model
    model = SDM(config)
    click.echo("\nStarting training...")
    model.train(dataset)

    # Save model
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    click.echo(f"\n✓ Model saved to: {output_path}")


@cli.command()
@click.option("--model", type=click.Path(exists=True), required=True, help="Path to trained model")
@click.option("--data-root", type=click.Path(exists=True), required=True, help="Path to dataset")
@click.option("--output-dir", type=click.Path(), default="results", help="Output directory")
def infer_cli(model: str, data_root: str, output_dir: str):
    """Run inference on test dataset."""
    click.echo("=" * 60)
    click.echo("SDM Face Alignment - Inference")
    click.echo("=" * 60)

    # Load model
    config = SDMConfig(mode="test")
    sdm_model = SDM(config)
    sdm_model.load(model)
    click.echo(f"Model loaded from: {model}")

    # Load test dataset
    dataset = LFPWDataset(data_root, split="test", config=config)
    click.echo(f"Test dataset size: {len(dataset)} images")

    # Evaluate
    click.echo("\nRunning evaluation...")
    results = sdm_model.evaluate(dataset)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write("SDM Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean MSE: {results['mean_error']:.4f}\n")
        f.write(f"Std MSE: {results['std_error']:.4f}\n")
        f.write(f"Median MSE: {results['median_error']:.4f}\n")

    click.echo(f"\n✓ Results saved to: {results_file}")


@cli.command()
@click.option("--model", type=click.Path(exists=True), required=True, help="Path to trained model")
@click.option("--image", type=click.Path(exists=True), required=True, help="Path to input image")
@click.option("--output", type=click.Path(), default="output.png", help="Output image path")
def demo_cli(model: str, image: str, output: str):
    """Demo: detect landmarks on a single image."""
    click.echo("=" * 60)
    click.echo("SDM Face Alignment - Demo")
    click.echo("=" * 60)

    click.echo("Note: This demo requires preprocessed images.")
    click.echo("For full pipeline, use the Streamlit web interface.")

    # Load model
    config = SDMConfig(mode="test")
    sdm_model = SDM(config)
    sdm_model.load(model)

    # Load image
    img = load_image(image, grayscale=True)

    # Predict
    landmarks, _ = sdm_model.predict(img)

    # Visualize
    result_img = draw_landmarks(img, landmarks)

    # Save
    import cv2
    cv2.imwrite(output, result_img)

    click.echo(f"\n✓ Result saved to: {output}")


if __name__ == "__main__":
    cli()
