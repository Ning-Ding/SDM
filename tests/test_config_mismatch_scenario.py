"""Standalone test to demonstrate the config mismatch bug fix.

This script demonstrates the bug and verifies the fix:

BUG SCENARIO:
1. Train a model with n_iterations=5
2. Save the model
3. Create a new config with n_iterations=3 (default)
4. Load the saved model
5. Call evaluate()

BEFORE FIX:
  - mse_per_iteration = [[] for _ in range(self.config.n_iterations)]  # length 3
  - for i in range(len(self.regressors)):  # loops 5 times
  - mse_per_iteration[i].append(...)  # IndexError when i=3!

AFTER FIX:
  - mse_per_iteration = [[] for _ in range(len(self.regressors))]  # length 5
  - Works correctly regardless of config mismatch

Run this script to verify the fix:
    python tests/test_config_mismatch_scenario.py
"""

import tempfile
from pathlib import Path

import numpy as np


def create_mock_data(n_samples=5, n_landmarks=68, image_size=400):
    """Create mock dataset."""
    images = [
        np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
        for _ in range(n_samples)
    ]
    landmarks = [
        np.random.randint(50, image_size - 50, (n_landmarks, 2), dtype=np.int32)
        for _ in range(n_samples)
    ]
    return images, landmarks


class MockDataset:
    """Simple mock dataset."""

    def __init__(self, images, landmarks):
        self.images = images
        self.landmarks = landmarks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.landmarks[idx], self.landmarks[idx]


def main():
    """Run the bug demonstration."""
    try:
        from sdm import SDM, SDMConfig
    except ImportError:
        print("❌ Error: SDM not installed")
        print("Run: uv sync")
        return

    print("=" * 70)
    print("SDM Config Mismatch Bug Fix Verification")
    print("=" * 70)

    # Step 1: Train model with 5 iterations
    print("\n📝 Step 1: Training model with n_iterations=5")
    train_config = SDMConfig(n_iterations=5, alpha=0.0, verbose=False)
    print(f"   Training config: n_iterations={train_config.n_iterations}")

    images, landmarks = create_mock_data(n_samples=5)
    train_dataset = MockDataset(images, landmarks)

    model = SDM(train_config)
    model.train(train_dataset)

    print(f"   ✓ Model trained with {len(model.regressors)} regressors")

    # Step 2: Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.mat"
        model.save(model_path)
        print(f"   ✓ Model saved to temporary file")

        # Step 3: Load with different config
        print("\n📝 Step 2: Loading model with default config (n_iterations=3)")
        load_config = SDMConfig(verbose=False)  # Default n_iterations=3
        print(f"   Loading config: n_iterations={load_config.n_iterations}")

        loaded_model = SDM(load_config)
        loaded_model.load(model_path)

        print(
            f"   ✓ Loaded model has {len(loaded_model.regressors)} regressors (from disk)"
        )
        print(f"   ✓ Current config has n_iterations={loaded_model.config.n_iterations}")
        print(f"   ⚠️  MISMATCH: {len(loaded_model.regressors)} != {loaded_model.config.n_iterations}")

        # Step 4: Evaluate (this is where the bug would occur)
        print("\n📝 Step 3: Evaluating model...")
        print("   Without fix: IndexError when mse_per_iteration[3] is accessed")
        print("   With fix: Works correctly\n")

        try:
            results = loaded_model.evaluate(train_dataset, metric="mse")

            print("   ✅ SUCCESS! Evaluation completed without IndexError")
            print(f"   ✓ Mean MSE: {results['mean_error']:.4f}")
            print(
                f"   ✓ MSE per iteration: {len(results['mse_per_iteration'])} values"
            )
            print(f"   ✓ Expected: {len(loaded_model.regressors)} values")

            if len(results["mse_per_iteration"]) == len(loaded_model.regressors):
                print("\n   ✅ VERIFIED: mse_per_iteration sized from regressors, not config")
            else:
                print("\n   ❌ ERROR: mse_per_iteration has wrong length")

        except IndexError as e:
            print(f"   ❌ FAILED: IndexError occurred: {e}")
            print("   This indicates the bug is NOT fixed")
            print("\n   Expected behavior:")
            print("     mse_per_iteration = [[] for _ in range(len(self.regressors))]")
            print("   NOT:")
            print("     mse_per_iteration = [[] for _ in range(self.config.n_iterations)]")
            return

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("The fix ensures that mse_per_iteration is sized based on the")
    print("actual number of regressors in the model, not the config value.")
    print("This allows models trained with different n_iterations to be")
    print("evaluated correctly after loading.")
    print("\nImplementation:")
    print("  mse_per_iteration = [[] for _ in range(len(self.regressors))]")
    print("=" * 70)


if __name__ == "__main__":
    main()
