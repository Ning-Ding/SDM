"""Tests for SDM core functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sdm.core.model import SDMConfig
from sdm.core.sdm import SDM


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, n_samples: int = 10, n_landmarks: int = 68, image_size: int = 400):
        self.n_samples = n_samples
        self.n_landmarks = n_landmarks
        self.image_size = image_size

        # Generate synthetic data
        self.images = [
            np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
            for _ in range(n_samples)
        ]

        self.landmarks = [
            np.random.randint(50, image_size - 50, (n_landmarks, 2), dtype=np.int32)
            for _ in range(n_samples)
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.images[idx], self.landmarks[idx], self.landmarks[idx]


def test_sdm_train_predict():
    """Test basic SDM training and prediction."""
    config = SDMConfig(
        n_iterations=2,
        alpha=0.0,  # No regularization for faster test
        verbose=False,
    )

    # Create small mock dataset
    dataset = MockDataset(n_samples=5)

    # Train
    model = SDM(config)
    model.train(dataset)

    # Check that model was trained
    assert len(model.regressors) == 2
    assert len(model.biases) == 2
    assert model.initial_shape is not None

    # Predict
    image, landmarks_true, _ = dataset[0]
    landmarks_pred, _ = model.predict(image)

    assert landmarks_pred.shape == landmarks_true.shape


def test_sdm_save_load():
    """Test SDM model save and load."""
    config = SDMConfig(n_iterations=2, alpha=0.0, verbose=False)
    dataset = MockDataset(n_samples=5)

    # Train and save
    model = SDM(config)
    model.train(dataset)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.mat"
        model.save(model_path)

        # Load
        loaded_model = SDM(config)
        loaded_model.load(model_path)

        # Check loaded model
        assert len(loaded_model.regressors) == 2
        assert len(loaded_model.biases) == 2
        assert loaded_model.initial_shape is not None

        # Compare predictions
        image, _, _ = dataset[0]
        pred1, _ = model.predict(image)
        pred2, _ = loaded_model.predict(image)

        np.testing.assert_array_almost_equal(pred1, pred2)


def test_sdm_config_mismatch_bug_fix():
    """Test that evaluation handles config.n_iterations mismatch with loaded model.

    This test verifies the fix for the bug where:
    - Model is trained with n_iterations=5
    - Model is saved and loaded with a config where n_iterations=3 (default)
    - evaluate() would crash with IndexError

    The fix: mse_per_iteration should be sized by len(self.regressors),
    not self.config.n_iterations.
    """
    # Train model with 5 iterations
    train_config = SDMConfig(n_iterations=5, alpha=0.0, verbose=False)
    dataset = MockDataset(n_samples=5)

    model = SDM(train_config)
    model.train(dataset)

    assert len(model.regressors) == 5, "Model should have 5 regressors"

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.mat"
        model.save(model_path)

        # Load with different config (n_iterations=3, the default)
        load_config = SDMConfig(verbose=False)  # Default n_iterations=3
        assert load_config.n_iterations == 3

        loaded_model = SDM(load_config)
        loaded_model.load(model_path)

        # Model should still have 5 regressors (from disk)
        assert len(loaded_model.regressors) == 5
        # But config says 3
        assert loaded_model.config.n_iterations == 3

        # This should NOT crash with IndexError
        # Before the fix, mse_per_iteration had length 3 but loop ran 5 times
        try:
            results = loaded_model.evaluate(dataset, metric="mse")

            # Verify results structure
            assert "mse_per_iteration" in results
            # Should have 5 entries (one per regressor), not 3
            assert len(results["mse_per_iteration"]) == 5

            print("✅ Bug fix verified: evaluate() handles config mismatch")

        except IndexError as e:
            pytest.fail(f"IndexError raised (bug not fixed): {e}")


def test_sdm_evaluate():
    """Test SDM evaluation."""
    config = SDMConfig(n_iterations=2, alpha=0.0, verbose=False)
    dataset = MockDataset(n_samples=5)

    model = SDM(config)
    model.train(dataset)

    # Evaluate
    results = model.evaluate(dataset, metric="mse")

    # Check results structure
    assert "mean_error" in results
    assert "std_error" in results
    assert "median_error" in results
    assert "mse_per_iteration" in results
    assert "all_errors" in results

    # Check mse_per_iteration has correct length
    assert len(results["mse_per_iteration"]) == config.n_iterations


def test_sdm_initial_shape():
    """Test initial shape computation."""
    config = SDMConfig(n_iterations=1, verbose=False)
    dataset = MockDataset(n_samples=10)

    model = SDM(config)

    # Extract landmarks
    landmarks = [dataset[i][1] for i in range(len(dataset))]
    landmarks_flat = [lm.ravel() for lm in landmarks]
    landmarks_array = np.array(landmarks_flat, dtype=np.float32)

    # Compute initial shape
    initial = model._compute_initial_shape(landmarks_array)

    # Check shape
    assert initial.shape == (68, 2)

    # Should be close to mean
    mean_landmarks = np.mean(landmarks_array, axis=0).reshape(-1, 2)
    np.testing.assert_array_almost_equal(initial, mean_landmarks, decimal=0)
