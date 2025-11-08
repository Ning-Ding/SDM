"""Supervised Descent Method for Face Alignment.

This module implements the core SDM algorithm for facial landmark detection.

Reference:
    Xiong, X., & De la Torre, F. (2013). Supervised descent method and its
    applications to face alignment. CVPR.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import io
from sklearn.linear_model import Lasso, LinearRegression
from tqdm import tqdm

from sdm.core.model import SDMConfig
from sdm.data.dataset import FaceAlignmentDataset
from sdm.features.hog import HOGExtractor


class SDM:
    """Supervised Descent Method for face alignment.

    SDM learns a sequence of descent directions through linear regression,
    iteratively refining landmark predictions.

    The algorithm works as follows:
    1. Initialize landmarks with mean shape
    2. For each iteration:
        a. Extract features (HOG) at current landmark positions
        b. Learn regression: Δx = R·φ(I, x) + b
        c. Update landmarks: x_new = x + Δx
    3. At test time, apply learned regressors sequentially

    Attributes:
        config: SDM configuration
        regressors: List of learned regression matrices (R)
        biases: List of learned bias vectors (b)
        initial_shape: Mean landmark shape for initialization
    """

    def __init__(self, config: Optional[SDMConfig] = None):
        """Initialize SDM.

        Args:
            config: SDM configuration (uses default if None)
        """
        self.config = config if config is not None else SDMConfig()
        self.hog_extractor = HOGExtractor(self.config)

        # Learned parameters
        self.regressors: list[NDArray[np.float32]] = []
        self.biases: list[NDArray[np.float32]] = []
        self.initial_shape: Optional[NDArray[np.int32]] = None

        # Training history
        self.training_history: dict = {}

    def train(self, dataset: FaceAlignmentDataset) -> None:
        """Train SDM on dataset.

        Args:
            dataset: Face alignment dataset
        """
        if self.config.verbose:
            self.config.display()
            print(f"\nTraining on {len(dataset)} samples...")

        # Step 1: Precompute features and ground truth landmarks
        hog_features_true, landmarks_true, gray_images = self._precompute_data(dataset)

        # Step 2: Compute initial shape (mean of all landmarks)
        self.initial_shape = self._compute_initial_shape(landmarks_true)

        # Step 3: Initialize current landmarks with mean shape
        landmarks_current = np.tile(self.initial_shape.ravel(), (len(dataset), 1))

        # Step 4: Iterative training
        self.regressors = []
        self.biases = []
        sparse_rates = []

        for iteration in range(self.config.n_iterations):
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.config.n_iterations}")
                print(f"{'='*60}")

            # Compute landmark delta (supervision signal)
            landmarks_delta = landmarks_true - landmarks_current

            # Extract HOG features at current landmark positions
            hog_features_current = self._extract_features_batch(
                gray_images, landmarks_current
            )

            # Learn regressor: Δx = R·φ + b
            regressor, bias = self._train_regressor(hog_features_current, landmarks_delta)

            self.regressors.append(regressor)
            self.biases.append(bias)

            # Compute sparse rate (for Lasso)
            if self.config.alpha > 0:
                sparse_rate = (regressor == 0).sum() / regressor.size
                sparse_rates.append(sparse_rate)
                if self.config.verbose:
                    print(f"Sparse rate: {sparse_rate:.2%}")

            # Update current landmarks
            landmarks_current = (
                landmarks_current + hog_features_current @ regressor + bias
            )

            # Compute training error
            mse = np.mean((landmarks_current - landmarks_true) ** 2)
            if self.config.verbose:
                print(f"Training MSE: {mse:.4f}")

        # Store training history
        self.training_history = {
            "n_iterations": self.config.n_iterations,
            "n_samples": len(dataset),
            "sparse_rates": sparse_rates,
        }

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Training completed!")
            print(f"{'='*60}")

    def predict(
        self,
        image: NDArray[np.uint8],
        initial_landmarks: Optional[NDArray[np.int32]] = None,
    ) -> Tuple[NDArray[np.int32], list[float]]:
        """Predict landmarks for a single image.

        Args:
            image: Grayscale image
            initial_landmarks: Initial landmark positions (uses mean shape if None)

        Returns:
            Tuple of (predicted_landmarks, mse_per_iteration)

        Raises:
            RuntimeError: If model hasn't been trained
        """
        if self.initial_shape is None:
            raise RuntimeError("Model must be trained before prediction")

        # Initialize landmarks
        if initial_landmarks is None:
            landmarks = self.initial_shape.copy()
        else:
            landmarks = initial_landmarks.copy()

        mse_history = []

        # Apply learned regressors iteratively
        for iteration in range(len(self.regressors)):
            # Extract features at current landmarks
            features = self.hog_extractor.extract(image, landmarks)

            # Apply regressor
            delta = features @ self.regressors[iteration] + self.biases[iteration]

            # Update landmarks
            landmarks = (landmarks.ravel() + delta).reshape(-1, 2)

            # Record MSE (if we have ground truth, this would be computed externally)
            mse_history.append(0.0)  # Placeholder

        return landmarks.astype(np.int32), mse_history

    def evaluate(
        self,
        dataset: FaceAlignmentDataset,
        metric: str = "mse",
    ) -> dict:
        """Evaluate model on dataset.

        Args:
            dataset: Test dataset
            metric: Evaluation metric ('mse' or 'nme')

        Returns:
            Dictionary containing evaluation results
        """
        if self.config.verbose:
            print(f"\nEvaluating on {len(dataset)} samples...")

        errors = []
        mse_per_iteration = [[] for _ in range(self.config.n_iterations)]

        for idx in tqdm(range(len(dataset)), disable=not self.config.verbose):
            image, landmarks_true, _ = dataset[idx]

            # Predict
            landmarks_pred, mse_history = self.predict(image)

            # Compute error
            if metric == "mse":
                error = np.mean((landmarks_pred - landmarks_true) ** 2)
            elif metric == "nme":
                # Normalized Mean Error (normalized by inter-ocular distance)
                error = self._compute_nme(landmarks_pred, landmarks_true)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            errors.append(error)

            # Track MSE per iteration (recompute for accuracy)
            landmarks = self.initial_shape.copy()
            for i in range(len(self.regressors)):
                features = self.hog_extractor.extract(image, landmarks)
                delta = features @ self.regressors[i] + self.biases[i]
                landmarks = (landmarks.ravel() + delta).reshape(-1, 2)
                iter_mse = np.mean((landmarks - landmarks_true) ** 2)
                mse_per_iteration[i].append(iter_mse)

        results = {
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "median_error": np.median(errors),
            "mse_per_iteration": [np.mean(mse) for mse in mse_per_iteration],
            "all_errors": errors,
        }

        if self.config.verbose:
            print(f"\nEvaluation Results:")
            print(f"  Mean {metric.upper()}: {results['mean_error']:.4f}")
            print(f"  Std {metric.upper()}: {results['std_error']:.4f}")
            print(f"  Median {metric.upper()}: {results['median_error']:.4f}")

        return results

    def save(self, path: str | Path) -> None:
        """Save trained model.

        Args:
            path: Path to save model (.mat format for compatibility)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        io.savemat(
            str(path),
            {
                "R": np.array(self.regressors),
                "B": np.array(self.biases),
                "I": self.initial_shape,
            },
        )

        if self.config.verbose:
            print(f"Model saved to: {path}")

    def load(self, path: str | Path) -> None:
        """Load trained model.

        Args:
            path: Path to model file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = io.loadmat(str(path))
        self.regressors = [data["R"][i] for i in range(len(data["R"]))]
        self.biases = [data["B"][i] for i in range(len(data["B"]))]
        self.initial_shape = data["I"]

        if self.config.verbose:
            print(f"Model loaded from: {path}")
            print(f"  Number of iterations: {len(self.regressors)}")
            print(f"  Initial shape: {self.initial_shape.shape}")

    def _precompute_data(
        self, dataset: FaceAlignmentDataset
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], list]:
        """Precompute HOG features and landmarks for all training data."""
        if self.config.verbose:
            print("\nPrecomputing features...")

        hog_list = []
        landmark_list = []
        image_list = []

        for idx in tqdm(range(len(dataset)), disable=not self.config.verbose):
            image, landmarks, _ = dataset[idx]

            # Extract HOG features at true landmark positions
            hog_features = self.hog_extractor.extract(image, landmarks)

            hog_list.append(hog_features)
            landmark_list.append(landmarks.ravel())
            image_list.append(image)

        hog_array = np.array(hog_list, dtype=np.float32)
        landmark_array = np.array(landmark_list, dtype=np.float32)

        return hog_array, landmark_array, image_list

    def _compute_initial_shape(
        self, landmarks: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        """Compute mean shape for initialization."""
        mean_shape = np.mean(landmarks, axis=0)
        return mean_shape.reshape(-1, 2).astype(np.int32)

    def _extract_features_batch(
        self, images: list, landmarks: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Extract HOG features for batch of images."""
        features = []

        iterator = enumerate(images)
        if self.config.verbose:
            iterator = tqdm(iterator, total=len(images), desc="Extracting features")

        for idx, image in iterator:
            landmark_2d = landmarks[idx].reshape(-1, 2).astype(np.int32)
            feat = self.hog_extractor.extract(image, landmark_2d)
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def _train_regressor(
        self, features: NDArray[np.float32], targets: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Train linear regressor for one iteration."""
        if self.config.verbose:
            print("Training regressor...")

        if self.config.alpha == 0:
            # Standard linear regression
            reg = LinearRegression(fit_intercept=True)
        else:
            # Lasso regression (L1 regularization)
            reg = Lasso(alpha=self.config.alpha, max_iter=2000)

        reg.fit(features, targets)

        # Extract coefficients and bias
        regressor = reg.coef_.T.astype(np.float32)
        bias = reg.intercept_.T.astype(np.float32)

        return regressor, bias

    def _compute_nme(
        self, pred: NDArray[np.float32], true: NDArray[np.float32]
    ) -> float:
        """Compute Normalized Mean Error.

        Normalizes by inter-ocular distance (distance between eye centers).
        """
        # For 68-point annotation, left eye: 36-41, right eye: 42-47
        if len(pred) == 68:
            left_eye = true[36:42].mean(axis=0)
            right_eye = true[42:48].mean(axis=0)
            normalization = np.linalg.norm(left_eye - right_eye)
        else:
            # Fallback: use face size
            normalization = np.max(true, axis=0) - np.min(true, axis=0)
            normalization = np.linalg.norm(normalization)

        error = np.mean(np.linalg.norm(pred - true, axis=1))
        return error / normalization
