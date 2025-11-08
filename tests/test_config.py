"""Tests for SDM configuration."""

import pytest
from sdm.core.model import SDMConfig


def test_sdm_config_default():
    """Test default configuration."""
    config = SDMConfig()

    assert config.n_iterations == 3
    assert config.alpha == 0.001
    assert config.image_size == (400, 400)
    assert config.n_landmarks == 68


def test_sdm_config_custom():
    """Test custom configuration."""
    config = SDMConfig(
        n_iterations=5,
        alpha=0.01,
        image_size=(512, 512),
    )

    assert config.n_iterations == 5
    assert config.alpha == 0.01
    assert config.image_size == (512, 512)


def test_sdm_config_validation():
    """Test configuration validation."""
    # Valid config
    config = SDMConfig(n_iterations=1, alpha=0.0)
    assert config.n_iterations == 1
    assert config.alpha == 0.0

    # Invalid iterations (should raise validation error)
    with pytest.raises(Exception):
        SDMConfig(n_iterations=0)

    # Invalid alpha (should raise validation error)
    with pytest.raises(Exception):
        SDMConfig(alpha=-0.1)


def test_hog_feature_dim():
    """Test HOG feature dimension calculation."""
    config = SDMConfig()
    dim = config.get_hog_feature_dim()

    # With default params and no block normalization:
    # 68 landmarks × 2 cells × 2 cells × 4 orientations = 2176
    assert dim == 68 * 2 * 2 * 4


def test_config_display(capsys):
    """Test configuration display."""
    config = SDMConfig(verbose=True)
    config.display()

    captured = capsys.readouterr()
    assert "SDM Configuration" in captured.out
    assert "Number of iterations" in captured.out
