"""Tests for PyTorch models."""

import pytest
import torch

from sdm.pytorch.model import LandmarkCNN, LandmarkResNet, WingLoss


def test_landmark_cnn_forward():
    """Test LandmarkCNN forward pass."""
    model = LandmarkCNN(n_landmarks=68, input_channels=1)

    # Test with batch of grayscale images
    batch_size = 4
    x = torch.randn(batch_size, 1, 400, 400)

    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, 68 * 2)


def test_landmark_resnet_forward():
    """Test LandmarkResNet forward pass."""
    model = LandmarkResNet(n_landmarks=68, input_channels=1)

    # Test with batch of grayscale images
    batch_size = 4
    x = torch.randn(batch_size, 1, 400, 400)

    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, 68 * 2)


def test_wing_loss_cpu():
    """Test Wing Loss on CPU."""
    loss_fn = WingLoss(omega=10.0, epsilon=2.0)

    pred = torch.randn(4, 136)  # 68 landmarks * 2
    target = torch.randn(4, 136)

    loss = loss_fn(pred, target)

    assert loss.ndim == 0  # Scalar
    assert loss.item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wing_loss_cuda():
    """Test Wing Loss on CUDA (device mismatch bug fix)."""
    loss_fn = WingLoss(omega=10.0, epsilon=2.0).cuda()

    pred = torch.randn(4, 136).cuda()
    target = torch.randn(4, 136).cuda()

    # This should NOT raise a device mismatch error
    loss = loss_fn(pred, target)

    assert loss.ndim == 0
    assert loss.item() > 0
    assert loss.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_device_transfer():
    """Test that model transfers to CUDA correctly."""
    model = LandmarkCNN(n_landmarks=68)
    model = model.cuda()

    x = torch.randn(2, 1, 400, 400).cuda()
    output = model(x)

    assert output.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wing_loss_with_model_on_cuda():
    """Test Wing Loss used with CUDA model (integration test)."""
    model = LandmarkCNN(n_landmarks=68).cuda()
    loss_fn = WingLoss().cuda()

    x = torch.randn(2, 1, 400, 400).cuda()
    target = torch.randn(2, 136).cuda()

    # Forward pass
    pred = model(x)

    # Compute loss - should work without device errors
    loss = loss_fn(pred, target)

    # Backward pass
    loss.backward()

    assert loss.item() > 0
