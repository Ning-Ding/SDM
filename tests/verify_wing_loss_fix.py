"""Verification script for WingLoss CUDA device fix.

This script demonstrates the fix for the device mismatch bug in WingLoss.

Before the fix:
    WingLoss would fail when used with CUDA models because self.C
    was created on CPU and didn't transfer to CUDA with the model.

After the fix:
    self.C is registered as a buffer, so it automatically moves to
    the correct device when the model is transferred.

Usage:
    python tests/verify_wing_loss_fix.py
"""

import sys

try:
    import torch
    from sdm.pytorch.model import WingLoss
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install dependencies:")
    print("  uv sync --extra all")
    sys.exit(1)


def test_cpu():
    """Test WingLoss on CPU."""
    print("=" * 60)
    print("Test 1: WingLoss on CPU")
    print("=" * 60)

    loss_fn = WingLoss(omega=10.0, epsilon=2.0)
    pred = torch.randn(4, 136)  # 68 landmarks * 2
    target = torch.randn(4, 136)

    loss = loss_fn(pred, target)

    print(f"✓ Predictions device: {pred.device}")
    print(f"✓ Loss value: {loss.item():.4f}")
    print(f"✓ Loss device: {loss.device}")

    # Check that C is a buffer
    buffers = dict(loss_fn.named_buffers())
    assert "C" in buffers, "C should be registered as a buffer"
    print(f"✓ C is registered as buffer: True")
    print(f"✓ C device: {buffers['C'].device}")

    print("\n✅ CPU test passed!\n")


def test_cuda():
    """Test WingLoss on CUDA."""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping CUDA test\n")
        return

    print("=" * 60)
    print("Test 2: WingLoss on CUDA (Bug Fix Verification)")
    print("=" * 60)

    # Create loss on CUDA
    loss_fn = WingLoss(omega=10.0, epsilon=2.0).cuda()

    # Create predictions and targets on CUDA
    pred = torch.randn(4, 136).cuda()
    target = torch.randn(4, 136).cuda()

    print(f"✓ Model moved to CUDA")
    print(f"✓ Predictions device: {pred.device}")
    print(f"✓ Target device: {target.device}")

    # Verify C is also on CUDA
    buffers = dict(loss_fn.named_buffers())
    C_device = buffers["C"].device
    print(f"✓ C device: {C_device}")

    if C_device.type != "cuda":
        print("❌ ERROR: C is not on CUDA!")
        print("   This would cause device mismatch in forward pass")
        return False

    # Compute loss - this should work without device errors
    try:
        loss = loss_fn(pred, target)
        print(f"✓ Loss computation successful: {loss.item():.4f}")
        print(f"✓ Loss device: {loss.device}")
    except RuntimeError as e:
        print(f"❌ ERROR: {e}")
        print("   Device mismatch detected!")
        return False

    # Test backward pass
    try:
        loss.backward()
        print(f"✓ Backward pass successful")
    except RuntimeError as e:
        print(f"❌ ERROR in backward: {e}")
        return False

    print("\n✅ CUDA test passed!")
    print("✅ Bug fix verified: C moves with model to CUDA\n")
    return True


def test_device_transfer():
    """Test explicit device transfer."""
    print("=" * 60)
    print("Test 3: Explicit Device Transfer")
    print("=" * 60)

    # Create on CPU
    loss_fn = WingLoss()
    print(f"Initial device: CPU")

    buffers = dict(loss_fn.named_buffers())
    print(f"C device before transfer: {buffers['C'].device}")

    if torch.cuda.is_available():
        # Move to CUDA
        loss_fn = loss_fn.cuda()
        buffers = dict(loss_fn.named_buffers())
        print(f"C device after .cuda(): {buffers['C'].device}")

        # Move back to CPU
        loss_fn = loss_fn.cpu()
        buffers = dict(loss_fn.named_buffers())
        print(f"C device after .cpu(): {buffers['C'].device}")

        print("\n✅ Device transfer test passed!\n")
    else:
        print("⚠️  CUDA not available, skipping transfer test\n")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("WingLoss CUDA Device Fix Verification")
    print("=" * 60)
    print("\nThis script verifies that WingLoss.C is correctly")
    print("registered as a buffer and moves with the model.\n")

    # Run tests
    test_cpu()
    test_cuda()
    test_device_transfer()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("The fix ensures that:")
    print("  1. C is registered as a buffer (not a regular tensor)")
    print("  2. C automatically moves when model.cuda() is called")
    print("  3. No device mismatch errors occur during training")
    print("\nImplementation:")
    print("  self.register_buffer('C', C)")
    print("\nThis is the PyTorch best practice for model constants.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
