# Bug Fixes

This document tracks critical bug fixes made to the SDM project after the initial v2.0.0 release.

---

## 🐛 Bug #1: WingLoss CUDA Device Mismatch

**Severity**: Critical
**Component**: `sdm.pytorch.model.WingLoss`
**Status**: ✅ Fixed
**Commit**: `86a5359`

### Problem

When using `WingLoss` with CUDA models, training would fail immediately with a device mismatch error:

```python
# Create model and loss on CUDA
model = LandmarkCNN().cuda()
loss_fn = WingLoss().cuda()

# Training would crash here
pred = model(x_cuda)  # pred on CUDA
loss = loss_fn(pred, target_cuda)  # ERROR: device mismatch
```

**Root Cause**:
- In `WingLoss.__init__()`, the constant `C` was created as a regular tensor on CPU:
  ```python
  self.C = self.omega - self.omega * torch.log(...)  # Always on CPU
  ```
- When the model was moved to CUDA via `.cuda()`, regular attributes don't transfer
- In `forward()`, the operation `delta - self.C` mixed CUDA and CPU tensors
- PyTorch raised: `RuntimeError: Expected all tensors to be on the same device`

### Solution

Register `C` as a **buffer** instead of a regular attribute:

```python
def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
    super().__init__()
    self.omega = omega
    self.epsilon = epsilon
    # Register C as a buffer so it moves with the model
    C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
    self.register_buffer('C', C)
```

**Why this works**:
- Buffers are tensors that are part of the module's state
- They automatically move when `.cuda()` or `.cpu()` is called
- They're saved in `state_dict` but not updated by optimizers
- This is the PyTorch best practice for model constants

### Testing

Comprehensive tests added in `tests/test_pytorch.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wing_loss_cuda():
    """Test Wing Loss on CUDA (device mismatch bug fix)."""
    loss_fn = WingLoss().cuda()
    pred = torch.randn(4, 136).cuda()
    target = torch.randn(4, 136).cuda()

    # Should NOT raise device mismatch error
    loss = loss_fn(pred, target)
    assert loss.device.type == 'cuda'
```

Verification script: `tests/verify_wing_loss_fix.py`

### Impact

- **Before fix**: Any CUDA training with WingLoss would fail immediately
- **After fix**: WingLoss works seamlessly on both CPU and CUDA

---

## 🐛 Bug #2: SDM Evaluate IndexError with Config Mismatch

**Severity**: High
**Component**: `sdm.core.sdm.SDM.evaluate()`
**Status**: ✅ Fixed
**Commit**: `459484d`

### Problem

When evaluating a model that was trained with different `n_iterations` than the current config:

```python
# Train with 5 iterations
config_train = SDMConfig(n_iterations=5)
model = SDM(config_train)
model.train(dataset)
model.save("model.mat")

# Load with default config (n_iterations=3)
config_load = SDMConfig()  # n_iterations=3 by default
model_loaded = SDM(config_load)
model_loaded.load("model.mat")  # Loads 5 regressors

# Evaluate crashes!
model_loaded.evaluate(dataset)  # IndexError: list index out of range
```

**Root Cause**:
```python
# In evaluate()
mse_per_iteration = [[] for _ in range(self.config.n_iterations)]  # Length 3

for idx in range(len(dataset)):
    landmarks = self.initial_shape.copy()
    for i in range(len(self.regressors)):  # Loops 5 times!
        ...
        mse_per_iteration[i].append(iter_mse)  # IndexError when i >= 3
```

The list was sized based on `config.n_iterations` (3) but the loop ran `len(self.regressors)` times (5).

### Solution

Size the buffer based on the **actual model**, not the config:

```python
# Before (WRONG)
mse_per_iteration = [[] for _ in range(self.config.n_iterations)]

# After (CORRECT)
mse_per_iteration = [[] for _ in range(len(self.regressors))]
```

### Why This Matters

1. **Persistent models**: When you save and load models, the number of regressors is determined by the training configuration, not the loading configuration
2. **Flexibility**: Users should be able to load models without worrying about config matching
3. **Robustness**: The evaluation should work based on the actual model structure

### Testing

Comprehensive tests added in `tests/test_sdm.py`:

```python
def test_sdm_config_mismatch_bug_fix():
    """Test that evaluation handles config mismatch."""
    # Train with 5 iterations
    train_config = SDMConfig(n_iterations=5, alpha=0.0, verbose=False)
    model = SDM(train_config)
    model.train(dataset)
    model.save(model_path)

    # Load with 3 iterations (default)
    load_config = SDMConfig(verbose=False)  # n_iterations=3
    loaded_model = SDM(load_config)
    loaded_model.load(model_path)

    # Should NOT crash
    results = loaded_model.evaluate(dataset)
    assert len(results['mse_per_iteration']) == 5  # From model, not config
```

Demonstration script: `tests/test_config_mismatch_scenario.py`

### Impact

- **Before fix**: Crash when evaluating models trained with different iterations
- **After fix**: Models can be evaluated regardless of config mismatch

---

## Summary

Both bugs were critical issues that would cause immediate failures in common use cases:

1. **WingLoss CUDA bug**: Prevented any CUDA training with WingLoss
2. **Config mismatch bug**: Prevented evaluation of persisted models with different training configs

Both have been fixed with:
- ✅ Root cause identified and corrected
- ✅ Comprehensive test coverage
- ✅ Verification scripts
- ✅ Documentation updated

**Credits**: Thank you to the user who discovered and reported these bugs! 🙏

---

## Testing Verification

To verify both fixes:

```bash
# Run all tests
uv run pytest tests/

# Run specific tests
uv run pytest tests/test_pytorch.py::test_wing_loss_cuda
uv run pytest tests/test_sdm.py::test_sdm_config_mismatch_bug_fix

# Run verification scripts
python tests/verify_wing_loss_fix.py
python tests/test_config_mismatch_scenario.py
```

---

*Last updated: 2025-11-08*
