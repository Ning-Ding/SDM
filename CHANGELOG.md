# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- 🐛 **CRITICAL: Coordinate axis confusion in bounding box operations**: Fixed systematic coordinate swap in `expand_bbox()`, `bbox_to_square()`, `clip_bbox()` and `crop_and_resize()`. These functions incorrectly treated bbox `[x0, y0, x1, y1]` coordinates, swapping x/y axes and height/width limits. This caused faces to be cropped from wrong regions and landmarks to be misaligned, making all training and inference learn on corrupted data. Now correctly uses x as horizontal (width) and y as vertical (height).

- 🐛 **CRITICAL: Coordinate axis confusion in HOG feature extraction**: Fixed incorrect numpy array indexing in `HOGExtractor._compute_orientation_histogram()`. The code used `filtered[x_start:x_end, y_start:y_end]` but numpy arrays are indexed `[row, col] = [y, x]`, causing features to be extracted from wrong pixel neighborhoods. Now correctly uses `filtered[y_start:y_end, x_start:x_end]` to match landmark (x, y) coordinates to image (row=y, col=x) indexing.

- 🐛 **WingLoss CUDA device mismatch**: Fixed critical bug where `WingLoss.C` was created on CPU but not transferred to CUDA with the model, causing immediate training failure when using CUDA. Now properly registered as a buffer using `register_buffer()`.

- 🐛 **SDM evaluate IndexError with config mismatch**: Fixed crash in `SDM.evaluate()` when a model trained with `n_iterations=N` is loaded with a config having different `n_iterations=M`. The `mse_per_iteration` buffer is now sized from `len(self.regressors)` (actual model) instead of `self.config.n_iterations` (current config), preventing IndexError when iterating over regressors.

## [2.0.0] - 2025-11-08

### 🎉 完全重构

这是对原始 2016 年 SDM 实现的完全重构版本，专注于教学和现代化。

### Added

- ✨ **Python 3.12+ 支持**：使用现代 Python 特性和类型注解
- 📦 **模块化设计**：清晰的包结构 (`sdm.core`, `sdm.features`, `sdm.data`, `sdm.utils`)
- 🔧 **配置系统**：基于 Pydantic 的类型安全配置
- 🤖 **PyTorch 实现**：CNN 和 ResNet 模型用于性能对比
- 📚 **教学 Notebooks**：8 个详细的 Jupyter Notebooks
- 📖 **完整文档**：API 文档、理论文档、教程
- 🧪 **单元测试**：pytest 测试框架
- 🎨 **可视化工具**：丰富的可视化函数
- 📊 **性能对比**：SDM vs 深度学习方法对比脚本
- 🖥️ **命令行工具**：`sdm-train`, `sdm-infer`, `sdm-demo`
- 📝 **Example 脚本**：训练、推理、对比示例

### Changed

- 🔄 **数据加载**：从单文件改为模块化数据加载器
- 🔄 **特征提取**：HOG 实现重构，增加注释和文档
- 🔄 **训练流程**：清晰的训练 API，支持进度条和日志
- 🔄 **包管理**：从 pip 迁移到 uv
- 🔄 **代码风格**：使用 Ruff 进行代码检查和格式化
- 🔄 **类型检查**：使用 mypy 进行静态类型检查

### Improved

- ⚡ **性能优化**：优化的数据加载和特征提取
- 📈 **可扩展性**：易于添加新的特征提取器和模型
- 🐛 **Bug 修复**：修复原始实现中的边界情况
- 📝 **代码注释**：详细的中文注释和文档字符串

### Technical Details

#### 核心模块

- `sdm.core.model.SDMConfig`: 配置管理类
- `sdm.core.sdm.SDM`: SDM 主类，包含训练和推理
- `sdm.features.hog.HOGExtractor`: HOG 特征提取器
- `sdm.data.dataset.LFPWDataset`: LFPW 数据集类

#### PyTorch 模块

- `sdm.pytorch.model.LandmarkCNN`: 基础 CNN 模型
- `sdm.pytorch.model.LandmarkResNet`: ResNet 模型
- `sdm.pytorch.trainer.PyTorchTrainer`: 训练器类
- `sdm.pytorch.model.WingLoss`: Wing Loss 实现

#### 工具模块

- `sdm.utils.bbox`: 边界框处理
- `sdm.utils.image`: 图像处理
- `sdm.utils.visualization`: 可视化工具

### Breaking Changes

⚠️ 与原始 2016 版本不兼容：

- API 完全重构
- 配置方式改变
- 需要 Python 3.12+
- 使用 uv 替代 pip

### Migration Guide

从原始版本迁移：

```python
# 原始代码 (2016)
from main import model_parameters, train

parameters = model_parameters(N=3, alpha=0.001)
R, B, I = train(parameters)

# 新代码 (2025)
from sdm import SDM, SDMConfig
from sdm.data.dataset import LFPWDataset

config = SDMConfig(n_iterations=3, alpha=0.001)
dataset = LFPWDataset("data", split="train", config=config)
model = SDM(config)
model.train(dataset)
model.save("model.mat")
```

---

## [1.0.0] - 2016-10-21

### Added

- 初始版本
- 基本的 SDM 实现
- HOG 特征提取
- LFPW 数据集支持

---

*注：版本 2.0.0 是完全重构的教学版本，保留原始算法思想但代码完全重写。*
