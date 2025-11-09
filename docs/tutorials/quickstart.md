# 快速开始

本指南将帮助你快速上手 SDM Face Alignment 项目。

## 1. 训练你的第一个模型

### 使用 Python API

```python
from sdm import SDM, SDMConfig
from sdm.data.dataset import LFPWDataset

# 创建配置
config = SDMConfig(
    n_iterations=3,      # SDM 迭代次数
    alpha=0.001,         # L1 正则化强度
    image_size=(400, 400),  # 目标图像大小
    verbose=True,        # 显示训练进度
)

# 加载训练数据
train_dataset = LFPWDataset(
    data_root="data",
    split="train",
    config=config,
)

print(f"训练样本数: {len(train_dataset)}")

# 初始化并训练模型
model = SDM(config)
model.train(train_dataset)

# 保存模型
model.save("models/my_sdm_model.mat")
```

### 使用命令行

```bash
# 训练模型
sdm-train --data-root data/ --output models/sdm_model.mat --iterations 3

# 查看帮助
sdm-train --help
```

## 2. 使用训练好的模型进行推理

```python
from sdm import SDM, SDMConfig
from sdm.data.dataset import LFPWDataset

# 加载模型
config = SDMConfig(mode="test")
model = SDM(config)
model.load("models/my_sdm_model.mat")

# 加载测试数据
test_dataset = LFPWDataset(
    data_root="data",
    split="test",
    config=config,
)

# 预测单张图像
image, landmarks_true, _ = test_dataset[0]
landmarks_pred, _ = model.predict(image)

print(f"预测关键点: {landmarks_pred.shape}")
```

## 3. 评估模型

```python
# 在测试集上评估
results = model.evaluate(test_dataset)

print(f"平均 MSE: {results['mean_error']:.4f}")
print(f"中位数 MSE: {results['median_error']:.4f}")
```

## 4. 可视化结果

```python
import matplotlib.pyplot as plt
from sdm.utils.visualization import draw_landmarks

# 获取第一张测试图像
image, landmarks_true, _ = test_dataset[0]
landmarks_pred, _ = model.predict(image)

# 绘制结果
vis_image = draw_landmarks(image, landmarks_pred, color=(255, 0, 0))  # 红色：预测
vis_image = draw_landmarks(vis_image, landmarks_true, color=(0, 255, 0))  # 绿色：真实

plt.figure(figsize=(10, 10))
plt.imshow(vis_image)
plt.title("面部关键点预测")
plt.axis('off')
plt.show()
```

## 5. 使用 PyTorch 深度学习模型

```python
import torch
from sdm import SDMConfig
from sdm.data.dataset import LFPWDataset
from sdm.pytorch.trainer import create_trainer
from sdm.pytorch.dataset import create_dataloaders

# 加载数据
config = SDMConfig()
train_dataset = LFPWDataset("data", split="train", config=config)
test_dataset = LFPWDataset("data", split="test", config=config)

# 创建 DataLoader
train_loader, test_loader = create_dataloaders(
    train_dataset,
    test_dataset,
    batch_size=32,
)

# 创建训练器
trainer = create_trainer(
    model_type="cnn",  # 'cnn' 或 'resnet'
    n_landmarks=68,
    loss_type="mse",   # 'mse' 或 'wing'
    learning_rate=0.001,
)

# 训练模型
history = trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    n_epochs=50,
    save_dir="models/pytorch",
)

print(f"最佳验证损失: {history['best_val_loss']:.6f}")
```

## 6. 方法对比

运行对比脚本来比较 SDM 和深度学习方法：

```bash
python examples/compare_methods.py \
    --data-root data/ \
    --sdm-model models/sdm_model.mat \
    --pytorch-model models/pytorch/best_model.pth
```

## 7. 使用 Jupyter Notebooks 学习

我们提供了详细的教学 notebooks：

```bash
# 启动 Jupyter
uv sync --extra notebook
jupyter lab

# 打开 notebooks/ 目录
# 从 00_setup_and_data.ipynb 开始
```

## 常见任务

### 修改配置

```python
# 创建自定义配置
config = SDMConfig(
    n_iterations=5,           # 增加迭代次数
    alpha=0.005,              # 增加正则化
    image_size=(512, 512),    # 更大的图像
    orientations=8,           # 更多的方向bins
)
```

### 使用 YAML 配置

```python
import yaml
from sdm.core.model import SDMConfig

# 从 YAML 加载配置
with open("configs/default.yaml") as f:
    config_dict = yaml.safe_load(f)

config = SDMConfig(**config_dict)
```

### 保存预测结果

```python
from sdm.data.loader import save_landmarks

# 保存为 .pts 格式
save_landmarks(landmarks_pred, "output/result.pts", format="pts")

# 保存为 .txt 格式
save_landmarks(landmarks_pred, "output/result.txt", format="txt")
```

## 下一步

- 📚 浏览 [教学 Notebooks](../../notebooks/)
- 📖 阅读 [Bug 修复记录](../bug_fixes.md)
- 📊 运行 [示例脚本](../../examples/)
- 🧪 运行 [单元测试](../../tests/)

## 获取帮助

如果遇到问题：

1. 查看 [文档](../)
2. 查看 [Issues](https://github.com/Ning-Ding/SDM/issues)
3. 查看项目 [README](../../README.md)
