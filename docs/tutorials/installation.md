# 安装指南

本文档将指导你完成 SDM Face Alignment 项目的安装。

## 环境要求

- **Python**: 3.12 或更高版本
- **操作系统**: Linux, macOS, or Windows
- **包管理器**: uv (推荐) 或 pip

## 使用 uv 安装（推荐）

### 1. 安装 uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 克隆仓库

```bash
git clone https://github.com/Ning-Ding/SDM.git
cd SDM
```

### 3. 安装依赖

```bash
# 基础安装
uv sync

# 安装开发依赖
uv sync --extra dev

# 安装 Notebook 依赖
uv sync --extra notebook

# 安装所有依赖
uv sync --extra all
```

## 使用 pip 安装

如果你更喜欢使用 pip：

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装项目
pip install -e .

# 安装开发依赖
pip install -e ".[dev,notebook]"
```

## 验证安装

```python
import sdm
print(sdm.__version__)

from sdm import SDMConfig, SDM
config = SDMConfig()
print("✓ SDM 安装成功！")
```

## 下载数据集

```bash
# 下载 LFPW 数据集
# 链接: https://pan.baidu.com/s/1jIJNg2q 密码: f36i

# 解压到项目根目录
SDM/
└── data/
    ├── trainset/
    ├── testset/
    └── bounding_boxes/
```

## 常见问题

### Q: uv sync 失败

如果 uv sync 失败，尝试：

```bash
# 清除缓存
rm -rf .venv
uv sync --reinstall
```

### Q: Python 版本不匹配

确保你使用的是 Python 3.12+：

```bash
python --version
# 或
python3.12 --version
```

### Q: CUDA 支持

如果你想使用 GPU 加速：

```bash
# 安装 PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 下一步

- 查看 [快速开始](quickstart.md)
- 浏览 [Jupyter Notebooks](../../notebooks/)
- 运行 [示例脚本](../../examples/)
