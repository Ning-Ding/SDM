"""SDM Face Alignment - 教学重构版本

Supervised Descent Method for Face Alignment using Python 3.12+

This package provides:
- Classic SDM implementation with HOG features
- PyTorch-based modern deep learning comparison
- Educational notebooks and examples
- Utility functions for face alignment tasks
"""

__version__ = "2.0.0"
__author__ = "Ning Ding"

from sdm.core.model import SDMConfig
from sdm.core.sdm import SDM

__all__ = ["SDMConfig", "SDM", "__version__"]
