"""Independent GPU wrappers for tnlearn.

This package is intentionally separated from `tnlearn` to avoid modifying
existing source files while providing GPU-first entry points.
"""

from tnlearn_gpu.mlpclassifier_gpu import MLPClassifier_gpu
from tnlearn_gpu.mlpregressor_gpu import MLPRegressor_gpu

__all__ = [
    "MLPClassifier_gpu",
    "MLPRegressor_gpu",
]
