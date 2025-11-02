"""
Backward compatibility module for nnMIL.trainers
All imports are redirected to nnMIL.training.trainers
"""
import warnings
warnings.warn(
    "nnMIL.trainers is deprecated. Use nnMIL.training.trainers instead.",
    DeprecationWarning,
    stacklevel=2
)

# Note: We don't import the training scripts themselves as they are typically run directly
__all__ = []

