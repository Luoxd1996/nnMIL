"""
Backward compatibility module for nnMIL.models
All imports are redirected to nnMIL.network_architecture
"""
import warnings
warnings.warn(
    "nnMIL.models is deprecated. Use nnMIL.network_architecture instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from network_architecture
from nnMIL.network_architecture.model_factory import create_mil_model
from nnMIL.network_architecture.models import *

__all__ = [
    'create_mil_model',
]
