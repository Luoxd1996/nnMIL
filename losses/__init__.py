"""
Backward compatibility module for nnMIL.losses
All imports are redirected to nnMIL.training.losses
"""
import warnings
warnings.warn(
    "nnMIL.losses is deprecated. Use nnMIL.training.losses instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from training.losses
from nnMIL.training.losses.survival_loss import SurvivalLoss, survival_c_index
from nnMIL.training.losses.survival_loss_nll import NLLSurvLoss
from nnMIL.training.losses.regression_loss import CombinedRegressionLoss

__all__ = [
    'SurvivalLoss', 
    'survival_c_index', 
    'NLLSurvLoss',
    'CombinedRegressionLoss',
]
