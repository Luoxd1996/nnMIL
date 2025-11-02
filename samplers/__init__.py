"""
Backward compatibility module for nnMIL.samplers
All imports are redirected to nnMIL.training.samplers
"""
import warnings
warnings.warn(
    "nnMIL.samplers is deprecated. Use nnMIL.training.samplers instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from training.samplers
from nnMIL.training.samplers.survival_sampler import BalancedSurvivalSampler, StratifiedSurvivalSampler, RiskSetBatchSampler
from nnMIL.training.samplers.classification_sampler import BalancedBatchSampler, AUCBatchSampler
from nnMIL.training.samplers.regression_sampler import RegressionBatchSampler

__all__ = [
    'BalancedSurvivalSampler', 
    'StratifiedSurvivalSampler', 
    'RiskSetBatchSampler',
    'BalancedBatchSampler',
    'AUCBatchSampler',
    'RegressionBatchSampler',
]
