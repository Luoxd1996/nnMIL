"""
Inference predictors for different task types.
"""

from nnMIL.inference.predictors.base_predictor import BasePredictor

# Import predictors (may fail if dependencies not available)
try:
    from nnMIL.inference.predictors.classification_predictor import ClassificationPredictor
except ImportError:
    ClassificationPredictor = None

try:
    from nnMIL.inference.predictors.survival_predictor import SurvivalPredictor
except ImportError:
    SurvivalPredictor = None

try:
    from nnMIL.inference.predictors.regression_predictor import RegressionPredictor
except ImportError:
    RegressionPredictor = None

__all__ = [
    'BasePredictor',
    'ClassificationPredictor',
    'SurvivalPredictor',
    'RegressionPredictor',
]

