"""
nnMIL Inference Module

Unified inference interface following nnUNet design principles.
"""

from nnMIL.inference.inference_engine import InferenceEngine
from nnMIL.inference.predictors import (
    ClassificationPredictor,
    SurvivalPredictor,
    RegressionPredictor,
)

__all__ = [
    'InferenceEngine',
    'ClassificationPredictor',
    'SurvivalPredictor',
    'RegressionPredictor',
]

