"""
Training trainers for nnMIL.
Provides unified trainer classes for different task types.
"""

from nnMIL.training.trainers.base_trainer import BaseTrainer

# Import task-specific trainers (may fail if dependencies not available)
try:
    from nnMIL.training.trainers.classification_trainer import ClassificationTrainer
except ImportError:
    ClassificationTrainer = None

try:
    from nnMIL.training.trainers.regression_trainer import RegressionTrainer
except ImportError:
    RegressionTrainer = None

try:
    from nnMIL.training.trainers.survival_trainer import SurvivalTrainer
except ImportError:
    SurvivalTrainer = None

try:
    from nnMIL.training.trainers.survival_porpoise_trainer import SurvivalPorpoiseTrainer
except ImportError:
    SurvivalPorpoiseTrainer = None

__all__ = [
    'BaseTrainer',
    'ClassificationTrainer',
    'RegressionTrainer',
    'SurvivalTrainer',
    'SurvivalPorpoiseTrainer',
]

