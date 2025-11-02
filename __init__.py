"""
nnMIL: Neural Network Multiple Instance Learning Package

A modular MIL framework following nnUNet v2 design principles.

Main modules:
- network_architecture: MIL model implementations
- data: Dataset loading and management
- training: Trainers, losses, samplers, and callbacks
- inference: Model inference utilities
- preprocessing: Data preprocessing and experiment planning
- utilities: Utility functions
- run: Command-line entry points
"""

__version__ = "1.0.0"

# Main module exports
from nnMIL import network_architecture
from nnMIL import data
from nnMIL import training
from nnMIL import inference
from nnMIL import preprocessing
from nnMIL import utilities
from nnMIL import run

__all__ = [
    'network_architecture',
    'data',
    'training',
    'inference',
    'preprocessing',
    'utilities',
    'run',
]
