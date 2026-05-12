"""
Network architecture module for nnMIL.
Provides MIL model implementations and factory functions.
"""
from nnMIL.network_architecture.model_factory import (
    create_mil_model,
    canonical_implementation_type,
    storage_model_type,
    is_simple_mil_family,
    get_available_models,
)

__all__ = [
    'create_mil_model',
    'canonical_implementation_type',
    'storage_model_type',
    'is_simple_mil_family',
    'get_available_models',
]

