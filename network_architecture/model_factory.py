import torch
import torch.nn as nn
import torch.nn.functional as F

# CLI / plan may use "simple_mil" or "nnmil"; checkpoints and save dirs use storage_model_type() -> "nnmil".
_SIMPLE_MIL_ALIASES = frozenset({"simple_mil", "nnmil"})


def canonical_implementation_type(model_type: str) -> str:
    """Map aliases to the branch key used in create_mil_model (simple_mil / ab_mil / ...)."""
    if model_type in _SIMPLE_MIL_ALIASES:
        return "simple_mil"
    return model_type


def storage_model_type(model_type: str) -> str:
    """Stable name for save_dir, best_*.pth, logs, and results_*.csv for the SimpleMIL family."""
    if model_type in _SIMPLE_MIL_ALIASES:
        return "nnmil"
    return model_type


def is_simple_mil_family(model_type: str) -> bool:
    return canonical_implementation_type(model_type) == "simple_mil"


def create_mil_model(model_type, input_dim=2560, hidden_dim=512, num_classes=2, activation='softmax', **kwargs):
    """
    Factory function to create MIL models from mil_zoo.
    
    Args:
        model_type (str): Type of MIL model
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension  
        num_classes (int): Number of output classes
        **kwargs: Additional model-specific parameters
    
    Returns:
        nn.Module: MIL model with original interface
    """
    dropout = kwargs.get('dropout', 0.25)
    impl_type = canonical_implementation_type(model_type)

    if impl_type == "simple_mil":
        from .models.simple_mil import SimpleMIL
        return SimpleMIL(input_dim=input_dim, hidden_dim=hidden_dim, pred_num=num_classes,
                        activation=activation, dropout=True)
    elif impl_type == "ab_mil":
        from .models.ab_mil import AB_MIL
        hidden_dim = 512
        return AB_MIL(L=hidden_dim, D=hidden_dim//4, num_classes=num_classes, 
                     dropout=dropout, in_dim=input_dim)
    
    else:
        available_models = get_available_models()
        raise ValueError(f"Unknown model type: {model_type} (impl={impl_type}). Available: {available_models}")


def get_available_models():
    """
    Get list of all available MIL models in mil_zoo.
    
    Returns:
        list: List of available model names
    """
    return ["nnmil", "simple_mil", "ab_mil"]
