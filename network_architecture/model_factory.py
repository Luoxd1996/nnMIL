import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    if model_type == "simple_mil":
        from .models.simple_mil import SimpleMIL
        return SimpleMIL(input_dim=input_dim, hidden_dim=hidden_dim, pred_num=num_classes,
                        activation=activation, dropout=True)
    elif model_type == "ab_mil":
        from .models.ab_mil import AB_MIL
        hidden_dim = 512
        return AB_MIL(L=hidden_dim, D=hidden_dim//4, num_classes=num_classes, 
                     dropout=dropout, in_dim=input_dim)
    
    else:
        available_models = get_available_models()
        raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")

def get_available_models():
    """
    Get list of all available MIL models in mil_zoo.
    
    Returns:
        list: List of available model names
    """
    return [
        "simple_mil",
        "ab_mil",
    ]
