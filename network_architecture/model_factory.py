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
    
    elif model_type == "ab_mil_fixed_feat":
        from .models.ab_mil_fixed_feat import AB_MIL
        hidden_dim = 512
        return AB_MIL(L=hidden_dim, D=hidden_dim//4, num_classes=num_classes, 
                     dropout=dropout, in_dim=input_dim)
        
    elif model_type == "trans_mil":
        from .models.trans_mil import TRANS_MIL
        return TRANS_MIL(num_classes=num_classes, dropout=dropout, 
                        act=nn.ReLU(), in_dim=input_dim)
    
    elif model_type == "wikg_mil":
        from .models.wikg_mil import WIKG_MIL
        # WIKG_MIL expects [B, N, D] input and returns dict
        # Adjust topk to be smaller for compatibility
        topk = min(6, kwargs.get('topk', 6))
        hidden_dim = 512
        return WIKG_MIL(in_dim=input_dim, dim_hidden=hidden_dim, 
                       num_classes=num_classes, dropout=dropout, topk=topk)
    
    elif model_type == "ilra_mil":
        from .models.ilra_mil import ILRA_MIL
        hidden_dim = 512
        return ILRA_MIL(num_layers=2, in_dim=input_dim, num_classes=num_classes, 
                       hidden_feat=hidden_dim, num_heads=8, topk=1, ln=False)
    
    elif model_type == "ds_mil":
        from .models.ds_mil import DS_MIL, IClassifier, BClassifier, FCLayer
        hidden_dim = 512
        # Create feature extractor that returns single output (not tuple)
        class FeatureExtractor(nn.Module):
            def __init__(self, in_size, out_size):
                super().__init__()
                self.fc = nn.Linear(in_size, out_size)
                
            def forward(self, x):
                return self.fc(x)  # Return single tensor, not tuple
        
        # DS_MIL requires instance classifier and bag classifier
        i_classifier = IClassifier(
            feature_extractor=FeatureExtractor(input_dim, hidden_dim),
            feature_size=hidden_dim,
            output_class=num_classes
        )
        b_classifier = BClassifier(
            input_size=hidden_dim,
            output_class=num_classes,
            dropout_v=dropout
        )
        return DS_MIL(i_classifier, b_classifier)
    
    elif model_type == "dtfd_mil":
        hidden_dim = 512
        from .models.dtfd_mil import Attention_with_Classifier
        base_model = Attention_with_Classifier(L=input_dim, D=hidden_dim//4, K=1, 
                                             num_cls=num_classes, droprate=dropout)
        
        # DTFD_MIL expects [N, L] input (traditional MIL, batch_size=1)
        class DTFD_MIL_Adapter(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x, **kwargs):
                # Squeeze batch dimension: [1, N, L] -> [N, L]
                x = x.squeeze(0)
                output = self.model(x, **kwargs)
                
                # Add batch dimension back to logits and fix shape
                if isinstance(output, dict) and 'logits' in output:
                    logits = output['logits']
                    if logits.dim() == 2:  # [K, num_cls] -> [1, num_cls]
                        output['logits'] = logits.mean(0, keepdim=True)
                    else:
                        output['logits'] = logits.unsqueeze(0)
                
                return output
        
        return DTFD_MIL_Adapter(base_model)
    
    elif model_type == "vision_transformer":
        from .models.vision_transformer import VisionTransformer
        
        # VisionTransformer parameters - hardcoded configuration
        dim_model = 512
        n_layers = 2
        n_heads = 8
        dim_feedforward = 512
        dropout = 0.0
        use_alibi = True
        
        return VisionTransformer(
            dim_output=num_classes,
            dim_input=input_dim,
            dim_model=dim_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_alibi=use_alibi
        )
    
    elif model_type == "rrt":
        from .models.rrt import RRT
        
        # RRT parameters - hardcoded configuration
        mlp_dim = 512
        n_layers = 2
        n_heads = 8
        region_num = 8
        trans_dropout = 0.1
        dropout = kwargs.get('dropout', 0.25)
        
        return RRT(
            input_dim=input_dim,
            mlp_dim=mlp_dim,
            n_classes=num_classes,
            dropout=dropout,
            n_layers=n_layers,
            n_heads=n_heads,
            region_num=region_num,
            trans_dropout=trans_dropout,
            attn='rmsa',
            pool='attn',
            ffn=False,
            epeg=True,
            qkv_bias=True
        )
    
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
        "trans_mil",
        "wikg_mil",
        "ilra_mil",
        "ds_mil",
        "dtfd_mil",
        "vision_transformer",
        "rrt"
    ]
