"""
nnMIL Inference Engine

Unified inference interface following nnUNet design principles.
Automatically selects the appropriate predictor based on task type.
"""

import os
import torch
import logging
from typing import Optional, Dict, Any

from nnMIL.inference.predictors import (
    ClassificationPredictor,
    SurvivalPredictor,
    RegressionPredictor,
)


class InferenceEngine:
    """
    Unified inference engine that automatically selects the appropriate predictor
    based on task type (from plan file or dataset configuration).
    
    Similar to nnUNetv2_predict, this provides a unified interface for all inference tasks.
    """
    
    def __init__(self, plan_path: Optional[str] = None, checkpoint_path: str = None, 
                 task_type: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            plan_path: Path to dataset_plan.json (if using plan-based workflow)
            checkpoint_path: Path to model checkpoint
            task_type: Task type ('classification', 'survival', 'regression')
                      If None, will be inferred from plan file
            device: Device to use ('cuda', 'cpu', etc.). If None, auto-detect.
        """
        self.plan_path = plan_path
        self.checkpoint_path = checkpoint_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine task type
        if task_type:
            self.task_type = task_type
        elif plan_path and os.path.exists(plan_path):
            import json
            with open(plan_path, 'r') as f:
                plan = json.load(f)
            self.task_type = plan.get('dataset_info', {}).get('task_type', 'classification')
        else:
            self.task_type = 'classification'  # Default
        
        # Initialize appropriate predictor
        self.predictor = self._create_predictor()
        
    def _create_predictor(self):
        """Create the appropriate predictor based on task type"""
        if self.task_type == 'classification':
            return ClassificationPredictor()
        elif self.task_type == 'survival':
            return SurvivalPredictor()
        elif self.task_type == 'regression':
            return RegressionPredictor()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def predict(self, test_dataset, model: Optional[torch.nn.Module] = None,
                save_dir: Optional[str] = None, logger: Optional[logging.Logger] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Run inference on test dataset.
        
        Args:
            test_dataset: Test dataset
            model: Trained model (if None, will load from checkpoint)
            save_dir: Directory to save results
            logger: Logger instance
            **kwargs: Additional arguments for predictor
            
        Returns:
            Dictionary containing metrics and results
        """
        # Load model if not provided
        if model is None:
            if self.checkpoint_path is None:
                raise ValueError("Either model or checkpoint_path must be provided")
            model = self._load_model(test_dataset, **kwargs)
        
        # Run prediction
        # Convert device string to torch.device if needed
        device_obj = torch.device(self.device) if isinstance(self.device, str) else self.device
        
        return self.predictor.predict(
            test_dataset=test_dataset,
            model=model,
            device=device_obj,
            save_dir=save_dir,
            logger=logger,
            **kwargs
        )
    
    def _load_model(self, dataset, **kwargs):
        """Load model from checkpoint"""
        from nnMIL.network_architecture.model_factory import create_mil_model
        
        # Get model configuration from kwargs or plan or dataset
        input_dim = kwargs.get('input_dim', 2560)
        hidden_dim = kwargs.get('hidden_dim', 512)
        
        # Try to get num_classes from multiple sources
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            if hasattr(dataset, 'num_classes'):
                num_classes = dataset.num_classes
            elif hasattr(dataset, 'label_to_idx'):
                num_classes = len(dataset.label_to_idx)
            else:
                num_classes = 2  # Default
        
        dropout = kwargs.get('dropout', 0.25)
        model_type = kwargs.get('model_type', 'simple_mil')
        
        # Create model
        model = create_mil_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Load checkpoint
        device = torch.device(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the whole dict is the state_dict
                model.load_state_dict(checkpoint)
        else:
            # Direct state_dict
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model

