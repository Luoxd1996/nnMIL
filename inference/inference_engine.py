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
                      If None, will be inferred from plan file, checkpoint, or training_config.json
            device: Device to use ('cuda', 'cpu', etc.). If None, auto-detect.
        """
        self.plan_path = plan_path
        self.checkpoint_path = checkpoint_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine task type from multiple sources
        if task_type:
            self.task_type = task_type
        elif plan_path and os.path.exists(plan_path):
            import json
            with open(plan_path, 'r') as f:
                plan = json.load(f)
            # task_type can be at top level or under dataset_info
            self.task_type = plan.get('task_type') or plan.get('dataset_info', {}).get('task_type', 'classification')
        elif checkpoint_path and os.path.exists(checkpoint_path):
            # Try to get task_type from checkpoint or training_config.json
            self.task_type = self._infer_task_type_from_checkpoint()
        else:
            self.task_type = 'classification'  # Default
        
        # Initialize appropriate predictor
        self.predictor = self._create_predictor()
    
    def _infer_task_type_from_checkpoint(self) -> str:
        """Try to infer task_type from checkpoint or training_config.json"""
        import json
        
        # First, try to load checkpoint and get task_type from it
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                device = torch.device(self.device)
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict):
                    # Check if checkpoint has dataset_info
                    if 'dataset_info' in checkpoint:
                        dataset_info = checkpoint['dataset_info']
                        if isinstance(dataset_info, dict) and 'task_type' in dataset_info:
                            return dataset_info['task_type']
            except Exception as e:
                logging.warning(f"Could not read task_type from checkpoint: {e}")
        
        # Second, try to find training_config.json in checkpoint directory
        if self.checkpoint_path:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            config_path = os.path.join(checkpoint_dir, 'training_config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    dataset_info = config.get('dataset_info', {})
                    if isinstance(dataset_info, dict) and 'task_type' in dataset_info:
                        return dataset_info['task_type']
                except Exception as e:
                    logging.warning(f"Could not read task_type from training_config.json: {e}")
        
        # Default fallback
        return 'classification'
        
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
        
        # Try to get config from checkpoint or training_config.json first
        checkpoint_config = self._load_config_from_checkpoint()
        
        # Get model configuration from multiple sources (priority: kwargs > checkpoint_config > plan > defaults)
        # Use get() with None check to handle 0 values correctly
        input_dim = (kwargs.get('input_dim') if kwargs.get('input_dim') is not None 
                    else checkpoint_config.get('input_dim') if checkpoint_config.get('input_dim') is not None
                    else checkpoint_config.get('feature_dimension') if checkpoint_config.get('feature_dimension') is not None
                    else 2560)
        hidden_dim = (kwargs.get('hidden_dim') if kwargs.get('hidden_dim') is not None
                     else checkpoint_config.get('hidden_dim') if checkpoint_config.get('hidden_dim') is not None
                     else 512)
        
        # Try to get num_classes from multiple sources (priority: kwargs > checkpoint_config > plan > task_type logic)
        # CRITICAL: For survival and regression, task_type determines num_classes first
        # Do NOT use dataset.num_classes for these tasks as they may be incorrect
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            if self.task_type == 'survival':
                # For survival: check if using NLLSurvLoss (requires bins)
                survival_loss = (kwargs.get('survival_loss') if kwargs.get('survival_loss') is not None
                                else checkpoint_config.get('survival_loss') if checkpoint_config.get('survival_loss') is not None
                                else 'cox')
                nll_bins = (kwargs.get('nll_bins') if kwargs.get('nll_bins') is not None
                           else checkpoint_config.get('nll_bins') if checkpoint_config.get('nll_bins') is not None
                           else None)
                
                # If still None, try plan file
                if nll_bins is None and self.plan_path and os.path.exists(self.plan_path):
                    import json
                    with open(self.plan_path, 'r') as f:
                        plan = json.load(f)
                    training_config = plan.get('training_configuration', {})
                    survival_loss = training_config.get('survival_loss', survival_loss)
                    nll_bins = training_config.get('nll_bins')
                
                # If nll_bins is None, use default 4 for nllsurv
                if nll_bins is None:
                    nll_bins = 4  # Default for NLLSurv
                
                if survival_loss == 'nllsurv':
                    num_classes = nll_bins  # NLLSurv: output per time bin
                else:
                    num_classes = 1  # Cox loss: single risk score
            elif self.task_type == 'regression':
                num_classes = 1  # Regression: single continuous value
            else:
                # Classification: try checkpoint config first, then dataset
                num_classes = checkpoint_config.get('num_classes')
                if num_classes is None:
                    if hasattr(dataset, 'num_classes'):
                        num_classes = dataset.num_classes
                    elif hasattr(dataset, 'label_to_idx'):
                        num_classes = len(dataset.label_to_idx)
                    else:
                        num_classes = 2  # Default for classification
        
        dropout = (kwargs.get('dropout') if kwargs.get('dropout') is not None
                  else checkpoint_config.get('dropout') if checkpoint_config.get('dropout') is not None
                  else 0.25)
        model_type = (kwargs.get('model_type') if kwargs.get('model_type') is not None
                     else checkpoint_config.get('model_type') if checkpoint_config.get('model_type') is not None
                     else 'simple_mil')
        
        # Debug: print num_classes determination for troubleshooting
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Creating model: task_type={self.task_type}, num_classes={num_classes}, model_type={model_type}")
        if self.task_type == 'survival' and self.plan_path and os.path.exists(self.plan_path):
            import json
            with open(self.plan_path, 'r') as f:
                plan = json.load(f)
            training_config = plan.get('training_configuration', {})
            logger.info(f"Plan config: survival_loss={training_config.get('survival_loss')}, nll_bins={training_config.get('nll_bins')}")
        
        # Create model
        model = create_mil_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Load checkpoint
        # First load to CPU to avoid CUDA device mismatch, then move to target device
        device = torch.device(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
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
    
    def _load_config_from_checkpoint(self) -> Dict[str, Any]:
        """Try to load configuration from checkpoint or training_config.json"""
        import json
        config = {}
        
        # First, try to load from checkpoint itself
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                device = torch.device(self.device)
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict):
                    # Check if checkpoint has 'config' field
                    if 'config' in checkpoint:
                        config.update(checkpoint['config'])
                    # Also check for actual_configuration (from training_config.json structure)
                    if 'actual_configuration' in checkpoint:
                        config.update(checkpoint['actual_configuration'])
            except Exception as e:
                logging.debug(f"Could not read config from checkpoint: {e}")
        
        # Second, try to load from training_config.json in checkpoint directory
        if self.checkpoint_path:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            config_path = os.path.join(checkpoint_dir, 'training_config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        training_config_dict = json.load(f)
                    # Check for actual_configuration field
                    if 'actual_configuration' in training_config_dict:
                        config.update(training_config_dict['actual_configuration'])
                except Exception as e:
                    logging.debug(f"Could not read config from training_config.json: {e}")
        
        return config

