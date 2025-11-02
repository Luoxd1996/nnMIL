#!/usr/bin/env python3
"""
Base Trainer for nnMIL
All task-specific trainers inherit from this base class.
"""

import os
import sys
import json
import logging
import random
import warnings
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

# Add parent directory to path for nnMIL imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nnMIL.utilities.plan_loader import load_plan, get_config_from_plan, get_dataset_info_from_plan, create_dataset_from_plan


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUDA deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms in PyTorch (with warn_only for operations without deterministic CUDA implementation)
    # Some operations like cumsum don't have deterministic CUDA implementations
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Filter out the cumsum deterministic warning since we're using warn_only=True
    # This warning is expected and doesn't affect functionality
    # Filter both UserWarning category and the specific message
    warnings.filterwarnings('ignore', category=UserWarning, message='.*cumsum.*deterministic.*')
    
    return seed


def setup_logging(save_dir: str, model_type: str) -> logging.Logger:
    """Setup logging directly in checkpoint directory"""
    log_file = os.path.join(save_dir, f"{model_type}_training.log")
    
    # Create a unique logger name based on save_dir to avoid conflicts across folds
    logger_name = f"nnMIL_{os.path.basename(save_dir)}"
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')  # 'w' mode to overwrite if exists
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class BaseTrainer(ABC):
    """
    Base class for all MIL trainers.
    
    This class provides common functionality:
    - Plan file loading
    - Configuration management
    - Dataset loading
    - Model creation
    - Checkpoint management
    - Logging setup
    """
    
    def __init__(
        self,
        plan_path: str,
        model_type: str,
        fold: Optional[int] = None,
        save_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            plan_path: Path to dataset_plan.json
            model_type: Model architecture (e.g., 'simple_mil')
            fold: Cross-validation fold (0-4) or None for official_split
            save_dir: Directory to save checkpoints and logs
            **kwargs: Additional arguments that can override plan settings
        """
        self.plan_path = plan_path
        self.model_type = model_type
        self.fold = fold
        
        # Load plan file
        if not os.path.exists(plan_path):
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        self.plan = load_plan(plan_path)
        self.config = get_config_from_plan(plan_path)
        self.dataset_info = get_dataset_info_from_plan(plan_path)
        
        # Determine save directory
        if save_dir is None:
            # Auto-generate save directory (nnUNet style)
            dataset_name = os.path.basename(os.path.dirname(plan_path))
            save_dir = os.path.join(
                "nnMIL_results",
                dataset_name,
                model_type,
                f"fold_{fold}" if fold is not None else "official_split"
            )
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.save_dir, self.model_type)
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.logger.info(f"Plan file: {plan_path}")
        self.logger.info(f"Model type: {model_type}")
        self.logger.info(f"Fold: {fold}")
        self.logger.info(f"Save directory: {self.save_dir}")
        
        # Set random seed
        seed = kwargs.get('seed', self.plan.get('random_seed', 42))
        self.seed = set_random_seeds(seed)
        self.logger.info(f"Random seed: {self.seed}")
        
        # Determine evaluation setting and folds
        self.evaluation_setting = self.dataset_info.get('evaluation_setting', 'official_split')
        self.logger.info(f"Evaluation setting: {self.evaluation_setting}")
        
        # Override config with kwargs if provided
        self._override_config(kwargs)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model (to be created by subclasses)
        self.model = None
        
        # Initialize data loaders (to be created by subclasses)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # TensorBoard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)
            self.tensorboard_available = True
        except ImportError:
            self.writer = None
            self.tensorboard_available = False
            self.logger.warning("TensorBoard not available")
    
    def _override_config(self, kwargs: Dict[str, Any]):
        """Override configuration with command-line arguments"""
        for key, value in kwargs.items():
            if value is not None and key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                self.logger.info(f"Overriding {key}: {old_value} -> {value}")
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration with all overrides applied"""
        return self.config
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return self.dataset_info
    
    @abstractmethod
    def create_model(self):
        """Create the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def create_data_loaders(self):
        """Create data loaders. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train(self):
        """Run training. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def evaluate(self, split: str = 'test'):
        """Evaluate on a split. Must be implemented by subclasses.
        
        Args:
            split: 'train', 'val', or 'test'
        """
        pass
    
    def save_checkpoint(self, checkpoint_path: str, model_state: Dict, epoch: int, metrics: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'config': self.config,
            'dataset_info': self.dataset_info,
            'seed': self.seed,
        }
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

