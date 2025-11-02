"""
Base predictor class for all inference tasks.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class BasePredictor(ABC):
    """Base class for all predictors"""
    
    @abstractmethod
    def predict(self, test_dataset, model: torch.nn.Module, device: torch.device,
                save_dir: Optional[str] = None, logger: Optional[logging.Logger] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Run inference on test dataset.
        
        Args:
            test_dataset: Test dataset
            model: Trained model
            device: Device to run inference on
            save_dir: Directory to save results
            logger: Logger instance
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing metrics and results
        """
        pass

