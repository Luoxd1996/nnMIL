"""
Classification Predictor

Wraps the existing inference_mil.py logic into a class-based interface.
"""

import os
import sys
from typing import Dict, Any, Optional
import logging

# Import existing inference function
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nnMIL.inference.predictors.base_predictor import BasePredictor
# Import the existing inference function as a module function
# We'll import it dynamically or wrap the existing code


class ClassificationPredictor(BasePredictor):
    """Predictor for classification tasks"""
    
    def predict(self, test_dataset, model, device, save_dir: Optional[str] = None,
                logger: Optional[logging.Logger] = None, **kwargs) -> Dict[str, Any]:
        """
        Run classification inference.
        
        This wraps the existing inference function from inference_mil.py
        """
        import torch
        
        # Import the existing inference function
        from nnMIL.inference.inference_mil import inference
        
        from torch.utils.data import DataLoader
        from functools import partial
        from nnMIL.data.dataset import random_length_collate_fn
        
        # Convert device string to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
        
        # Create data loader
        # For test/val, we use all original patches (variable length), batch_size MUST be 1
        # because each sample has different number of patches
        batch_size = 1  # Force batch_size=1 for test/val (variable length sequences)
        
        if logger:
            logger.info(f"Test inference: Using batch_size=1 (variable-length sequences, using all original patches)")
        
        # Use random_length_collate_fn for variable-length sequences (test/val always use original length)
        from nnMIL.data.dataset import random_length_collate_fn
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            collate_fn=random_length_collate_fn
        )
        
        # Run inference using existing function
        save_csv_path = None
        if save_dir:
            model_type = kwargs.get('model_type', 'simple_mil')
            save_csv_path = os.path.join(save_dir, f"inference_results_{model_type}.csv")
        
        metrics = inference(
            test_loader=test_loader,
            model=model,
            num_classes=test_dataset.num_classes,
            device=device,
            prefix='test',
            save_csv_path=save_csv_path,
            logger=logger,
            aggregate_patient_level=kwargs.get('aggregate_patient_level', True),
            test_dataset=test_dataset,
            **kwargs
        )
        
        return metrics

