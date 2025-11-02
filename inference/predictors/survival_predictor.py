"""
Survival Predictor

Wraps the existing inference_surv.py logic into a class-based interface.
"""

import os
import sys
from typing import Dict, Any, Optional
import logging

from nnMIL.inference.predictors.base_predictor import BasePredictor


class SurvivalPredictor(BasePredictor):
    """Predictor for survival analysis tasks"""
    
    def predict(self, test_dataset, model, device, save_dir: Optional[str] = None,
                logger: Optional[logging.Logger] = None, **kwargs) -> Dict[str, Any]:
        """
        Run survival inference.
        
        This wraps the existing inference_survival function from inference_surv.py
        """
        # Import the existing inference function
        from nnMIL.inference.inference_surv import inference_survival
        
        from torch.utils.data import DataLoader
        
        # Create data loader
        batch_size = kwargs.get('batch_size', 1)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Run inference using existing function
        metrics, results_df = inference_survival(
            test_loader=test_loader,
            model=model,
            device=device,
            logger=logger,
            stride_divisor=kwargs.get('stride_divisor', None),
            test_dataset=test_dataset,
            dataset_name=kwargs.get('dataset_name', None),
            **kwargs
        )
        
        # Save results if save_dir provided
        if save_dir and results_df is not None:
            model_type = kwargs.get('model_type', 'simple_mil')
            save_csv_path = os.path.join(save_dir, f"inference_results_{model_type}.csv")
            results_df.to_csv(save_csv_path, index=False)
            if logger:
                logger.info(f"Results saved to {save_csv_path}")
        
        return metrics

