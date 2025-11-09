"""
Survival Predictor

Implements survival analysis inference logic directly.
"""

import os
from typing import Dict, Any, Optional
import logging

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from nnMIL.inference.predictors.base_predictor import BasePredictor
from nnMIL.data.dataset import random_length_collate_fn
from nnMIL.training.losses.survival_loss import survival_c_index


class SurvivalPredictor(BasePredictor):
    """Predictor for survival analysis tasks"""
    
    def predict(self, test_dataset, model, device, save_dir: Optional[str] = None,
                logger: Optional[logging.Logger] = None, **kwargs) -> Dict[str, Any]:
        """
        Run survival inference.
        """
        # Convert device string to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
        
        # Create data loader
        # For test/val, we use all original patches (variable length), batch_size MUST be 1
        batch_size = 1  # Force batch_size=1 for test/val (variable length sequences)
        
        if logger:
            logger.info(f"Test inference: Using batch_size=1 (variable-length sequences, using all original patches)")
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            collate_fn=random_length_collate_fn
        )
        
        model_type = kwargs.get('model_type', 'simple_mil')
        stride_divisor = kwargs.get('stride_divisor', None)
        dataset_name = kwargs.get('dataset_name', None)
        save_csv_path = None
        if save_dir:
            save_csv_path = os.path.join(save_dir, f"results_{model_type}.csv")
        
        # Initialize lists for collecting results
        preds_all = []
        status_all = []
        time_all = []
        patient_ids_all = []
        slide_ids = []
        datasets_all = []
        
        # Uncertainty metrics (for SimpleMIL)
        MI_all = []
        H_mean_all = []
        H_each_all = []
        var_logits_all = []
        
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        
        if logger:
            logger.info(f"Starting test inference")
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Test Inference")):
                # Handle different batch formats
                # Survival datasets return: features, coords, bag_sizes, status, time, patient_id, slide_id
                if len(batch) >= 6:
                    features, coords, bag_sizes, status, time = batch[:5]
                    # Extract patient_id and slide_id if available
                    if len(batch) >= 7:
                        patient_id = batch[5]
                        slide_id = batch[6] if len(batch) > 6 else patient_id
                    else:
                        # Old format without slide_id
                        patient_id = batch[5]
                        slide_id = patient_id
                    
                    # Handle string/list conversion
                    if isinstance(slide_id, str):
                        slide_id = [slide_id]
                    if isinstance(patient_id, str):
                        patient_id = [patient_id]
                else:
                    # Fallback format
                    features, coords, bag_sizes, status, time = batch
                    slide_id = [f"batch_{batch_idx}_sample_{i}" for i in range(len(status))]
                    patient_id = slide_id.copy()
                
                features = features.to(device)
                status = status.to(device)
                time = time.to(device)
                
                # Handle different model types
                if stride_divisor is not None and hasattr(model, 'forward'):
                    # SimpleMIL with stride_divisor parameter
                    if 'stride_divisor' in model.forward.__code__.co_varnames:
                        if 'is_cox' in model.forward.__code__.co_varnames:
                            output = model(features, is_cox=True, stride_divisor=stride_divisor)
                        else:
                            output = model(features, stride_divisor=stride_divisor)
                    elif 'is_cox' in model.forward.__code__.co_varnames:
                        output = model(features, is_cox=True)
                    else:
                        output = model(features)
                else:
                    # Other models
                    if hasattr(model, 'forward') and 'is_cox' in model.forward.__code__.co_varnames:
                        output = model(features, is_cox=True)
                    else:
                        output = model(features)
                
                if isinstance(output, dict):
                    logits = output['logits']
                    # Extract uncertainty metrics if available (SimpleMIL)
                    if 'MI' in output and output['MI'] is not None:
                        MI_all.append(output['MI'].float().cpu().numpy())
                    if 'H_mean' in output and output['H_mean'] is not None:
                        H_mean_all.append(output['H_mean'].float().cpu().numpy())
                    if 'H_each' in output and output['H_each'] is not None:
                        h_each = output['H_each'].float().cpu().numpy()
                        if h_each.ndim == 3 and h_each.shape[-1] == 1:
                            h_each = h_each.squeeze(-1)
                        H_each_all.append(h_each)
                    if 'var_logits' in output and output['var_logits'] is not None:
                        var_logits_all.append(output['var_logits'].float().cpu().numpy())
                else:
                    logits = output
                
                logits = logits.float()
                
                # For NLLSurv models: logits shape is [batch_size, nll_bins]
                # We need to convert to risk scores before appending
                # Check if this is NLLSurv output (logits has 2D shape with >1 column)
                if logits.dim() == 2 and logits.shape[1] > 1:
                    # NLLSurv: convert discrete hazards to risk score
                    # Like in survival_porpoise_trainer.py: lines 348-352
                    hazards = torch.sigmoid(logits)  # [batch_size, nll_bins]
                    survival = torch.cumprod(1 - hazards, dim=1)  # [batch_size, nll_bins]
                    # Risk score = negative expected survival time (area under survival curve)
                    risk_scores = -survival.sum(dim=1, keepdim=True)  # [batch_size, 1]
                    preds_all.append(risk_scores.cpu().numpy())
                else:
                    # Cox loss: logits is already risk score [batch_size, 1] or [batch_size]
                    preds_all.append(logits.cpu().numpy())
                
                status_all.append(status.cpu().numpy())
                time_all.append(time.cpu().numpy())
                patient_ids_all.extend(patient_id)
                slide_ids.extend(slide_id)
                
                # Get dataset information for DSS/PFS datasets
                if hasattr(test_dataset, 'is_dss_dataset') and test_dataset.is_dss_dataset:
                    batch_start_idx = batch_idx * test_loader.batch_size
                    batch_end_idx = min(batch_start_idx + test_loader.batch_size, len(test_dataset.df))
                    batch_datasets = test_dataset.df.iloc[batch_start_idx:batch_end_idx]['dataset'].tolist()
                    datasets_all.extend(batch_datasets)
                else:
                    datasets_all.extend([dataset_name] * len(patient_id))
        
        # Concatenate all results
        preds_all = np.concatenate(preds_all)
        status_all = np.concatenate(status_all)
        time_all = np.concatenate(time_all)
        
        # Ensure 1D arrays
        if preds_all.ndim > 1:
            preds_all = preds_all.flatten()
        if status_all.ndim > 1:
            status_all = status_all.flatten()
        if time_all.ndim > 1:
            time_all = time_all.flatten()
        
        # Handle unsqueeze for scalar status/time (when batch_size=1)
        status_all = np.atleast_1d(status_all)
        time_all = np.atleast_1d(time_all)
        
        # Concatenate uncertainty metrics if available
        has_uncertainty = len(MI_all) > 0 or len(H_each_all) > 0 or len(var_logits_all) > 0
        if has_uncertainty:
            if len(MI_all) > 0:
                MI_all = np.concatenate(MI_all).flatten()
                H_mean_all = np.concatenate(H_mean_all).flatten()
            
            # H_each: handle [K, B] format
            if len(H_each_all) > 0:
                H_each_concat = []
                for h_batch in H_each_all:
                    # h_batch is [K, B], transpose to [B, K]
                    if h_batch.ndim == 2:
                        h_batch_transposed = h_batch.T
                        for b in range(h_batch_transposed.shape[0]):
                            H_each_concat.append(h_batch_transposed[b])
                    else:
                        # Already [B, K] or [K]
                        if h_batch.ndim == 1:
                            H_each_concat.append(h_batch)
                        else:
                            H_each_concat.extend([h_batch[i] for i in range(h_batch.shape[0])])
                H_each_all = np.array(H_each_concat) if H_each_concat else np.array([])
            
            if len(var_logits_all) > 0:
                var_logits_all = np.concatenate(var_logits_all).flatten()
            
            if logger:
                n_samples = len(MI_all) if len(MI_all) > 0 else len(H_each_all) if len(H_each_all) > 0 else len(var_logits_all)
                logger.info(f"Uncertainty metrics collected for {n_samples} samples")
        
        if logger:
            logger.info("Computing patient-level metrics...")
        
        # Check if we have any events (not all censored)
        has_events = status_all.sum() > 0
        
        # Calculate C-index at patient level (overall) only if we have events
        if has_events:
            c_index = survival_c_index(
                torch.from_numpy(preds_all), 
                torch.from_numpy(status_all), 
                torch.from_numpy(time_all),
                patient_ids=patient_ids_all
            )
            
            metrics = {
                "overall": {
                    "Test_Overall_C-index": float(c_index),
                    "Test_Overall_Events": int(status_all.sum()),
                    "Test_Overall_Censored": int((1 - status_all).sum()),
                    "Test_Overall_Event_Rate": float(status_all.mean()),
                    "Test_Overall_Mean_Survival_Time": float(time_all.mean()),
                    "Test_Overall_Median_Survival_Time": float(np.median(time_all))
                }
            }
            
            # Log overall results
            if logger:
                logger.info("Test Overall Results:")
                logger.info(f"  Overall C-index: {c_index:.4f}")
                logger.info(f"  Overall Events: {int(status_all.sum())}")
                logger.info(f"  Overall Censored: {int((1 - status_all).sum())}")
                logger.info(f"  Overall Event Rate: {status_all.mean():.4f}")
                logger.info(f"  Overall Mean Survival Time: {time_all.mean():.2f}")
                logger.info(f"  Overall Median Survival Time: {np.median(time_all):.2f}")
        else:
            # No events - external test data without labels
            metrics = {
                "overall": {
                    "Test_Overall_Events": int(status_all.sum()),
                    "Test_Overall_Censored": int((1 - status_all).sum()),
                }
            }
            
            if logger:
                logger.info("Test Overall Results:")
                logger.info("  No events found (external test data without labels)")
                logger.info(f"  Total samples: {len(status_all)}")
        
        # Calculate metrics by dataset for DSS/PFS datasets
        if hasattr(test_dataset, 'is_dss_dataset') and test_dataset.is_dss_dataset:
            unique_datasets = sorted(set(datasets_all))
            if logger:
                logger.info("Test Results by Dataset:")
            
            metrics["by_dataset"] = {}
            
            for ds in unique_datasets:
                dataset_mask = np.array([d == ds for d in datasets_all])
                if np.sum(dataset_mask) > 0:
                    dataset_preds = preds_all[dataset_mask]
                    dataset_status = status_all[dataset_mask]
                    dataset_time = time_all[dataset_mask]
                    dataset_patient_ids = [patient_ids_all[i] for i in range(len(patient_ids_all)) if dataset_mask[i]]
                    
                    # Check if this dataset has any events
                    dataset_has_events = dataset_status.sum() > 0
                    
                    if dataset_has_events:
                        # Calculate C-index for this dataset
                        dataset_c_index = survival_c_index(
                            torch.from_numpy(dataset_preds),
                            torch.from_numpy(dataset_status),
                            torch.from_numpy(dataset_time),
                            patient_ids=dataset_patient_ids
                        )
                        
                        # Store dataset-specific metrics
                        metrics["by_dataset"][ds] = {
                            f"Test_{ds}_C-index": float(dataset_c_index),
                            f"Test_{ds}_Events": int(dataset_status.sum()),
                            f"Test_{ds}_Censored": int((1 - dataset_status).sum()),
                            f"Test_{ds}_Event_Rate": float(dataset_status.mean()),
                            f"Test_{ds}_Mean_Survival_Time": float(dataset_time.mean()),
                            f"Test_{ds}_Median_Survival_Time": float(np.median(dataset_time))
                        }
                        
                        if logger:
                            logger.info(f"  {ds}:")
                            logger.info(f"    C-index: {dataset_c_index:.4f}")
                            logger.info(f"    Events: {int(dataset_status.sum())}")
                            logger.info(f"    Censored: {int((1 - dataset_status).sum())}")
                    else:
                        # No events in this dataset
                        metrics["by_dataset"][ds] = {
                            f"Test_{ds}_Events": int(dataset_status.sum()),
                            f"Test_{ds}_Censored": int((1 - dataset_status).sum()),
                        }
                        
                        if logger:
                            logger.info(f"  {ds}:")
                            logger.info(f"    No events (external test data)")
                            logger.info(f"    Total samples: {len(dataset_status)}")
        
        # Create results DataFrame
        results_dict = {
            'slide_id': slide_ids,
            'patient_id': patient_ids_all,
            'dataset': datasets_all,
            'status': status_all,
            'time': time_all,
            'risk_score': preds_all.flatten()
        }
        
        # Add uncertainty metrics if available
        if has_uncertainty:
            if len(MI_all) > 0:
                results_dict['MI'] = MI_all
                results_dict['H_mean'] = H_mean_all
            
            if len(H_each_all) > 0:
                # H_each is [N, K], save each chunk as separate columns
                # Or save as comma-separated string if K varies
                if H_each_all.ndim == 2:
                    for k in range(H_each_all.shape[1]):
                        results_dict[f'H_chunk_{k}'] = H_each_all[:, k]
                else:
                    # Convert to string representation if shape is inconsistent
                    results_dict['H_each'] = [str(h) for h in H_each_all]
            
            if len(var_logits_all) > 0:
                results_dict['var_logits'] = var_logits_all
        
        results_df = pd.DataFrame(results_dict)
        
        # Save results to CSV
        if save_csv_path:
            results_df.to_csv(save_csv_path, index=False)
            if logger:
                logger.info(f"Results saved to {save_csv_path}")
        
        # Return both metrics and predictions
        return {
            'metrics': metrics,
            'risk_scores': preds_all,
            'patient_ids': patient_ids_all,
            'slide_ids': slide_ids,
            'status': status_all,
            'time': time_all,
            'datasets': datasets_all
        }
