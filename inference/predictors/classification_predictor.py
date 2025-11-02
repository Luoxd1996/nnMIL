"""
Classification Predictor

Implements classification inference logic directly.
"""

import os
from typing import Dict, Any, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

from nnMIL.inference.predictors.base_predictor import BasePredictor
from nnMIL.data.dataset import random_length_collate_fn
from nnMIL.utilities.utils import get_eval_metrics


def create_mask_from_bag_sizes(bags, bag_sizes):
    """Create mask tensor from bag sizes for VisionTransformer"""
    max_possible_bag_size = bags.size(1)
    mask = torch.arange(max_possible_bag_size).type_as(bag_sizes).unsqueeze(0).repeat(
        len(bags), 1
    ) >= bag_sizes.unsqueeze(1)
    return mask


def aggregate_by_patient(df, task_type='classification'):
    """Aggregate results by patient for classification tasks"""
    if 'patient_id' not in df.columns:
        return df
    
    grouped = df.groupby('patient_id')
    
    def avg_probabilities(probs):
        # Convert to numpy arrays and average
        if isinstance(probs.iloc[0], np.ndarray):
            # Multi-class: stack arrays and mean
            return np.mean(np.stack(probs.values), axis=0)
        else:
            # Binary: directly average scalar
            return probs.mean()
    
    out = pd.DataFrame({
        'true_label': grouped['true_label'].first(),
        'probability': grouped.apply(lambda x: avg_probabilities(x['probability'])),
        'predicted_label': grouped['predicted_label'].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
    }).reset_index()
    
    # If dataset column exists, keep the first one
    if 'dataset' in df.columns:
        out['dataset'] = grouped['dataset'].first()
    
    return out


class ClassificationPredictor(BasePredictor):
    """Predictor for classification tasks"""
    
    def predict(self, test_dataset, model, device, save_dir: Optional[str] = None,
                logger: Optional[logging.Logger] = None, **kwargs) -> Dict[str, Any]:
        """
        Run classification inference.
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
        
        num_classes = test_dataset.num_classes
        model_type = kwargs.get('model_type', 'simple_mil')
        aggregate_patient_level = kwargs.get('aggregate_patient_level', True)
        save_csv_path = None
        if save_dir:
            save_csv_path = os.path.join(save_dir, f"results_{model_type}.csv")
        
        # Initialize lists for collecting results
        preds_all = []
        probs_all = []
        targets_all = []
        slide_ids = []
        patient_ids = []
        datasets = []
        
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
                if len(batch) >= 6:
                    features, coords, bag_sizes, label, slide_id, dataset_name = batch
                    slide_id = [slide_id] if isinstance(slide_id, str) else slide_id
                    dataset_name = [dataset_name] if isinstance(dataset_name, str) else dataset_name
                    # Extract patient_id from dataset if available
                    patient_id = []
                    for sid in slide_id:
                        if hasattr(test_dataset, 'df') and hasattr(test_dataset, 'slide_col'):
                            try:
                                idx = test_dataset.df[test_dataset.slide_col] == sid
                                if idx.any():
                                    pid = test_dataset.df.loc[idx, 'patient_id'].iloc[0]
                                    patient_id.append(pid)
                                else:
                                    patient_id.append(sid)
                            except:
                                patient_id.append(sid)
                        else:
                            patient_id.append(sid)
                else:
                    features, coords, bag_sizes, label = batch
                    slide_id = [f"batch_{batch_idx}_sample_{i}" for i in range(len(label))]
                    dataset_name = ["UNKNOWN"] * len(label)
                    patient_id = slide_id.copy()
                
                features = features.to(device)
                coords = coords.to(device)
                bag_sizes = bag_sizes.to(device)
                label = label.to(device)
                
                # Handle different model types
                if model_type == 'vision_transformer':
                    mask = create_mask_from_bag_sizes(features, bag_sizes)
                    output = model(features, coords=coords, mask=mask)
                elif model_type == 'simple_mil':
                    stride_divisor = kwargs.get('stride_divisor', 4)
                    output = model(features, stride_divisor=stride_divisor)
                else:
                    output = model(features)
                
                logits = output['logits'] if isinstance(output, dict) else output
                logits = logits.float()
                preds = logits.argmax(1)
                
                if num_classes == 2:
                    probs = F.softmax(logits, dim=1)[:, 1]
                    roc_kwargs = {}
                else:
                    probs = F.softmax(logits, dim=1)
                    roc_kwargs = {"multi_class": "ovr", "average": "macro"}
                
                preds_all.append(preds.cpu().numpy())
                probs_all.append(probs.cpu().numpy())
                targets_all.append(label.cpu().numpy())
                
                # Extract uncertainty metrics if available (SimpleMIL)
                if isinstance(output, dict):
                    if 'MI' in output and output['MI'] is not None:
                        MI_all.append(output['MI'].float().cpu().numpy())
                    if 'H_mean' in output and output['H_mean'] is not None:
                        H_mean_all.append(output['H_mean'].float().cpu().numpy())
                    if 'H_each' in output and output['H_each'] is not None:
                        # H_each is [K, B], we want to transpose and save as [B, K]
                        h_each = output['H_each'].float().cpu().numpy()
                        H_each_all.append(h_each)
                    if 'var_logits' in output and output['var_logits'] is not None:
                        var_logits_all.append(output['var_logits'].float().cpu().numpy())
                
                # Collect IDs
                for i in range(len(label)):
                    slide_ids.append(str(slide_id[i]))
                    patient_ids.append(str(patient_id[i]))
                    datasets.append(str(dataset_name[i]))
        
        # Concatenate all results
        preds_all = np.concatenate(preds_all)
        probs_all = np.concatenate(probs_all)
        targets_all = np.concatenate(targets_all)
        
        # Concatenate uncertainty metrics if available
        has_uncertainty = len(MI_all) > 0 or len(H_each_all) > 0 or len(var_logits_all) > 0
        if has_uncertainty:
            if len(MI_all) > 0:
                MI_all = np.concatenate(MI_all)
                H_mean_all = np.concatenate(H_mean_all)
            
            # H_each: handle [K, B] format
            if len(H_each_all) > 0:
                H_each_concat = []
                for h_batch in H_each_all:
                    # h_batch is [K, B], transpose to [B, K]
                    h_batch_transposed = h_batch.T
                    for b in range(h_batch_transposed.shape[0]):
                        H_each_concat.append(h_batch_transposed[b])
                H_each_all = np.array(H_each_concat)  # [total_samples, K]
            
            if len(var_logits_all) > 0:
                var_logits_all = np.concatenate(var_logits_all)
            
            if logger:
                n_samples = len(MI_all) if len(MI_all) > 0 else len(H_each_all) if len(H_each_all) > 0 else len(var_logits_all)
                logger.info(f"Uncertainty metrics collected for {n_samples} samples")
        
        # Calculate slide-level metrics
        eval_metrics = get_eval_metrics(
            targets_all=targets_all,
            preds_all=preds_all,
            probs_all=probs_all,
            unique_classes=list(range(num_classes)),
            prefix='test'
        )
        
        # Add Weighted Kappa for PANDA dataset
        dataset_name_arg = kwargs.get('dataset_name', None)
        if dataset_name_arg == "panda":
            kappa = cohen_kappa_score(targets_all, preds_all, weights='quadratic')
            eval_metrics['test_kappa'] = kappa
        
        # Patient-level aggregation if requested
        patient_metrics = None
        if aggregate_patient_level and 'patient_id' in patient_ids:
            if logger:
                logger.info("Computing patient-level metrics...")
            
            # Create DataFrame for patient-level aggregation
            results_df = pd.DataFrame({
                'patient_id': patient_ids,
                'slide_id': slide_ids,
                'dataset': datasets,
                'true_label': targets_all,
                'predicted_label': preds_all,
                'probability': list(probs_all)  # Keep as list for aggregation
            })
            
            # Aggregate by patient
            patient_df = aggregate_by_patient(results_df, task_type='classification')
            
            if len(patient_df) > 0:
                # Calculate patient-level metrics
                patient_preds = patient_df['predicted_label'].values
                patient_targets = patient_df['true_label'].values
                patient_probs = patient_df['probability'].values
                
                # Convert patient_probs to proper format
                if len(patient_probs) > 0 and isinstance(patient_probs[0], np.ndarray):
                    patient_probs = np.stack(patient_probs)
                
                patient_metrics = get_eval_metrics(
                    targets_all=patient_targets,
                    preds_all=patient_preds,
                    probs_all=patient_probs,
                    unique_classes=list(range(num_classes)),
                    prefix='test_patient'
                )
                
                # Add Weighted Kappa for PANDA dataset
                if dataset_name_arg == "panda":
                    kappa = cohen_kappa_score(patient_targets, patient_preds, weights='quadratic')
                    patient_metrics['test_patient_kappa'] = kappa
        
        # Log results
        if logger:
            logger.info("Test Slide-level Results:")
            for metric, value in eval_metrics.items():
                logger.info(f"  {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
            
            if patient_metrics:
                logger.info("Test Patient-level Results:")
                for metric, value in patient_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
        
        # Save results to CSV
        if save_csv_path:
            results_dict = {
                'slide_id': slide_ids,
                'patient_id': patient_ids,
                'dataset': datasets,
                'label': targets_all,
                'prediction': preds_all,
            }
            
            # Add probability columns
            if num_classes == 2:
                results_dict['probability'] = probs_all
            else:
                for class_idx in range(num_classes):
                    results_dict[f'probability_class_{class_idx}'] = probs_all[:, class_idx]
            
            # Add uncertainty metrics if available
            if has_uncertainty:
                if len(MI_all) > 0:
                    results_dict['MI'] = MI_all
                    results_dict['H_mean'] = H_mean_all
                else:
                    if logger:
                        logger.info("Note: MI and H_mean not available")
                
                if len(H_each_all) > 0:
                    # Save H_each as string representation of array [K]
                    results_dict['H_each'] = [str(h.tolist()) for h in H_each_all]
                
                # Var_logits: handle multi-dimensional case
                if len(var_logits_all) > 0:
                    if var_logits_all.ndim == 1:
                        results_dict['var_logits'] = var_logits_all
                    else:
                        # Multi-dimensional [B, num_classes] or [B, pred_num]
                        if num_classes == 2 and var_logits_all.shape[1] == 2:
                            # Binary: use class 1 variance
                            results_dict['var_logits'] = var_logits_all[:, 1]
                        elif var_logits_all.shape[1] == num_classes:
                            # Multi-class: save as string for all class variances
                            results_dict['var_logits'] = [str(v) for v in var_logits_all]
                        else:
                            results_dict['var_logits'] = var_logits_all.flatten()
                
                if logger:
                    if len(MI_all) > 0:
                        logger.info(f"  MI range: [{MI_all.min():.4f}, {MI_all.max():.4f}]")
                        logger.info(f"  H_mean range: [{H_mean_all.min():.4f}, {H_mean_all.max():.4f}]")
                    if len(var_logits_all) > 0:
                        logger.info(f"  Risk variance range: [{var_logits_all.min():.6f}, {var_logits_all.max():.6f}]")
            
            results_df = pd.DataFrame(results_dict)
            results_df.to_csv(save_csv_path, index=False)
            if logger:
                logger.info(f"Results saved to {save_csv_path}")
                if has_uncertainty:
                    logger.info("  - Saved uncertainty metrics: MI, H_mean, H_each, var_logits")
        
        # Return metrics
        metrics_dict = {
            'slide_level': eval_metrics,
            'patient_level': patient_metrics
        }
        
        return metrics_dict
