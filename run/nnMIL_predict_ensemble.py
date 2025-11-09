#!/usr/bin/env python3
"""
nnMIL Ensemble Prediction - Predict using multiple fold models and average results

For 5-fold cross-validation, this script loads all 5 fold models and averages their predictions
for external test data.

Usage:
    # Ensemble prediction for classification
    python nnMIL/run/nnMIL_predict_ensemble.py \
        -t Dataset001_ebrains \
        -m simple_mil \
        -i /path/to/external/features \
        -o predictions/ensemble
    
    # Ensemble prediction for survival
    python nnMIL/run/nnMIL_predict_ensemble.py \
        -t Dataset002_tcga_brca \
        -m simple_mil \
        -i /path/to/external/features \
        -o predictions/ensemble \
        --task_type survival
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from nnMIL.inference import InferenceEngine
from nnMIL.utilities.plan_loader import (
    create_dataset_from_features_dir,
    find_plan_file,
    load_plan,
    get_config_from_plan,
    get_dataset_info_from_plan
)


def find_fold_checkpoints(task_id, model_type, results_base_dir="nnMIL_results"):
    """
    Find all fold checkpoints for a given task and model.
    
    Returns:
        List of checkpoint paths for folds 0-4
    """
    checkpoints = []
    dataset_name = task_id
    
    for fold in range(5):
        checkpoint_path = os.path.join(
            results_base_dir,
            dataset_name,
            model_type,
            f"fold_{fold}",
            f"best_{model_type}.pth"
        )
        
        if os.path.exists(checkpoint_path):
            checkpoints.append(checkpoint_path)
        else:
            print(f"⚠️  Warning: Checkpoint not found for fold {fold}: {checkpoint_path}")
    
    return checkpoints


def ensemble_predict_classification(engines, test_dataset, save_dir, logger=None, **kwargs):
    """
    Ensemble prediction for classification tasks.
    
    Args:
        engines: List of InferenceEngine instances (one per fold)
        test_dataset: Test dataset
        save_dir: Directory to save results
        logger: Optional logger
        **kwargs: Additional arguments
    
    Returns:
        Dictionary containing ensemble predictions and metrics
    """
    all_fold_probs = []
    all_fold_preds = []
    slide_ids = None
    true_labels = None
    
    print(f"\n{'='*60}")
    print(f"Running ensemble prediction with {len(engines)} models")
    print(f"{'='*60}\n")
    
    # Collect predictions from each fold
    for fold_idx, engine in enumerate(engines):
        print(f"Predicting with fold {fold_idx} model...")
        results = engine.predict(
            test_dataset=test_dataset,
            save_dir=None,  # Don't save individual fold results
            logger=logger,
            **kwargs
        )
        
        # Extract probabilities and predictions
        if 'probabilities' in results:
            all_fold_probs.append(results['probabilities'])
        if 'predictions' in results:
            all_fold_preds.append(results['predictions'])
        
        # Get slide IDs and true labels (same for all folds)
        if slide_ids is None and 'slide_ids' in results:
            slide_ids = results['slide_ids']
        if true_labels is None and 'true_labels' in results:
            true_labels = results['true_labels']
    
    # Average probabilities across folds
    avg_probs = np.mean(all_fold_probs, axis=0)  # Shape: [N, num_classes]
    
    # Get ensemble predictions from averaged probabilities
    if avg_probs.ndim == 2:
        # Multi-class: argmax
        ensemble_preds = np.argmax(avg_probs, axis=1)
    else:
        # Binary: threshold at 0.5
        ensemble_preds = (avg_probs > 0.5).astype(int)
    
    # Calculate metrics if true labels available
    metrics = {}
    if true_labels is not None:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
        
        metrics['accuracy'] = accuracy_score(true_labels, ensemble_preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(true_labels, ensemble_preds)
        
        try:
            if avg_probs.ndim == 2 and avg_probs.shape[1] > 2:
                # Multi-class AUC
                metrics['auroc'] = roc_auc_score(true_labels, avg_probs, multi_class='ovr', average='macro')
            else:
                # Binary AUC
                prob_pos = avg_probs[:, 1] if avg_probs.ndim == 2 else avg_probs
                metrics['auroc'] = roc_auc_score(true_labels, prob_pos)
        except:
            metrics['auroc'] = None
        
        print(f"\n{'='*60}")
        print(f"Ensemble Results:")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        if metrics['auroc'] is not None:
            print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"{'='*60}\n")
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create results DataFrame
        results_data = {
            'slide_id': slide_ids if slide_ids is not None else [f"sample_{i}" for i in range(len(ensemble_preds))],
            'prediction': ensemble_preds
        }
        
        # Add probabilities
        if avg_probs.ndim == 2:
            for i in range(avg_probs.shape[1]):
                results_data[f'probability_class_{i}'] = avg_probs[:, i]
        else:
            results_data['probability'] = avg_probs
        
        # Add true labels if available
        if true_labels is not None:
            results_data['true_label'] = true_labels
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = os.path.join(save_dir, 'ensemble_predictions.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"✅ Ensemble predictions saved to: {csv_path}")
        
        # Save metrics
        if metrics:
            metrics_path = os.path.join(save_dir, 'ensemble_metrics.txt')
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
            print(f"✅ Ensemble metrics saved to: {metrics_path}")
    
    return {
        'predictions': ensemble_preds,
        'probabilities': avg_probs,
        'metrics': metrics,
        'slide_ids': slide_ids,
        'true_labels': true_labels
    }


def ensemble_predict_survival(engines, test_dataset, save_dir, logger=None, **kwargs):
    """
    Ensemble prediction for survival tasks.
    
    Args:
        engines: List of InferenceEngine instances (one per fold)
        test_dataset: Test dataset
        save_dir: Directory to save results
        logger: Optional logger
        **kwargs: Additional arguments
    
    Returns:
        Dictionary containing ensemble predictions and metrics
    """
    all_fold_risk_scores = []
    patient_ids = None
    slide_ids = None
    true_status = None
    true_time = None
    
    print(f"\n{'='*60}")
    print(f"Running ensemble prediction with {len(engines)} models")
    print(f"{'='*60}\n")
    
    # Collect predictions from each fold
    for fold_idx, engine in enumerate(engines):
        print(f"Predicting with fold {fold_idx} model...")
        results = engine.predict(
            test_dataset=test_dataset,
            save_dir=None,  # Don't save individual fold results
            logger=logger,
            **kwargs
        )
        
        # Extract risk scores
        if 'risk_scores' in results:
            risk_scores = results['risk_scores']
            print(f"  Fold {fold_idx}: Got {len(risk_scores)} risk scores")
            all_fold_risk_scores.append(risk_scores)
        else:
            print(f"  Warning: No risk_scores in results for fold {fold_idx}")
            print(f"  Available keys: {results.keys()}")
        
        # Get patient IDs, slide IDs and true values (same for all folds)
        if patient_ids is None and 'patient_ids' in results:
            patient_ids = results['patient_ids']
        if slide_ids is None and 'slide_ids' in results:
            slide_ids = results['slide_ids']
        if true_status is None and 'status' in results:
            true_status = results['status']
        if true_time is None and 'time' in results:
            true_time = results['time']
    
    if len(all_fold_risk_scores) == 0:
        raise ValueError("No risk scores collected from any fold!")
    
    # Average risk scores across folds
    print(f"\nAveraging risk scores from {len(all_fold_risk_scores)} folds...")
    print(f"Shape of each fold's risk scores: {all_fold_risk_scores[0].shape if hasattr(all_fold_risk_scores[0], 'shape') else len(all_fold_risk_scores[0])}")
    avg_risk_scores = np.mean(all_fold_risk_scores, axis=0)
    print(f"Average risk scores shape: {avg_risk_scores.shape if hasattr(avg_risk_scores, 'shape') else len(avg_risk_scores)}")
    
    # Calculate C-index if true values available and there are events
    metrics = {}
    if true_status is not None and true_time is not None:
        # Check if we have any events (not all censored)
        has_events = true_status.sum() > 0
        
        if has_events:
            from nnMIL.training.losses.survival_loss import survival_c_index
            
            c_index = survival_c_index(
                torch.tensor(avg_risk_scores),
                torch.tensor(true_status),
                torch.tensor(true_time),
                patient_ids
            )
            metrics['c_index'] = c_index
            
            print(f"\n{'='*60}")
            print(f"Ensemble Results:")
            print(f"{'='*60}")
            print(f"C-index: {c_index:.4f}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"Ensemble Results:")
            print(f"{'='*60}")
            print(f"No events found (external test data without labels)")
            print(f"Total samples: {len(avg_risk_scores)}")
            print(f"{'='*60}\n")
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create results DataFrame - use slide_id (file name) as the main identifier
        results_data = {
            'slide_id': slide_ids if slide_ids is not None else [f"sample_{i}" for i in range(len(avg_risk_scores))],
            'risk_score': avg_risk_scores
        }
        
        # Add patient_id if different from slide_id
        if patient_ids is not None and patient_ids != slide_ids:
            results_data['patient_id'] = patient_ids
        
        # Add true values if available
        if true_status is not None:
            results_data['status'] = true_status
        if true_time is not None:
            results_data['time'] = true_time
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = os.path.join(save_dir, 'ensemble_predictions.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"✅ Ensemble predictions saved to: {csv_path}")
        
        # Save metrics
        if metrics:
            metrics_path = os.path.join(save_dir, 'ensemble_metrics.txt')
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
            print(f"✅ Ensemble metrics saved to: {metrics_path}")
    
    return {
        'risk_scores': avg_risk_scores,
        'metrics': metrics,
        'slide_ids': slide_ids,
        'patient_ids': patient_ids,
        'status': true_status,
        'time': true_time
    }


def main():
    parser = argparse.ArgumentParser(
        description='nnMIL ensemble prediction using multiple fold models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ensemble prediction for classification
    python nnMIL/run/nnMIL_predict_ensemble.py \\
        -t Dataset001_ebrains \\
        -m simple_mil \\
        -i /path/to/external/features \\
        -o predictions/ensemble
    
    # Ensemble prediction for survival
    python nnMIL/run/nnMIL_predict_ensemble.py \\
        -t Dataset002_tcga_brca \\
        -m simple_mil \\
        -i /path/to/external/features \\
        -o predictions/ensemble
        """
    )
    
    # Required arguments
    parser.add_argument('--task_id', '-t', type=str, required=True,
                       help='Task ID (e.g., Dataset001_ebrains)')
    parser.add_argument('--model_type', '-m', type=str, required=True,
                       help='Model type (e.g., simple_mil, ab_mil)')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='Input directory containing external test features (.h5 files)')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for ensemble predictions')
    
    # Optional arguments
    parser.add_argument('--results_dir', type=str, default='nnMIL_results',
                       help='Base directory containing trained models (default: nnMIL_results)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). If None, auto-detect.')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                       help='Comma-separated fold indices to use (default: 0,1,2,3,4)')
    
    args = parser.parse_args()
    
    # Parse folds
    fold_indices = [int(f) for f in args.folds.split(',')]
    
    # Find plan file
    try:
        plan_path = find_plan_file(args.task_id)
        print(f"📋 Found plan file: {plan_path}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Load plan to get task type
    plan = load_plan(plan_path)
    task_type = plan.get('task_type', 'classification')
    config = get_config_from_plan(plan_path)
    
    print(f"Task type: {task_type}")
    print(f"Model type: {args.model_type}")
    
    # Find all fold checkpoints
    checkpoints = []
    for fold in fold_indices:
        checkpoint_path = os.path.join(
            args.results_dir,
            args.task_id,
            args.model_type,
            f"fold_{fold}",
            f"best_{args.model_type}.pth"
        )
        
        if os.path.exists(checkpoint_path):
            checkpoints.append(checkpoint_path)
            print(f"✅ Found checkpoint for fold {fold}: {checkpoint_path}")
        else:
            print(f"⚠️  Warning: Checkpoint not found for fold {fold}: {checkpoint_path}")
    
    if len(checkpoints) == 0:
        print("❌ Error: No valid checkpoints found!")
        sys.exit(1)
    
    print(f"\nUsing {len(checkpoints)} fold models for ensemble prediction")
    
    # Create test dataset from input directory
    print(f"\n📁 Loading test data from: {args.input_dir}")
    test_dataset = create_dataset_from_features_dir(
        features_dir=args.input_dir,
        plan_path=plan_path
    )
    print(f"Loaded {len(test_dataset)} samples")
    
    # Initialize inference engines for each fold
    engines = []
    for checkpoint_path in checkpoints:
        engine = InferenceEngine(
            plan_path=plan_path,
            checkpoint_path=checkpoint_path,
            device=args.device
        )
        engines.append(engine)
    
    # Run ensemble prediction based on task type
    kwargs = {
        'model_type': args.model_type,
        'input_dim': config.get('feature_dimension', 2560),
        'hidden_dim': config.get('hidden_dim', 512),
        'dropout': config.get('dropout', 0.25),
        'batch_size': 1,
    }
    
    if task_type == 'survival':
        results = ensemble_predict_survival(
            engines=engines,
            test_dataset=test_dataset,
            save_dir=args.output_dir,
            **kwargs
        )
    else:  # classification or regression
        results = ensemble_predict_classification(
            engines=engines,
            test_dataset=test_dataset,
            save_dir=args.output_dir,
            **kwargs
        )
    
    print(f"\n✅ Ensemble prediction completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

