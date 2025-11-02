"""
Generate dataset.json file for nnUNet-style dataset management

This script automatically generates dataset.json files from existing datasets
by analyzing the CSV files and feature directories.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def analyze_dataset_statistics(csv_path: str, task_type: str, dataset_name: str) -> Dict[str, Any]:
    """Analyze dataset and return statistics"""
    df = pd.read_csv(csv_path)
    
    stats = {}
    
    if task_type == 'classification':
        # Determine label column
        label_col = _get_label_column(df, dataset_name)
        if label_col and label_col in df.columns:
            stats['num_classes'] = int(df[label_col].nunique())
            stats['class_distribution'] = df[label_col].value_counts().to_dict()
            
            # Split statistics if split column exists
            if 'split' in df.columns:
                for split in ['train', 'val', 'test']:
                    split_df = df[df['split'] == split]
                    if len(split_df) > 0:
                        stats[f'num_{split}_samples'] = int(len(split_df))
    
    elif task_type == 'survival':
        if 'status' in df.columns and 'time' in df.columns:
            stats['event_rate'] = float(df['status'].mean())
            stats['median_survival_time'] = float(df['time'].median())
            stats['mean_survival_time'] = float(df['time'].mean())
            
            # Split statistics
            if 'split' in df.columns:
                for split in ['train', 'val', 'test']:
                    split_df = df[df['split'] == split]
                    if len(split_df) > 0:
                        stats[f'num_{split}_samples'] = int(len(split_df))
            elif any(col.startswith('fold_') for col in df.columns):
                # Cross-validation format
                stats['num_folds'] = len([col for col in df.columns if col.startswith('fold_')])
                stats['num_total_samples'] = int(len(df))
    
    elif task_type == 'regression':
        # Determine target column
        target_col = _get_target_column(df, dataset_name)
        if target_col and target_col in df.columns:
            target_data = pd.to_numeric(df[target_col], errors='coerce').dropna()
            stats['target_statistics'] = {
                'mean': float(target_data.mean()),
                'std': float(target_data.std()),
                'min': float(target_data.min()),
                'max': float(target_data.max())
            }
            
            if 'split' in df.columns:
                for split in ['train', 'val', 'test']:
                    split_df = df[df['split'] == split]
                    if len(split_df) > 0:
                        stats[f'num_{split}_samples'] = int(len(split_df))
    
    return stats


def _get_label_column(df: pd.DataFrame, dataset_name: str) -> Optional[str]:
    """Determine label column based on dataset name"""
    if dataset_name == "ebrains":
        return 'strat_label'
    elif dataset_name == "ebrains_coarse":
        return 'diagnosis_group'
    elif dataset_name in ["bracs", "bracs_coarse"]:
        return 'WSI label'
    elif dataset_name == "bccc_5class":
        return '5_class_number'
    elif dataset_name == "bccc_3class":
        return '3_class_number'
    elif dataset_name == "bccc_2class":
        return '2_class_number'
    elif 'label' in df.columns:
        return 'label'
    elif 'target' in df.columns:
        return 'target'
    return None


def _get_target_column(df: pd.DataFrame, dataset_name: str) -> Optional[str]:
    """Determine target column for regression"""
    if 'target' in df.columns:
        return 'target'
    elif 'label' in df.columns:
        return 'label'
    return None


def generate_dataset_json(
    dataset_id: str,
    dataset_name: str,
    task_type: str,
    csv_path: str,
    features_dir: str,
    foundation_model: str,
    feature_dimension: int,
    description: Optional[str] = None,
    reference: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate dataset.json file
    
    Args:
        dataset_id: Dataset ID (e.g., "001", "002")
        dataset_name: Dataset name (e.g., "ebrains", "tcga_brca")
        task_type: Task type ("classification", "regression", "survival")
        csv_path: Path to CSV file
        features_dir: Path to features directory
        foundation_model: Foundation model name (e.g., "uni", "virchow2")
        feature_dimension: Feature dimension
        description: Dataset description
        reference: Reference/paper citation
        output_path: Path to save JSON file
    
    Returns:
        Dictionary containing dataset.json structure
    """
    # Analyze dataset
    stats = analyze_dataset_statistics(csv_path, task_type, dataset_name)
    
    # Get labels for classification tasks
    labels = None
    if task_type == 'classification':
        df = pd.read_csv(csv_path)
        label_col = _get_label_column(df, dataset_name)
        if label_col and label_col in df.columns:
            unique_labels = sorted(df[label_col].unique())
            labels = {str(label): f"Class_{label}" for label in unique_labels}
    
    # Build dataset.json structure
    dataset_json = {
        "name": dataset_name.replace('_', ' ').title(),
        "description": description or f"{task_type.title()} task for {dataset_name}",
        "reference": reference or "None",
        "licence": "None",
        "release": "1.0",
        
        "task_type": task_type,
        "task_name": dataset_name,
        "dataset_id": dataset_id,
        
        "modality": {
            "0": "Histopathology"
        },
        
        "data_format": {
            "features_format": "h5",
            "features_key": "features",
            "coords_key": "coords",
            "csv_columns": _get_csv_columns(csv_path, dataset_name, task_type),
            "file_ending": ".h5"
        },
        
        "dataset_statistics": {
            **stats,
            "feature_dimension": feature_dimension,
            "foundation_model": foundation_model
        }
    }
    
    # Add task-specific fields
    if task_type == 'classification':
        dataset_json["labels"] = labels or {}
        dataset_json["dataset_statistics"]["num_classes"] = stats.get('num_classes', len(labels) if labels else 0)
    
    elif task_type == 'survival':
        dataset_json["survival_targets"] = {
            "status": "Event status (0=censored, 1=event)",
            "time": "Survival time in days"
        }
        if 'fold_0' in pd.read_csv(csv_path).columns:
            dataset_json["data_format"]["cross_validation"] = {
                "num_folds": stats.get('num_folds', 5),
                "fold_format": "fold_0, fold_1, ..."
            }
    
    elif task_type == 'regression':
        dataset_json["regression_targets"] = {
            "target": "Regression target (continuous value)"
        }
    
    # Save if output_path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset_json, f, indent=4)
        print(f"Dataset JSON saved to: {output_path}")
    
    return dataset_json


def _get_csv_columns(csv_path: str, dataset_name: str, task_type: str) -> Dict[str, Any]:
    """Determine CSV column mappings"""
    df = pd.read_csv(csv_path)
    
    columns = {}
    
    if task_type == 'classification':
        columns['slide_id_col'] = _get_slide_id_column(df, dataset_name)
        columns['label_col'] = _get_label_column(df, dataset_name)
        if 'split' in df.columns:
            columns['split_col'] = 'split'
    
    elif task_type == 'survival':
        columns['slide_id_col'] = 'slide_id'
        columns['status_col'] = 'status'
        columns['time_col'] = 'time'
        if 'patient_id' in df.columns:
            columns['patient_id_col'] = 'patient_id'
        if any(col.startswith('fold_') for col in df.columns):
            columns['fold_col_prefix'] = 'fold_'
    
    elif task_type == 'regression':
        columns['slide_id_col'] = 'slide_id'
        columns['target_col'] = 'target' if 'target' in df.columns else 'label'
        if 'split' in df.columns:
            columns['split_col'] = 'split'
    
    return columns


def _get_slide_id_column(df: pd.DataFrame, dataset_name: str) -> str:
    """Determine slide ID column"""
    if dataset_name in ["bracs", "bracs_coarse"]:
        return 'WSI Filename'
    elif 'slide_id' in df.columns:
        return 'slide_id'
    elif 'Slide' in df.columns:
        return 'Slide'
    return 'slide_id'


def main():
    parser = argparse.ArgumentParser(description='Generate dataset.json file')
    parser.add_argument('-d', '--dataset_id', type=str, required=True,
                       help='Dataset ID (e.g., 001, 002)')
    parser.add_argument('-n', '--dataset_name', type=str, required=True,
                       help='Dataset name (e.g., ebrains, tcga_brca)')
    parser.add_argument('-t', '--task_type', type=str, required=True,
                       choices=['classification', 'regression', 'survival'],
                       help='Task type')
    parser.add_argument('-c', '--csv_path', type=str, required=True,
                       help='Path to CSV file')
    parser.add_argument('-f', '--features_dir', type=str, required=True,
                       help='Path to features directory')
    parser.add_argument('-m', '--foundation_model', type=str, required=True,
                       help='Foundation model name (e.g., uni, virchow2)')
    parser.add_argument('--feature_dim', type=int, required=True,
                       help='Feature dimension')
    parser.add_argument('--description', type=str, default=None,
                       help='Dataset description')
    parser.add_argument('--reference', type=str, default=None,
                       help='Reference/paper citation')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output path for dataset.json')
    
    args = parser.parse_args()
    
    dataset_json = generate_dataset_json(
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        task_type=args.task_type,
        csv_path=args.csv_path,
        features_dir=args.features_dir,
        foundation_model=args.foundation_model,
        feature_dimension=args.feature_dim,
        description=args.description,
        reference=args.reference,
        output_path=args.output or f"dataset{args.dataset_id}_dataset.json"
    )
    
    print("\nGenerated dataset.json:")
    print(json.dumps(dataset_json, indent=2))


if __name__ == '__main__':
    main()

