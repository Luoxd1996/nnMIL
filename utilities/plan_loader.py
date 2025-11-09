"""
Utility functions for loading datasets and configurations from plan files.
"""
import os
import json
import pandas as pd
import tempfile
from typing import Dict, List, Optional
from nnMIL.data.dataset import UnifiedMILDataset


def find_plan_file(task_id: str) -> str:
    """
    Find plan file for a given task ID (like nnUNet).
    
    Args:
        task_id: Task identifier (e.g., 'Dataset001_ebrains')
    
    Returns:
        Path to dataset_plan.json
    
    Search order:
    1. examples/{task_id}/dataset_plan.json
    2. nnMIL_raw_data/{task_id}/dataset_plan.json
    3. {task_id}/dataset_plan.json (if task_id is already a path)
    4. Current directory: {task_id}/dataset_plan.json
    """
    # Check if task_id is already a full path
    if os.path.isabs(task_id) or os.path.exists(task_id):
        plan_path = os.path.join(task_id, 'dataset_plan.json') if os.path.isdir(task_id) else task_id
        if os.path.exists(plan_path):
            return plan_path
    
    # Try examples directory (most common location)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    examples_dir = os.path.join(project_root, 'examples')
    plan_path = os.path.join(examples_dir, task_id, 'dataset_plan.json')
    if os.path.exists(plan_path):
        return plan_path
    
    # Try nnMIL_raw_data directory (for trained models)
    raw_data_dir = os.path.join(project_root, 'nnMIL_raw_data')
    plan_path = os.path.join(raw_data_dir, task_id, 'dataset_plan.json')
    if os.path.exists(plan_path):
        return plan_path
    
    # Try current directory
    plan_path = os.path.join(task_id, 'dataset_plan.json')
    if os.path.exists(plan_path):
        return plan_path
    
    raise FileNotFoundError(
        f"Could not find dataset_plan.json for task '{task_id}'. "
        f"Searched in: {examples_dir}/{task_id}/dataset_plan.json, "
        f"{raw_data_dir}/{task_id}/dataset_plan.json, {plan_path}"
    )


def load_plan(plan_path: str) -> Dict:
    """Load plan file and return as dictionary"""
    with open(plan_path, 'r') as f:
        return json.load(f)


def create_dataset_from_plan(
    plan_path: str,
    split: str,
    fold: Optional[int] = None,
    transform=None
) -> UnifiedMILDataset:
    """
    Create dataset from plan file slide_info.
    
    Since plan file already contains everything:
    - max_seq_length (calculated from feature statistics: median * 0.5)
    - slide_info (all necessary columns: slide_id, label/event/time, etc.)
    - data splits (already filtered by split)
    - feature validation (already done during planning)
    
    This function passes all info to UnifiedMILDataset, which skips complex logic
    when skip_feature_validation=True and max_seq_length is provided.
    
    Args:
        plan_path: Path to dataset_plan.json
        split: 'train', 'val', or 'test' (passed to UnifiedMILDataset)
        fold: Fold number for 5-fold CV (0-4), or None for official_split
        transform: Optional transform to apply
    
    Returns:
        UnifiedMILDataset instance (with plan data, skips complex logic)
    """
    plan = load_plan(plan_path)
    
    # Get data splits from plan
    if fold is not None:
        split_key = f'fold_{fold}'
        if split_key not in plan['data_splits']:
            raise ValueError(f"Fold {fold} not found in plan file")
        if split not in plan['data_splits'][split_key]:
            raise ValueError(f"Split '{split}' not found in fold {fold}")
        slide_info = plan['data_splits'][split_key][split]['slide_info']
    else:
        if 'official_split' not in plan['data_splits']:
            raise ValueError("Official split not found in plan file")
        if split not in plan['data_splits']['official_split']:
            raise ValueError(f"Split '{split}' not found in official_split")
        slide_info = plan['data_splits']['official_split'][split]['slide_info']
    
    # Handle empty split (e.g., no test split)
    if len(slide_info) == 0:
        # Return empty dataset - create a minimal DataFrame
        df = pd.DataFrame(columns=['slide_id', 'patient_id'])
        # Add task-specific columns
        task_type = plan['task_type']
        if task_type == 'classification':
            df['label'] = []
        elif task_type == 'survival':
            df['event'] = []
            df['time'] = []
        elif task_type == 'regression':
            df['target'] = []
    else:
        # Create DataFrame from slide_info (plan already has all columns)
        df = pd.DataFrame(slide_info)
    
    # Get configuration from plan (already calculated)
    config = plan['training_configuration']
    task_type = plan['task_type']
    feature_dir = plan['feature_dir']
    
    # Create temporary CSV for UnifiedMILDataset (it expects CSV input)
    temp_csv = os.path.join(tempfile.gettempdir(), 
                           f"plan_{os.path.basename(plan_path)}_{split}_{fold or 'official'}.csv")
    df.to_csv(temp_csv, index=False)
    
    # Get dataset name from plan
    dataset_name = plan.get('task_name', plan.get('name', 'plan_dataset'))
    
    # Create UnifiedMILDataset with all info from plan
    # Since everything is already calculated, it will skip complex logic
    dataset = UnifiedMILDataset(
        csv_path=temp_csv,
        features_dir=feature_dir,
        task_type=task_type,
        dataset_name=dataset_name,
        split=split,  # Pass split so it knows train vs val/test for random crop
        fold=None,  # Already filtered by slide_info
        max_seq_length=config['max_seq_length'],  # Already calculated: median * 0.5
        use_original_length=config['use_original_length'],
        transform=transform,
        skip_feature_validation=True,  # Already validated during planning
        seed=42,  # Use fixed seed for deterministic random sampling
        max_seq_length_ratio=0.5  # Not used when max_seq_length provided
    )
    
    return dataset


def get_config_from_plan(plan_path: str) -> Dict:
    """Get training configuration from plan file"""
    plan = load_plan(plan_path)
    return plan['training_configuration']


def get_dataset_info_from_plan(plan_path: str) -> Dict:
    """Get dataset information from plan file"""
    plan = load_plan(plan_path)
    return {
        'task_type': plan['task_type'],
        'task_name': plan.get('task_name', ''),
        'feature_dir': plan['feature_dir'],
        'metric': plan.get('metric', 'bacc'),
        'evaluation_setting': plan.get('evaluation_setting', 'official_split'),
        'labels': plan.get('labels', {}),
        'num_classes': plan['training_configuration'].get('num_classes'),
    }


def create_dataset_from_features_dir(
    features_dir: str,
    plan_path: Optional[str] = None,
    transform=None
) -> UnifiedMILDataset:
    """
    Create dataset by scanning a features directory for all .h5 files.
    Similar to nnUNet style: just point to a directory and predict.
    
    Args:
        features_dir: Directory containing feature .h5 files
        plan_path: Optional plan file path to get configuration (task_type, max_seq_length, etc.)
                   If None, will use default classification task
        transform: Optional transform to apply
    
    Returns:
        UnifiedMILDataset instance for inference
    """
    # Scan for all .h5 files in the directory (recursively)
    h5_files = []
    if os.path.isdir(features_dir):
        # Check if TCGA-style (has TCGA-* subdirectories)
        for root, dirs, files in os.walk(features_dir):
            for fn in files:
                if fn.endswith('.h5'):
                    h5_path = os.path.join(root, fn)
                    # Get slide_id from filename (remove .h5 extension)
                    slide_id = os.path.splitext(fn)[0]
                    h5_files.append((slide_id, h5_path))
    else:
        raise ValueError(f"Features directory does not exist: {features_dir}")
    
    if len(h5_files) == 0:
        raise ValueError(f"No .h5 files found in directory: {features_dir}")
    
    print(f"Found {len(h5_files)} feature files in {features_dir}")
    
    # Create DataFrame from found files
    df = pd.DataFrame({
        'slide_id': [slide_id for slide_id, _ in h5_files],
        'patient_id': [slide_id for slide_id, _ in h5_files]  # Use slide_id as patient_id for inference
    })
    
    # Get configuration from plan if provided
    if plan_path and os.path.exists(plan_path):
        plan = load_plan(plan_path)
        config = plan.get('training_configuration', {})
        task_type = plan.get('task_type', 'classification')
        dataset_name = plan.get('task_name', plan.get('name', 'inference_dataset'))
        max_seq_length = config.get('max_seq_length')
        use_original_length = config.get('use_original_length', False)
    else:
        # Default configuration for inference
        task_type = 'classification'
        dataset_name = 'inference_dataset'
        max_seq_length = None
        use_original_length = True  # For inference, use all patches
    
    # For inference, we don't need labels, but UnifiedMILDataset expects them
    # Add dummy labels based on task_type
    if task_type == 'classification':
        df['label'] = 0  # Dummy label, won't be used for inference
    elif task_type == 'survival':
        df['event'] = 0  # Dummy
        df['time'] = 0.0  # Dummy
    elif task_type == 'regression':
        df['target'] = 0.0  # Dummy
    
    # Create temporary CSV for UnifiedMILDataset
    temp_csv = os.path.join(tempfile.gettempdir(), 
                           f"features_dir_{os.path.basename(features_dir)}.csv")
    df.to_csv(temp_csv, index=False)
    
    # Create UnifiedMILDataset
    # For inference (test split), use all original patches
    dataset = UnifiedMILDataset(
        csv_path=temp_csv,
        features_dir=features_dir,
        task_type=task_type,
        dataset_name=dataset_name,
        split='test',  # Always use test mode for inference (use all patches)
        fold=None,
        max_seq_length=max_seq_length,
        use_original_length=use_original_length,
        transform=transform,
        skip_feature_validation=False,  # Validate files exist
        seed=42,  # Use fixed seed for deterministic random sampling
        max_seq_length_ratio=0.5
    )
    
    return dataset

