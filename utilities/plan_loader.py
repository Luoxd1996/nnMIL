"""
Utility functions for loading datasets and configurations from plan files.
"""
import os
import json
import pandas as pd
import tempfile
from typing import Dict, List, Optional
from nnMIL.data.dataset import UnifiedMILDataset


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

