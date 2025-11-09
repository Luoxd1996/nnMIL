"""
Experiment Planner for nnMIL

Automatically analyzes datasets and generates training plans.
Similar to nnUNet's experiment planning functionality.
"""
import os
import json
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import argparse


class ExperimentPlanner:
    """
    Plan experiments by analyzing datasets and generating configurations.
    """
    
    def __init__(self, dataset_dir: str, random_seed: int = 42):
        """
        Args:
            dataset_dir: Directory containing dataset.json and dataset.csv
            random_seed: Random seed for reproducibility
        """
        self.dataset_dir = Path(dataset_dir)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Load dataset.json and dataset.csv
        self.dataset_json_path = self.dataset_dir / "dataset.json"
        self.dataset_csv_path = self.dataset_dir / "dataset.csv"
        
        if not self.dataset_json_path.exists():
            raise FileNotFoundError(f"dataset.json not found: {self.dataset_json_path}")
        if not self.dataset_csv_path.exists():
            raise FileNotFoundError(f"dataset.csv not found: {self.dataset_csv_path}")
        
        with open(self.dataset_json_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        self.df = pd.read_csv(self.dataset_csv_path)
        
        # Extract key information
        self.task_type = self.dataset_info['task_type']
        # Handle both '5fold' and '5_fold' formats
        eval_setting = self.dataset_info.get('evaluation_setting', 'official_split')
        self.evaluation_setting = '5fold' if eval_setting in ['5fold', '5_fold'] else eval_setting
        self.feature_dir = self.dataset_info['feature_dir']
        
    def analyze_features(self) -> Dict:
        """
        Analyze feature files to get statistics.
        
        Returns:
            Dictionary containing feature statistics
        """
        print("Analyzing feature files...")
        
        # Get all unique slide_ids
        slide_id_col = 'slide_id'
        if self.task_type == 'survival':
            slide_ids = self.df[slide_id_col].unique()
        else:
            slide_ids = self.df[slide_id_col].unique()
        
        patch_counts = []
        feature_dim = None
        
        # Sample up to 100 files for dimension detection, then analyze all for patch counts
        sample_size = min(100, len(slide_ids))
        sample_slide_ids = np.random.choice(slide_ids, size=sample_size, replace=False)
        
        # First pass: detect feature dimension
        for slide_id in sample_slide_ids:
            h5_path = self._resolve_h5_path(slide_id)
            if h5_path and os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        if 'features' in f:
                            features = f['features']
                            if feature_dim is None:
                                feature_dim = features.shape[1]
                            break
                except Exception as e:
                    print(f"Warning: Could not read {h5_path}: {e}")
                    continue
        
        if feature_dim is None:
            raise ValueError("Could not determine feature dimension from feature files")
        
        # Second pass: count patches for all slides
        print(f"Analyzing {len(slide_ids)} slides...")
        for idx, slide_id in enumerate(slide_ids):
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(slide_ids)} slides...")
            
            h5_path = self._resolve_h5_path(slide_id)
            if h5_path and os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        if 'features' in f:
                            num_patches = f['features'].shape[0]
                            patch_counts.append(num_patches)
                except Exception as e:
                    print(f"Warning: Could not read {h5_path}: {e}")
                    continue
        
        if len(patch_counts) == 0:
            raise ValueError("No valid feature files found")
        
        patch_counts = np.array(patch_counts)
        
        statistics = {
            "feature_dimension": int(feature_dim),
            "num_patches_per_slide": {
                "min": int(patch_counts.min()),
                "max": int(patch_counts.max()),
                "mean": float(patch_counts.mean()),
                "median": float(np.median(patch_counts)),
                "percentile_25": float(np.percentile(patch_counts, 25)),
                "percentile_75": float(np.percentile(patch_counts, 75)),
                "percentile_95": float(np.percentile(patch_counts, 95))
            },
            "recommended_max_seq_length": int(np.median(patch_counts) * 0.5)
        }
        
        print(f"Feature analysis complete:")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Median patches per slide: {statistics['num_patches_per_slide']['median']:.0f}")
        print(f"  Recommended max_seq_length: {statistics['recommended_max_seq_length']}")
        
        return statistics
    
    def _resolve_h5_path(self, slide_id: str) -> Optional[str]:
        """Resolve H5 file path from slide_id"""
        # Remove common extensions
        base_filename = str(slide_id)
        for ext in ['.svs', '.tif', '.tiff', '.ndpi', '.vsi', '.scn', '.mrxs', '.bif', '.h5']:
            if base_filename.endswith(ext):
                base_filename = base_filename[:-len(ext)]
                break
        
        # Try feature directory
        h5_path = os.path.join(self.feature_dir, f"{base_filename}.h5")
        if os.path.exists(h5_path):
            return h5_path
        
        return None
    
    def create_data_splits(self) -> Dict:
        """
        Create data splits at patient level with stratification.
        
        Returns:
            Dictionary containing data splits
        """
        print("\nCreating data splits at patient level...")
        
        if self.evaluation_setting == '5fold':
            return self._create_5fold_splits()
        else:
            return self._create_official_splits()
    
    def _create_5fold_splits(self) -> Dict:
        """
        Create 5-fold cross-validation splits with:
        1. Patient-level splitting (all slides of same patient in same fold)
        2. Stratified by label/event for balanced distribution
        """
        splits = {}
        
        # Step 1: Get unique patients and their labels/events for stratification
        # This ensures patient-level splitting and stratified distribution
        patient_data = []
        for patient_id in self.df['patient_id'].unique():
            patient_slides = self.df[self.df['patient_id'] == patient_id]
            
            if self.task_type == 'classification':
                # Use most common label for this patient
                label = patient_slides['label'].mode()[0]
                patient_data.append({
                    'patient_id': patient_id,
                    'stratum': int(label)  # For stratification
                })
            elif self.task_type == 'survival':
                # Use most common event status for this patient
                event = patient_slides['event'].mode()[0]
                patient_data.append({
                    'patient_id': patient_id,
                    'stratum': int(event)  # For stratification
                })
            else:  # regression
                # For regression, use median target as stratum (binned if needed)
                # Or just use a simple value for stratification
                target = float(patient_slides['label'].median())
                # Bin targets for stratification (simple approach)
                patient_data.append({
                    'patient_id': patient_id,
                    'stratum': int(target)  # Or could bin into categories
                })
        
        patient_df = pd.DataFrame(patient_data)
        patient_ids_list = patient_df['patient_id'].tolist()
        strata = patient_df['stratum'].tolist()
        
        # Step 2: Create stratified 5-fold splits at patient level
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        fold_assignments = list(skf.split(patient_ids_list, strata))
        
        # Step 3: For each fold, create train/val/test splits
        for fold_idx in range(5):
            # Get patient indices for this fold
            train_patient_indices, test_patient_indices = fold_assignments[fold_idx]
            
            # Get patient IDs
            test_patient_ids = [patient_ids_list[i] for i in test_patient_indices]
            train_patient_ids_all = [patient_ids_list[i] for i in train_patient_indices]
            
            test_patients = set(test_patient_ids)
            train_patients_all = set(train_patient_ids_all)
            
            # Step 4: Stratified split of train_patients into train (87.5%) and val (12.5%)
            train_patient_list = sorted(list(train_patients_all))
            val_patient_list = []
            
            if self.task_type == 'classification':
                # Stratify by label - get labels for train patients
                train_patient_labels = []
                for patient_id in train_patient_list:
                    patient_slides = self.df[self.df['patient_id'] == patient_id]
                    label = patient_slides['label'].mode()[0]
                    train_patient_labels.append(int(label))
                
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=self.random_seed + fold_idx)
                train_indices, val_indices = next(sss.split(train_patient_list, train_patient_labels))
                val_patient_list = [train_patient_list[i] for i in val_indices]
                train_patient_list = [train_patient_list[i] for i in train_indices]
                
            elif self.task_type == 'survival':
                # Stratify by event - get events for train patients
                train_patient_events = []
                for patient_id in train_patient_list:
                    patient_slides = self.df[self.df['patient_id'] == patient_id]
                    event = patient_slides['event'].mode()[0]
                    train_patient_events.append(int(event))
                
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=self.random_seed + fold_idx)
                train_indices, val_indices = next(sss.split(train_patient_list, train_patient_events))
                val_patient_list = [train_patient_list[i] for i in val_indices]
                train_patient_list = [train_patient_list[i] for i in train_indices]
            
            else:  # regression
                # For regression, still use stratified split based on binned targets
                train_patient_targets = []
                for patient_id in train_patient_list:
                    patient_slides = self.df[self.df['patient_id'] == patient_id]
                    target = float(patient_slides['label'].median())
                    train_patient_targets.append(int(target))  # Or use bins
                
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=self.random_seed + fold_idx)
                train_indices, val_indices = next(sss.split(train_patient_list, train_patient_targets))
                val_patient_list = [train_patient_list[i] for i in val_indices]
                train_patient_list = [train_patient_list[i] for i in train_indices]
            
            # Step 5: Get all slides for each patient group
            # CRITICAL: All slides of a patient go to the same split
            train_slides = self.df[self.df['patient_id'].isin(train_patient_list)]['slide_id'].tolist()
            val_slides = self.df[self.df['patient_id'].isin(val_patient_list)]['slide_id'].tolist()
            test_slides = self.df[self.df['patient_id'].isin(test_patients)]['slide_id'].tolist()
            
            # Verify patient-level integrity: check that no patient appears in multiple splits
            train_patient_set = set(train_patient_list)
            val_patient_set = set(val_patient_list)
            test_patient_set = test_patients
            
            # Check for overlaps
            if train_patient_set & val_patient_set:
                raise ValueError(f"Fold {fold_idx}: Overlap between train and val patients!")
            if train_patient_set & test_patient_set:
                raise ValueError(f"Fold {fold_idx}: Overlap between train and test patients!")
            if val_patient_set & test_patient_set:
                raise ValueError(f"Fold {fold_idx}: Overlap between val and test patients!")
            
            # Calculate statistics
            splits[f'fold_{fold_idx}'] = {
                "train": self._calculate_split_statistics(train_patient_set, train_slides, "train"),
                "val": self._calculate_split_statistics(val_patient_set, val_slides, "val"),
                "test": self._calculate_split_statistics(test_patient_set, test_slides, "test")
            }
            
            # Print fold statistics with distribution info
            if self.task_type == 'classification':
                train_labels = self.df[self.df['patient_id'].isin(train_patient_list)]['label'].value_counts().to_dict()
                val_labels = self.df[self.df['patient_id'].isin(val_patient_list)]['label'].value_counts().to_dict()
                test_labels = self.df[self.df['patient_id'].isin(test_patients)]['label'].value_counts().to_dict()
                print(f"  Fold {fold_idx}: Train={len(train_patient_list)} patients ({train_labels}), "
                      f"Val={len(val_patient_list)} patients ({val_labels}), "
                      f"Test={len(test_patients)} patients ({test_labels})")
            elif self.task_type == 'survival':
                train_events = self.df[self.df['patient_id'].isin(train_patient_list)]['event'].sum()
                val_events = self.df[self.df['patient_id'].isin(val_patient_list)]['event'].sum()
                test_events = self.df[self.df['patient_id'].isin(test_patients)]['event'].sum()
                train_total = len(self.df[self.df['patient_id'].isin(train_patient_list)])
                val_total = len(self.df[self.df['patient_id'].isin(val_patient_list)])
                test_total = len(self.df[self.df['patient_id'].isin(test_patients)])
                print(f"  Fold {fold_idx}: Train={len(train_patient_list)} patients ({train_events}/{train_total} events), "
                      f"Val={len(val_patient_list)} patients ({val_events}/{val_total} events), "
                      f"Test={len(test_patients)} patients ({test_events}/{test_total} events)")
            else:
                print(f"  Fold {fold_idx}: Train={len(train_patient_list)} patients, "
                      f"Val={len(val_patient_list)} patients, Test={len(test_patients)} patients")
        
        return splits
    
    def _create_official_splits(self) -> Dict:
        """
        Create official splits (train/val/test) with:
        1. Patient-level splitting (all slides of same patient in same split)
        2. Stratified by label/event for balanced distribution
        """
        # Check if val split exists
        has_val = 'split' in self.df.columns and 'val' in self.df['split'].values
        
        if has_val:
            # Use existing splits, but verify patient-level integrity
            train_patients = set(self.df[self.df['split'] == 'train']['patient_id'].unique())
            val_patients = set(self.df[self.df['split'] == 'val']['patient_id'].unique())
            # Test split is optional - check if it exists
            has_test = 'split' in self.df.columns and 'test' in self.df['split'].values
            if has_test:
                test_patients = set(self.df[self.df['split'] == 'test']['patient_id'].unique())
            else:
                test_patients = set()
                print("  Note: No test split found in dataset. Test split will be empty.")
            
            # Verify no patient appears in multiple splits (only check non-empty splits)
            if train_patients & val_patients:
                raise ValueError("Existing splits: Overlap between train and val patients! "
                               "Please ensure each patient appears in only one split.")
            if has_test:
                if train_patients & test_patients:
                    raise ValueError("Existing splits: Overlap between train and test patients! "
                                   "Please ensure each patient appears in only one split.")
                if val_patients & test_patients:
                    raise ValueError("Existing splits: Overlap between val and test patients! "
                                   "Please ensure each patient appears in only one split.")
            
            train_slides = self.df[self.df['split'] == 'train']['slide_id'].tolist()
            val_slides = self.df[self.df['split'] == 'val']['slide_id'].tolist()
            if has_test:
                test_slides = self.df[self.df['split'] == 'test']['slide_id'].tolist()
            else:
                test_slides = []
            
            # Print distribution statistics
            if self.task_type == 'classification':
                train_labels = self.df[self.df['patient_id'].isin(train_patients)]['label'].value_counts().to_dict()
                val_labels = self.df[self.df['patient_id'].isin(val_patients)]['label'].value_counts().to_dict()
                test_labels = self.df[self.df['patient_id'].isin(test_patients)]['label'].value_counts().to_dict()
                print(f"  Using existing splits: Train={len(train_patients)} patients ({train_labels}), "
                      f"Val={len(val_patients)} patients ({val_labels}), "
                      f"Test={len(test_patients)} patients ({test_labels})")
            elif self.task_type == 'survival':
                train_events = self.df[self.df['patient_id'].isin(train_patients)]['event'].sum()
                val_events = self.df[self.df['patient_id'].isin(val_patients)]['event'].sum()
                test_events = self.df[self.df['patient_id'].isin(test_patients)]['event'].sum()
                train_total = len(self.df[self.df['patient_id'].isin(train_patients)])
                val_total = len(self.df[self.df['patient_id'].isin(val_patients)])
                test_total = len(self.df[self.df['patient_id'].isin(test_patients)])
                print(f"  Using existing splits: Train={len(train_patients)} patients ({train_events}/{train_total} events), "
                      f"Val={len(val_patients)} patients ({val_events}/{val_total} events), "
                      f"Test={len(test_patients)} patients ({test_events}/{test_total} events)")
            else:
                print(f"  Using existing splits: Train={len(train_patients)}, "
                      f"Val={len(val_patients)}, Test={len(test_patients)} patients")
            
        else:
            # Split train into train (80%) and val (20%) at PATIENT level with STRATIFICATION
            # CRITICAL: All slides of the same patient must go to the same split
            
            # Step 1: Get unique patients from train and test splits
            train_patients_all = set(self.df[self.df['split'] == 'train']['patient_id'].unique())
            # Test split is optional - check if it exists
            has_test = 'split' in self.df.columns and 'test' in self.df['split'].values
            if has_test:
                test_patients = set(self.df[self.df['split'] == 'test']['patient_id'].unique())
            else:
                test_patients = set()
                print("  Note: No test split found in dataset. Test split will be empty.")
            
            train_patient_list = sorted(list(train_patients_all))
            
            # Step 2: Collect label/event for each patient (for stratification)
            if self.task_type == 'classification':
                # For each patient, determine their label (most common label among their slides)
                train_patient_labels = []
                for patient_id in train_patient_list:
                    patient_slides = self.df[self.df['patient_id'] == patient_id]
                    label = patient_slides['label'].mode()[0]  # Most common label for this patient
                    train_patient_labels.append(int(label))
                
                # Step 3: Stratified split at PATIENT level by LABEL
                # This ensures balanced label distribution between train and val
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_seed)
                train_indices, val_indices = next(sss.split(train_patient_list, train_patient_labels))
                
                # Step 4: Get patient IDs for each split
                val_patients = {train_patient_list[i] for i in val_indices}
                train_patients = {train_patient_list[i] for i in train_indices}
                
            elif self.task_type == 'survival':
                # For each patient, determine their event status (most common event among their slides)
                train_patient_events = []
                for patient_id in train_patient_list:
                    patient_slides = self.df[self.df['patient_id'] == patient_id]
                    event = patient_slides['event'].mode()[0]  # Most common event for this patient
                    train_patient_events.append(int(event))
                
                # Stratified split at PATIENT level by EVENT
                # This ensures balanced event distribution between train and val
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_seed)
                train_indices, val_indices = next(sss.split(train_patient_list, train_patient_events))
                
                # Get patient IDs for each split
                val_patients = {train_patient_list[i] for i in val_indices}
                train_patients = {train_patient_list[i] for i in train_indices}
                
            else:  # regression
                # For each patient, determine their target value (median target among their slides)
                train_patient_targets = []
                for patient_id in train_patient_list:
                    patient_slides = self.df[self.df['patient_id'] == patient_id]
                    target = float(patient_slides['label'].median())  # Median target for this patient
                    train_patient_targets.append(int(target))  # Simple binning approach
                
                # Stratified split at PATIENT level by TARGET (binned)
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_seed)
                train_indices, val_indices = next(sss.split(train_patient_list, train_patient_targets))
                
                # Get patient IDs for each split
                val_patients = {train_patient_list[i] for i in val_indices}
                train_patients = {train_patient_list[i] for i in train_indices}
            
            # Step 5: Verify patient-level integrity
            # CRITICAL CHECK: Ensure no patient appears in multiple splits (only check non-empty splits)
            if train_patients & val_patients:
                raise ValueError("Overlap between train and val patients! "
                               "This should not happen - please check the split logic.")
            if has_test:
                if train_patients & test_patients:
                    raise ValueError("Overlap between train and test patients! "
                                   "This should not happen - please check the split logic.")
                if val_patients & test_patients:
                    raise ValueError("Overlap between val and test patients! "
                                   "This should not happen - please check the split logic.")
            
            # Step 6: Get ALL slides for each patient group
            # CRITICAL: All slides belonging to a patient go to the same split
            # This ensures patient-level splitting
            train_slides = self.df[self.df['patient_id'].isin(train_patients)]['slide_id'].tolist()
            val_slides = self.df[self.df['patient_id'].isin(val_patients)]['slide_id'].tolist()
            if has_test:
                test_slides = self.df[self.df['patient_id'].isin(test_patients)]['slide_id'].tolist()
            else:
                test_slides = []
            
            # Print detailed statistics
            if self.task_type == 'classification':
                train_labels = self.df[self.df['patient_id'].isin(train_patients)]['label'].value_counts().to_dict()
                val_labels = self.df[self.df['patient_id'].isin(val_patients)]['label'].value_counts().to_dict()
                test_labels = self.df[self.df['patient_id'].isin(test_patients)]['label'].value_counts().to_dict()
                print(f"  Created splits: Train={len(train_patients)} patients ({train_labels}), "
                      f"Val={len(val_patients)} patients ({val_labels}) (from train), "
                      f"Test={len(test_patients)} patients ({test_labels})")
            elif self.task_type == 'survival':
                train_events = self.df[self.df['patient_id'].isin(train_patients)]['event'].sum()
                val_events = self.df[self.df['patient_id'].isin(val_patients)]['event'].sum()
                test_events = self.df[self.df['patient_id'].isin(test_patients)]['event'].sum()
                train_total = len(self.df[self.df['patient_id'].isin(train_patients)])
                val_total = len(self.df[self.df['patient_id'].isin(val_patients)])
                test_total = len(self.df[self.df['patient_id'].isin(test_patients)])
                print(f"  Created splits: Train={len(train_patients)} patients ({train_events}/{train_total} events), "
                      f"Val={len(val_patients)} patients ({val_events}/{val_total} events) (from train), "
                      f"Test={len(test_patients)} patients ({test_events}/{test_total} events)")
            else:
                print(f"  Created splits: Train={len(train_patients)} patients, "
                      f"Val={len(val_patients)} patients (from train), Test={len(test_patients)} patients")
        
        return {
            "official_split": {
                "train": self._calculate_split_statistics(train_patients, train_slides, "train"),
                "val": self._calculate_split_statistics(val_patients, val_slides, "val"),
                "test": self._calculate_split_statistics(test_patients, test_slides, "test")
            }
        }
    
    def _calculate_split_statistics(self, patient_ids: set, slide_ids: List, split_name: str) -> Dict:
        """Calculate statistics for a split and include slide-level information"""
        # Handle empty split (e.g., no test split)
        if len(slide_ids) == 0:
            return {
                "patient_ids": [],
                "slide_ids": [],
                "slide_info": [],
                "statistics": {
                    "num_patients": 0,
                    "num_slides": 0
                }
            }
        
        split_df = self.df[self.df['slide_id'].isin(slide_ids)].copy()
        
        # Build slide information list (each slide with its metadata)
        slide_info = []
        for slide_id in slide_ids:
            slide_row = split_df[split_df['slide_id'] == slide_id].iloc[0]
            slide_dict = {
                "slide_id": slide_id,
                "patient_id": slide_row['patient_id'],
                "dataset_id": slide_row.get('dataset_id', self.dataset_info.get('task_name', ''))
            }
            
            # Add task-specific information
            if self.task_type == 'classification':
                slide_dict["label"] = int(slide_row['label'])
            elif self.task_type == 'survival':
                slide_dict["event"] = int(slide_row['event'])
                slide_dict["time"] = float(slide_row['time'])
            elif self.task_type == 'regression':
                slide_dict["target"] = float(slide_row['label'])
            
            slide_info.append(slide_dict)
        
        stats = {
            "patient_ids": sorted(list(patient_ids)),
            "slide_ids": slide_ids,
            "slide_info": slide_info,  # Detailed information for each slide
            "statistics": {
                "num_patients": len(patient_ids),
                "num_slides": len(slide_ids)
            }
        }
        
        if self.task_type == 'classification':
            class_dist = split_df['label'].value_counts().to_dict()
            stats["statistics"]["class_distribution"] = {str(k): int(v) for k, v in class_dist.items()}
            stats["statistics"]["num_classes"] = len(class_dist)
            
        elif self.task_type == 'survival':
            event_rate = split_df['event'].mean()
            median_time = split_df['time'].median()
            stats["statistics"]["event_rate"] = float(event_rate)
            stats["statistics"]["median_time"] = float(median_time)
            stats["statistics"]["num_events"] = int(split_df['event'].sum())
            stats["statistics"]["num_censored"] = int((1 - split_df['event']).sum())
            
        else:  # regression
            target_stats = split_df['label'].describe()
            stats["statistics"]["target_statistics"] = {
                "mean": float(target_stats['mean']),
                "std": float(target_stats['std']),
                "min": float(target_stats['min']),
                "max": float(target_stats['max'])
            }
        
        return stats
    
    def generate_training_config(self, feature_stats: Dict) -> Dict:
        """Generate training configuration based on statistics"""
        recommended_max_seq_length = feature_stats['recommended_max_seq_length']
        feature_dimension = feature_stats['feature_dimension']
        
        # Estimate batch size based on sequence length
        if recommended_max_seq_length < 5000:
            batch_size = 32
        elif recommended_max_seq_length < 10000:
            batch_size = 16
        else:
            batch_size = 8
        
        # Determine batch_sampler based on metric
        metric = self.dataset_info.get('metric', 'bacc').lower()
        
        if self.task_type == 'survival':
            # For survival tasks, use random sampling (no special batch sampler)
            batch_sampler = None  # None means use DataLoader's default random shuffle
            learning_rate = 1e-4  # Survival tasks use 1e-4
        else:
            # For classification/regression tasks
            if 'auc' in metric:
                batch_sampler = 'auc'  # AUC-friendly sampler
            elif metric in ['bacc', 'balanced_accuracy', 'f1', 'f1_score']:
                batch_sampler = 'balanced'  # Balanced sampler
            else:
                batch_sampler = 'random'  # Default random sampler
            learning_rate = 3e-4  # Classification/regression tasks use 3e-4
        
        config = {
            "feature_dimension": feature_dimension,
            "hidden_dim": 256,  # Fixed hidden dimension
            "max_seq_length": recommended_max_seq_length,
            "use_original_length": False,
            "batch_size": batch_size,
            "batch_sampler": batch_sampler,
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "num_epochs": 100,
            "warmup_epochs": 5,  # Warmup epochs
            "dropout": 0.25,  # Model dropout rate
            "patience": 10,  # Early stopping patience
        }
        
        # Add num_classes for classification tasks (from labels)
        if self.task_type == 'classification' and 'labels' in self.dataset_info:
            config["num_classes"] = len(self.dataset_info['labels'])
        
        # Add trainer_type for survival tasks (to distinguish between SurvivalTrainer and SurvivalPorpoiseTrainer)
        if self.task_type == 'survival':
            # Check if this is a Porpoise-style dataset (typically batch_size=1, nllsurv loss)
            # For now, we'll set it based on batch_size - can be overridden in dataset.json
            if batch_size == 1:
                config['trainer_type'] = 'survival_porpoise'
                config['survival_loss'] = 'nllsurv'
                config['nll_bins'] = 4  # Default number of bins for NLLSurv
            else:
                config['trainer_type'] = 'survival'
                config['survival_loss'] = 'cox'  # Default survival loss
        
        return config
    
    def plan_experiment(self, output_path: Optional[str] = None) -> Dict:
        """
        Main planning function.
        
        Args:
            output_path: Path to save plan file. If None, saves to dataset_dir/dataset_plan.json
        
        Returns:
            Dictionary containing the complete plan
        """
        print(f"Planning experiment for dataset: {self.dataset_dir}")
        print(f"Task type: {self.task_type}")
        print(f"Evaluation setting: {self.evaluation_setting}\n")
        
        # 1. Analyze features
        feature_stats = self.analyze_features()
        
        # 2. Create data splits
        data_splits = self.create_data_splits()
        
        # 3. Generate training config
        training_config = self.generate_training_config(feature_stats)
        
        # 4. Build complete plan (includes dataset.json content)
        plan = {
            **self.dataset_info,  # Include all fields from dataset.json
            "feature_statistics": feature_stats,
            "data_splits": data_splits,
            "training_configuration": training_config,
            "random_seed": self.random_seed,
            "planning_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 5. Save plan
        if output_path is None:
            output_path = self.dataset_dir / "dataset_plan.json"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(plan, f, indent=4)
        
        print(f"\n✅ Plan saved to: {output_path}")
        
        return plan


def main():
    parser = argparse.ArgumentParser(description='Plan nnMIL experiment')
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                       help='Directory containing dataset.json and dataset.csv')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output path for plan file (default: dataset_dir/dataset_plan.json)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    planner = ExperimentPlanner(args.dataset_dir, random_seed=args.seed)
    plan = planner.plan_experiment(output_path=args.output)
    
    print("\nPlanning complete!")


if __name__ == '__main__':
    main()

