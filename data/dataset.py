"""
Unified MIL Dataset for Classification, Regression, and Survival Analysis

This module provides a single UnifiedMILDataset class that handles:
- Classification tasks (GenericMILDataset, TCGAMILDataset)
- Survival analysis (SurvivalMILDataset)
- External validation (ExternalValMILDataset)
- Regression tasks

All previous dataset classes are maintained for backward compatibility.
"""
import os
import h5py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class UnifiedMILDataset(Dataset):
    """
    Unified MIL dataset - simple and generic, like nnUNet.
    
    For plan files: all configuration comes from plan (no dataset-specific logic).
    For legacy CSV: minimal, generic column detection.
    
    Key principles:
    - Simple and generic (no dataset-specific hardcoding)
    - Training: pure random crop for data augmentation
    - Validation/Test: fixed crop (first max_seq_length patches)
    - All task-specific info comes from plan file or standard CSV format
    """
    def __init__(self, csv_path, features_dir, task_type='classification', 
                 transform=None, dataset_name="ebrains", split=None, fold=None,
                 max_seq_length=None, use_original_length=False, 
                 max_seq_length_ratio=0.5, skip_feature_validation=False):
        """
        Args:
            csv_path (str): Path to CSV file
            features_dir (str or list): Directory(ies) containing H5 feature files
            task_type (str): 'classification', 'regression', 'survival', or 'external_val'
            transform: Optional transform to apply to features
            dataset_name (str): Dataset name (used for column name detection)
            split (str): Data split ('train', 'val', 'test')
            fold (int): Cross-validation fold (0-4) for survival tasks
            max_seq_length (int): Maximum sequence length for padding
            use_original_length (bool): If True, use original feature length
            max_seq_length_ratio (float): Ratio of median sequence length to use
            skip_feature_validation (bool): Skip feature file validation (faster)
        """
        self.df = pd.read_csv(csv_path)
        self.task_type = task_type
        self.transform = transform
        self.dataset_name = dataset_name
        self.split = split  # Passed from training script (train/val/test)
        self.fold = fold
        self.skip_feature_validation = skip_feature_validation
        self.max_seq_length_ratio = max_seq_length_ratio
        self.use_original_length = use_original_length  # Store for __getitem__
        
        # Normalize features_dir to handle multiple directories
        if isinstance(features_dir, (list, tuple)):
            self.features_dirs = list(features_dir)
        elif isinstance(features_dir, str) and "," in features_dir:
            self.features_dirs = [p.strip() for p in features_dir.split(",") if p.strip()]
        else:
            self.features_dirs = [features_dir]
        
        # Check if TCGA-style features root (has TCGA-* subdirectories)
        self.is_tcga_features_root = self._check_tcga_features_root(self.features_dirs[0])
        
        # Determine column names based on dataset_name and task_type
        self._determine_column_names()
        
        # Filter by split and fold
        self._filter_by_split_and_fold()
        
        # Handle label conversion based on task_type
        self._process_labels()
        
        # Build feature index or validate features
        # If skip_feature_validation=True (e.g., from plan), skip all validation
        if skip_feature_validation:
            # Plan file already validated features, just build simple lookup if TCGA
            if self.is_tcga_features_root:
                self.slide_to_h5 = self._build_tcga_index(self.features_dirs[0])
        else:
            # Only do complex validation if not from plan
            if self.is_tcga_features_root:
                self.slide_to_h5 = self._build_tcga_index(self.features_dirs[0])
                self._filter_tcga_samples()
            else:
                self._validate_features()
        
        # Compute sequence length settings
        # Key rules:
        # - Training: use configured length (max_seq_length) OR original with 10% drop
        # - Validation/Test: ALWAYS use all original patches (no trimming, no dropping)
        if split in ['val', 'test']:
            # Validation and Test: ALWAYS use all original patches (override any config)
            self.max_seq_length = None
            self.use_original_length = True
            print(f"{split.capitalize()}: Using ALL original patches (no trimming, no dropping)")
        elif use_original_length:
            # Training with original length: drop 10% for data augmentation
            self.max_seq_length = None
            self.use_original_length = True
            print(f"Training: Using original length with 10% random drop for data augmentation")
        elif skip_feature_validation:
            # From plan file: max_seq_length already calculated for training
            # NEVER calculate again - planning already did it!
            self.max_seq_length = max_seq_length
            self.use_original_length = False
            print(f"Training: Using max_seq_length from plan: {self.max_seq_length} (random selection)")
        elif max_seq_length is not None:
            # Legacy usage: max_seq_length provided directly for training
            self.max_seq_length = max_seq_length
            self.use_original_length = False
            print(f"Training: Using max_seq_length: {self.max_seq_length} (random selection)")
        elif split == "train" or (split is None and "train" in str(csv_path)):
            # Legacy usage: calculate only for training if not from plan and not provided
            self.max_seq_length = self._calculate_adaptive_max_length(ratio=max_seq_length_ratio)
            self.use_original_length = False
            print(f"Training: Calculated max sequence length: {self.max_seq_length} (random selection)")
        else:
            # Fallback: use original for unknown splits
            self.max_seq_length = None
            self.use_original_length = True
            print(f"Unknown split '{split}': Using all original patches")
        
        print(f"{self.dataset_name} ({task_type}) dataset loaded: {len(self.df)} samples")
        if task_type == 'classification' and len(self.df) > 0:
            print(f"Class distribution: {self.df[self.label_col].value_counts().sort_index().to_dict()}")
    
    def _check_tcga_features_root(self, features_dir):
        """Check if features_dir is a TCGA-style root with TCGA-* subdirectories"""
        if not os.path.isdir(features_dir):
            return False
        for root, dirs, files in os.walk(features_dir):
            parts = root.strip(os.sep).split(os.sep)
            if any(p.startswith('TCGA-') for p in parts):
                return True
        return False
    
    def _determine_column_names(self):
        """Determine column names - simple and generic, no dataset-specific logic"""
        # For plan files (standard format): use standard column names
        if self.skip_feature_validation:
            # Plan file: always use standard names (planning handled everything)
            self.slide_id_col = 'slide_id'
            if self.task_type == 'survival':
                self.label_col = None  # Survival uses status/time, not label
            elif self.task_type == 'regression':
                self.label_col = 'target' if 'target' in self.df.columns else 'label'
            else:
                self.label_col = 'label'
            return
        
        # Legacy CSV format: simple detection (minimal, no dataset-specific)
        self.slide_id_col = 'slide_id' if 'slide_id' in self.df.columns else (
            'Slide' if 'Slide' in self.df.columns else 'slide_id'
        )
        
        if self.task_type == 'survival':
            self.label_col = None  # Survival uses status/time
        elif self.task_type == 'regression':
            self.label_col = 'target' if 'target' in self.df.columns else 'label'
        else:
            # Classification: try standard column names
            if 'label' in self.df.columns:
                self.label_col = 'label'
            elif 'target' in self.df.columns:
                self.label_col = 'target'
            else:
                raise ValueError(f"Classification task requires 'label' or 'target' column, found: {self.df.columns.tolist()}")
    
    def _filter_by_split_and_fold(self):
        """Filter dataframe by split and fold - simple and generic"""
        # If from plan file, data already filtered - skip all filtering
        if self.skip_feature_validation:
            return  # Plan slide_info already contains only the requested split
        
        # Legacy CSV format: simple filtering
        if self.split is not None and 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == self.split]
        
        # Handle fold filtering (generic, no task-specific logic)
        if self.fold is not None:
            if f'fold_{self.fold}' in self.df.columns:
                self.df = self.df[self.df[f'fold_{self.fold}'] == 1]
            elif 'fold' in self.df.columns:
                self.df = self.df[self.df['fold'] == self.fold]
    
    def _process_labels(self):
        """Process labels - simple and generic, no dataset-specific logic"""
        if self.task_type == 'survival':
            # Survival: ensure status and time columns exist
            if 'status' not in self.df.columns:
                # Try 'event' column if 'status' not found
                if 'event' in self.df.columns:
                    self.df['status'] = self.df['event']
                else:
                    raise ValueError(f"Survival task requires 'status' or 'event' and 'time' columns")
            if 'time' not in self.df.columns:
                raise ValueError(f"Survival task requires 'time' column")
            self.df = self.df.dropna(subset=['status', 'time'])
            self.df['status'] = self.df['status'].astype(int)
            self.df['time'] = self.df['time'].astype(float)
            return
        
        # Classification and regression tasks
        if self.label_col is None or self.label_col not in self.df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in CSV. Available columns: {self.df.columns.tolist()}")
        
        self.df = self.df.dropna(subset=[self.label_col])
        
        # Convert to numeric - generic, no dataset-specific mapping
        if self.task_type == 'regression':
            self.df[self.label_col] = pd.to_numeric(self.df[self.label_col], errors='coerce')
        else:
            # Classification: convert to integer
            self.df[self.label_col] = pd.to_numeric(self.df[self.label_col], errors='coerce')
            self.df[self.label_col] = self.df[self.label_col].astype(int)
        
        self.df = self.df.dropna(subset=[self.label_col])
        
        # For classification: compute class info (needed for label_to_idx mapping)
        if self.task_type == 'classification' and len(self.df) > 0:
            self.unique_labels = sorted(self.df[self.label_col].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
            self.num_classes = len(self.unique_labels)
    
    def _build_tcga_index(self, root_dir):
        """Build index of slide_id -> h5_path for TCGA-style features root"""
        index = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            parts = dirpath.strip(os.sep).split(os.sep)
            if not any(p.startswith('TCGA-') for p in parts):
                continue
            for fn in filenames:
                if fn.endswith('.h5'):
                    slide_id = fn[:-3]
                    abs_path = os.path.join(dirpath, fn)
                    if slide_id not in index:
                        index[slide_id] = abs_path
        return index
    
    def _filter_tcga_samples(self):
        """Filter samples that have corresponding TCGA features"""
        original_count = len(self.df)
        has_feat_mask = self.df[self.slide_id_col].map(lambda s: s in self.slide_to_h5)
        self.df = self.df[has_feat_mask].reset_index(drop=True)
        missing = original_count - len(self.df)
        if missing > 0:
            print(f"Filtered {missing} samples without TCGA features")
    
    def _validate_features(self):
        """Validate and filter samples with existing feature files"""
        if len(self.df) == 0:
            return
        
        original_count = len(self.df)
        existing_samples = []
        missing_samples = []
        
        file_id_col = getattr(self, 'file_id_col', self.slide_id_col)
        
        for idx, row in self.df.iterrows():
            file_id = row[file_id_col]
            h5_path = self._resolve_h5_path(file_id)
            
            if h5_path is not None and os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        if 'features' in f and 'coords' in f:
                            existing_samples.append(idx)
                        else:
                            missing_samples.append(file_id)
                except:
                    missing_samples.append(file_id)
            else:
                missing_samples.append(file_id)
        
        self.df = self.df.iloc[existing_samples].reset_index(drop=True)
        
        if len(missing_samples) > 0 and not self.skip_feature_validation:
            print(f"Feature validation: {len(existing_samples)} valid, {len(missing_samples)} missing")
    
    def _resolve_h5_path(self, file_id):
        """Resolve H5 file path from file identifier"""
        # Normalize ID by removing extensions
        base_filename = str(file_id)
        for ext in ['.svs', '.tif', '.tiff', '.ndpi', '.vsi', '.scn', '.mrxs', '.bif', '.h5']:
            if base_filename.endswith(ext):
                base_filename = base_filename[:-len(ext)]
                break
        
        # Try each features directory
        for base_dir in self.features_dirs:
            candidate = os.path.join(base_dir, f"{base_filename}.h5")
            if os.path.exists(candidate):
                return candidate
        
        # Try TCGA-style lookup if applicable
        if hasattr(self, 'slide_to_h5'):
            if base_filename in self.slide_to_h5:
                return self.slide_to_h5[base_filename]
        
        return None
    
    def _calculate_adaptive_max_length(self, ratio=0.5):
        """Calculate adaptive max_seq_length based on dataset median"""
        if len(self.df) == 0:
            return None
        
        lengths = []
        file_id_col = getattr(self, 'file_id_col', self.slide_id_col)
        
        sample_size = min(100, len(self.df))  # Sample up to 100 files for speed
        sample_df = self.df.sample(n=sample_size, random_state=42) if len(self.df) > sample_size else self.df
        
        for _, row in tqdm(sample_df.iterrows(), desc="Calculating max length", total=len(sample_df)):
            file_id = row[file_id_col]
            h5_path = self._resolve_h5_path(file_id)
            
            if h5_path and os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        if 'features' in f:
                            lengths.append(f['features'].shape[0])
                except:
                    pass
        
        if len(lengths) == 0:
            return None
        
        max_length = int(np.median(lengths) * ratio)
        return max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get file identifier
        file_id_col = getattr(self, 'file_id_col', self.slide_id_col)
        file_id = row[file_id_col]
        
        # Resolve H5 path
        if self.is_tcga_features_root:
            h5_path = self.slide_to_h5.get(row[self.slide_id_col])
        else:
            h5_path = self._resolve_h5_path(file_id)
        
        if h5_path is None or not os.path.exists(h5_path):
            raise FileNotFoundError(f"Feature file not found for {file_id}")
        
        # Load features
        with h5py.File(h5_path, 'r') as f:
            features = torch.from_numpy(f['features'][:]).float()
            coords = torch.from_numpy(f['coords'][:]).float()
        
        # Apply sequence length handling based on split and settings
        # Rules (same as GenericMILDataset):
        # - Training: if max_seq_length is set, ALL samples are processed to max_seq_length 
        #            (random selection if longer, padding if shorter - done at dataset level)
        #            if use_original_length, randomly drop 10% for data augmentation
        # - Validation/Test: ALWAYS use all original patches (no trimming, no dropping)
        # Store original number of patches before any processing (for bag_size mask)
        original_num_patches = len(features)
        
        if self.split in ['val', 'test']:
            # Validation and Test: ALWAYS use all original patches (no modification)
            # Keep all features and coords as-is
            actual_num_patches = original_num_patches
            pass
        elif self.use_original_length:
            # Training with original length: randomly drop 10% for data augmentation
            num_patches = len(features)
            num_to_keep = int(num_patches * 0.9)  # Keep 90%
            if num_to_keep < num_patches and num_to_keep > 0:
                indices = np.random.choice(num_patches, num_to_keep, replace=False)
                indices = np.sort(indices)  # Keep order for consistency
                features = features[indices]
                coords = coords[indices]
                actual_num_patches = num_to_keep
            else:
                actual_num_patches = original_num_patches
        elif self.max_seq_length is not None:
            # Training with max_seq_length: process to fixed length at dataset level
            num_patches = len(features)
            
            if num_patches > self.max_seq_length:
                # Random selection if longer
                indices = np.random.choice(num_patches, self.max_seq_length, replace=False)
                features = features[indices]
                coords = coords[indices]
                actual_num_patches = self.max_seq_length
            elif num_patches < self.max_seq_length:
                # Pad if shorter (at dataset level, like GenericMILDataset)
                padding_size = self.max_seq_length - num_patches
                features = torch.cat([
                    features,
                    torch.zeros(padding_size, features.shape[1], dtype=features.dtype)
                ], dim=0)
                coords = torch.cat([
                    coords,
                    torch.zeros(padding_size, coords.shape[1], dtype=coords.dtype)
                ], dim=0)
                actual_num_patches = original_num_patches  # Keep original count for mask
            else:
                # Exact length
                actual_num_patches = original_num_patches
        else:
            # No max_seq_length set, use original length
            actual_num_patches = original_num_patches
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(features)
        
        # Prepare output based on task_type
        # Use actual_num_patches for bag_size (original count before padding, for mask)
        bag_size = torch.tensor(actual_num_patches, dtype=torch.long)
        
        if self.task_type == 'survival':
            status = torch.tensor(int(row['status']), dtype=torch.float32)
            time = torch.tensor(float(row['time']), dtype=torch.float32)
            patient_id = row.get('patient_id', row.get(self.slide_id_col, str(idx)))
            return features, coords, bag_size, status, time, patient_id
        elif self.task_type == 'regression':
            label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
            return features, coords, bag_size, label
        else:  # classification
            label = torch.tensor(self.label_to_idx[int(row[self.label_col])], dtype=torch.long)
            slide_id = row.get(self.slide_id_col, str(idx))
            return features, coords, bag_size, label, slide_id, self.dataset_name
    
    def get_class_weights(self):
        """Get class weights for imbalanced classification"""
        if self.task_type != 'classification':
            return None
        
        class_counts = self.df[self.label_col].value_counts().sort_index()
        total_samples = len(self.df)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights.values)


# Backward compatibility: Create wrapper classes for compatibility
class GenericMILDataset(UnifiedMILDataset):
    """Classification dataset (backward compatibility)"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('task_type', 'classification')
        super().__init__(*args, **kwargs)

class TCGAMILDataset(UnifiedMILDataset):
    """TCGA-style classification dataset (backward compatibility)"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('task_type', 'classification')
        super().__init__(*args, **kwargs)

class ExternalValMILDataset(UnifiedMILDataset):
    """External validation dataset (backward compatibility)"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('task_type', 'external_val')
        super().__init__(*args, **kwargs)

class SurvivalMILDataset(UnifiedMILDataset):
    """Survival analysis dataset (backward compatibility)"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('task_type', 'survival')
        super().__init__(*args, **kwargs)
    
    def get_survival_statistics(self):
        """Get survival statistics for the dataset"""
        stats = {
            'total_samples': len(self.df),
            'events': int(self.df['status'].sum()),
            'censored': int((1 - self.df['status']).sum()),
            'event_rate': float(self.df['status'].mean()),
            'mean_survival_time': float(self.df['time'].mean()),
            'median_survival_time': float(self.df['time'].median()),
            'max_survival_time': float(self.df['time'].max()),
            'min_survival_time': float(self.df['time'].min())
        }
        return stats

# Helper functions for backward compatibility (kept for legacy code)
def get_tcga_dataset_name(csv_filename):
    """Extract TCGA dataset name from CSV filename"""
    if csv_filename.startswith('tcga_') and csv_filename.endswith('_5fold.csv'):
        dataset_part = csv_filename[5:-10]
        return f"TCGA-{dataset_part.upper()}"
    elif isinstance(csv_filename, str) and 'TCGA-' in csv_filename:
        # Extract from full path or name
        parts = csv_filename.split('TCGA-')
        if len(parts) > 1:
            return f"TCGA-{parts[1].split('/')[0].split('_')[0].upper()}"
    else:
        raise ValueError(f"Unexpected CSV filename format: {csv_filename}")

def get_dss_dataset_name(csv_filename):
    """Extract DSS dataset name from CSV filename"""
    if csv_filename.endswith('_manifest.csv'):
        base = os.path.basename(csv_filename)
        return base.replace('_manifest.csv', '')
    return "dss"

def get_tcga_features_dir(features_base_dir, model_name, dataset_name):
    """Get TCGA features directory path"""
    # Handle both TCGA-BRCA and BRCA format
    if dataset_name.startswith('TCGA-'):
        dataset_clean = dataset_name[5:]  # Remove TCGA- prefix
    else:
        dataset_clean = dataset_name.upper()
    return os.path.join(features_base_dir, model_name, f"TCGA-{dataset_clean}", "h5_files")

def get_dss_features_dir(features_base_dir, model_name, dataset_name):
    """Get DSS features directory path"""
    return os.path.join(features_base_dir, model_name, dataset_name, "h5_files")

def get_pfs_dataset_name(csv_filename):
    """Extract PFS dataset name from CSV filename"""
    if csv_filename.endswith('_manifest.csv'):
        base = os.path.basename(csv_filename)
        return base.replace('_manifest.csv', '')
    return "pfs"

def get_pfs_features_dir(features_base_dir, model_name, dataset_name):
    """Get PFS features directory path"""
    return os.path.join(features_base_dir, model_name, dataset_name, "h5_files")


def random_length_collate_fn(batch, max_seq_length=None, min_ratio=0.5, max_ratio=1.0):
    """
    Custom collate function that creates batches with uniform length within each batch,
    but variable length across different batches.
    
    For test/val: max_seq_length=None means use all original patches (no trimming).
    """
    # Extract features and other data
    features_list = [item[0] for item in batch]
    coords_list = [item[1] for item in batch]
    lengths_list = [item[2] for item in batch]
    
    # Determine target length for this batch
    # For test/val (max_seq_length=None): use max length in batch (all patches)
    if max_seq_length is None:
        target_len = max([len(f) for f in features_list])
    else:
        if min_ratio == max_ratio == 1.0:
            target_len = max_seq_length
        else:
            ratio = np.random.uniform(min_ratio, max_ratio)
            target_len = int(max_seq_length * ratio)
            target_len = min(target_len, max([len(f) for f in features_list]))
    
    # Pad/trim all sequences to target_len
    padded_features = []
    padded_coords = []
    
    for features, coords in zip(features_list, coords_list):
        if len(features) >= target_len:
            # Random crop (for training) or use all (for test/val when max_seq_length=None)
            # For test/val with max_seq_length=None, target_len is the max in batch, so keep all
            if max_seq_length is None:
                # Test/val: use all patches, no random crop
                padded_features.append(features)
                padded_coords.append(coords)
            else:
                # Training: random crop
                start_idx = np.random.randint(0, len(features) - target_len + 1)
                padded_features.append(features[start_idx:start_idx + target_len])
                padded_coords.append(coords[start_idx:start_idx + target_len])
        else:
            # Pad
            pad_len = target_len - len(features)
            padded_features.append(torch.cat([features, torch.zeros(pad_len, features.shape[1], dtype=features.dtype, device=features.device)]))
            padded_coords.append(torch.cat([coords, torch.zeros(pad_len, coords.shape[1], dtype=coords.dtype, device=coords.device)]))
    
    # Stack into batches
    batch_features = torch.stack(padded_features)
    batch_coords = torch.stack(padded_coords)
    batch_lengths = torch.stack(lengths_list)
    
    # Handle remaining items based on task type
    # Classification: features, coords, length, label, slide_id, dataset_name (6 items, but label is long, slide_id is str)
    # Survival: features, coords, length, status, time, patient_id (6 items, but status/time are float, patient_id is str)
    # Regression: features, coords, length, label (4 items)
    
    # Check by examining the types of item[3] and item[4]
    first_item = batch[0]
    if len(first_item) == 4:  # Regression
        labels = torch.stack([item[3] for item in batch])
        return batch_features, batch_coords, batch_lengths, labels
    elif len(first_item) == 6:
        # Need to distinguish between classification and survival
        # Check item[4]: classification has slide_id (str), survival has time (float/tensor)
        if isinstance(first_item[4], str) or (hasattr(first_item[4], 'item') and isinstance(first_item[4].item() if hasattr(first_item[4], 'item') else first_item[4], str)):
            # Classification: features, coords, length, label, slide_id, dataset_name
            labels = torch.stack([item[3] for item in batch])
            slide_ids = [item[4] for item in batch]
            datasets = [item[5] for item in batch]
            return batch_features, batch_coords, batch_lengths, labels, slide_ids, datasets
        else:
            # Survival: features, coords, length, status, time, patient_id
            status_list = [item[3] for item in batch]
            time_list = [item[4] for item in batch]
            patient_ids = [item[5] for item in batch]
            status = torch.stack(status_list)
            time = torch.stack(time_list)
            return batch_features, batch_coords, batch_lengths, status, time, patient_ids
    else:
        raise ValueError(f"Unexpected batch item length: {len(first_item)}")
