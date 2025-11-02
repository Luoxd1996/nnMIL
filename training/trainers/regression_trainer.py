#!/usr/bin/env python3
"""
Regression Trainer for nnMIL
Handles regression tasks with target normalization
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nnMIL.training.trainers.base_trainer import BaseTrainer
from nnMIL.network_architecture.model_factory import create_mil_model
from nnMIL.utilities.utils import cosine_lr
from nnMIL.utilities.plan_loader import create_dataset_from_plan
from nnMIL.training.samplers.regression_sampler import RegressionBatchSampler
from nnMIL.training.losses.regression_loss import CombinedRegressionLoss
from nnMIL.training.callbacks.early_stopping import RegressionEarlyStopping


class TargetNormalizer:
    """Target normalization for regression tasks"""
    
    def __init__(self, method='zscore', fit_data=None):
        self.method = method
        self.fitted = False
        if fit_data is not None:
            self.fit(fit_data)
    
    def fit(self, data):
        """Fit normalization parameters"""
        data = np.array(data)
        
        if self.method == 'zscore':
            self.mean = np.mean(data)
            self.std = np.std(data)
        elif self.method == 'minmax':
            self.min_val = np.min(data)
            self.max_val = np.max(data)
        elif self.method == 'robust':
            self.median = np.median(data)
            q25, q75 = np.percentile(data, [25, 75])
            self.iqr = q75 - q25
        elif self.method == 'log_zscore':
            log_data = np.log1p(data)
            self.mean = np.mean(log_data)
            self.std = np.std(log_data)
        elif self.method == 'log':
            self.log_base = np.log1p(data).mean()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        self.fitted = True
    
    def transform(self, data):
        """Normalize data"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        data = np.array(data)
        
        if self.method == 'zscore':
            return (data - self.mean) / self.std
        elif self.method == 'minmax':
            return (data - self.min_val) / (self.max_val - self.min_val)
        elif self.method == 'robust':
            return (data - self.median) / self.iqr
        elif self.method == 'log_zscore':
            return (np.log1p(data) - self.mean) / self.std
        elif self.method == 'log':
            return np.log1p(data)
        return data
    
    def inverse_transform(self, data):
        """Inverse transform normalized data"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        data = np.array(data)
        
        if self.method == 'zscore':
            return data * self.std + self.mean
        elif self.method == 'minmax':
            return data * (self.max_val - self.min_val) + self.min_val
        elif self.method == 'robust':
            return data * self.iqr + self.median
        elif self.method == 'log_zscore':
            return np.expm1(data * self.std + self.mean)
        elif self.method == 'log':
            return np.expm1(data)
        return data


class RegressionTrainer(BaseTrainer):
    """Trainer for regression tasks"""
    
    def __init__(self, plan_path, model_type, fold=None, **kwargs):
        super().__init__(plan_path, model_type, fold, **kwargs)
        
        # Regression-specific initialization
        self.normalize_target = kwargs.get('normalize_target', True)
        self.normalization_method = kwargs.get('normalization_method', 'log')
        self.target_normalizer = None
        self.loss_function = kwargs.get('loss_function', 'combined')
        
        self.logger.info(f"Regression task initialized")
        self.logger.info(f"Normalize target: {self.normalize_target}, Method: {self.normalization_method}")
    
    def create_model(self):
        """Create regression model (single output)"""
        self.model = create_mil_model(
            model_type=self.model_type,
            input_dim=self.config['feature_dimension'],
            hidden_dim=self.config['hidden_dim'],
            num_classes=1,  # Regression: single output
            dropout=self.config['dropout']
        )
        self.model = self.model.to(self.device)
        self.logger.info(f"Model created: {self.model_type}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        return self.model
    
    def create_data_loaders(self):
        """Create data loaders for regression task"""
        self.logger.info("Loading datasets from plan file...")
        
        train_dataset = create_dataset_from_plan(self.plan_path, split='train', fold=self.fold)
        val_dataset = create_dataset_from_plan(self.plan_path, split='val', fold=self.fold)
        test_dataset = create_dataset_from_plan(self.plan_path, split='test', fold=self.fold)
        
        self.logger.info(f"Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Fit target normalizer on training data
        if self.normalize_target:
            train_targets = [train_dataset[i][3].item() for i in range(len(train_dataset))]
            self.target_normalizer = TargetNormalizer(method=self.normalization_method, fit_data=train_targets)
            self.logger.info(f"Target normalizer fitted: {self.normalization_method}")
        
        # Default collate function
        def default_collate_fn(batch):
            features_list = [item[0] for item in batch]
            coords_list = [item[1] for item in batch]
            bag_sizes = [item[2] for item in batch]
            labels = [item[3].item() for item in batch]
            
            batch_features = torch.stack(features_list)
            batch_coords = torch.stack(coords_list)
            batch_bag_sizes = torch.stack(bag_sizes)
            batch_labels = torch.tensor(labels, dtype=torch.float32)
            
            # Normalize targets if needed
            if self.target_normalizer:
                batch_labels = torch.tensor(
                    self.target_normalizer.transform(np.array(labels)),
                    dtype=torch.float32
                )
            
            return batch_features, batch_coords, batch_bag_sizes, batch_labels
        
        batch_size = self.config.get('batch_size', 32)
        batch_sampler_type = self.config.get('batch_sampler', 'random')
        
        # Create train loader
        if batch_sampler_type == 'regression' and batch_size > 1:
            train_sampler = RegressionBatchSampler(train_dataset, batch_size, shuffle=True, seed=self.seed)
            self.train_loader = DataLoader(
                train_dataset, batch_sampler=train_sampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
                collate_fn=default_collate_fn
            )
            self.logger.info("Using RegressionBatchSampler")
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, generator=generator,
                worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
                collate_fn=default_collate_fn
            )
            self.logger.info("Using random sampling")
        
        # Create val and test loaders
        def _worker_init_fn(worker_id):
            np.random.seed(self.seed + worker_id)
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=4,
            worker_init_fn=_worker_init_fn, collate_fn=default_collate_fn
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4,
            worker_init_fn=_worker_init_fn, collate_fn=default_collate_fn
        )
        
        self.logger.info(f"Data loaders created - Train: {len(self.train_loader)}, Val: {len(self.val_loader)}, Test: {len(self.test_loader)}")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def train(self):
        """Run training loop"""
        # CRITICAL: Reset PyTorch global state before training
        # This is especially important when running multiple folds sequentially
        torch.set_grad_enabled(True)
        
        # Clear CUDA cache to avoid memory issues between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.model is None:
            self.create_model()
        
        # Ensure model is in training mode and all parameters require grad
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        if self.train_loader is None:
            self.create_data_loaders()
        
        # Save training configuration
        self.save_training_config()
        
        # Setup optimizer
        named_parameters = list(self.model.named_parameters())
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
        include = lambda n, p: not exclude(n, p)
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        
        optimizer = torch.optim.AdamW([
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": self.config.get('weight_decay', 0.01)}
        ], lr=self.config.get('learning_rate', 3e-4))
        
        # Setup loss function
        loss_fn = CombinedRegressionLoss(loss_type=self.loss_function)
        
        # Setup LR scheduler
        num_epochs = self.config.get('num_epochs', 100)
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = len(self.train_loader) * self.config.get('warmup_epochs', 5)
        lr_scheduler = cosine_lr(optimizer, self.config.get('learning_rate', 3e-4), warmup_steps, total_steps)
        
        # Setup early stopping
        metric = self.dataset_info.get('metric', 'pearson')
        early_stopping = RegressionEarlyStopping(
            patience=self.config.get('patience', 10),
            verbose=True,
            metric=metric,
            save_dir=self.save_dir,
            model_type=self.model_type
        )
        
        # Setup mixed precision
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        fp16_scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None
        
        # Training loop
        self.model.train()
        global_step = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            epoch_start_time = time.time()
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Learning Rate: {current_lr:.2e}")
            if self.writer:
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                features, coords, bag_sizes, labels = batch
                
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type, dtype=torch.bfloat16):
                    output = self.model(features)
                    
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
                    
                    logits = logits.squeeze(-1)  # Ensure 1D
                    loss = loss_fn(logits, labels)
                
                if fp16_scaler is not None:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                lr_scheduler(global_step)
                global_step += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / max(epoch_steps, 1)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
            
            if self.writer:
                self.writer.add_scalar('Train/Loss', avg_loss, epoch)
            
            # Validation
            self.model.eval()
            torch.set_grad_enabled(False)
            val_metrics = self.evaluate('val')
            torch.set_grad_enabled(True)
            self.model.train()
            
            # Early stopping
            metric_value = val_metrics.get(f'val_{metric}', val_metrics.get('val_pearson', 0.0))
            if early_stopping(metric_value, self.model, epoch):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info("Loaded best model for final evaluation")
        
        self.model.eval()
        torch.set_grad_enabled(False)
        
        return self.model
    
    def evaluate(self, split='test'):
        """Evaluate on a split"""
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")
        
        loader = self.val_loader if split == 'val' else self.test_loader
        prefix = split.capitalize()
        
        self.model.eval()
        torch.set_grad_enabled(False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{prefix} Eval"):
                features, coords, bag_sizes, labels = batch
                
                features = features.to(self.device)
                
                output = self.model(features)
                if isinstance(output, dict):
                    logits = output['logits']
                else:
                    logits = output
                
                preds = logits.squeeze(-1).cpu().numpy()
                
                # Inverse transform if normalized
                if self.target_normalizer:
                    preds = self.target_normalizer.inverse_transform(preds)
                    labels = self.target_normalizer.inverse_transform(labels.cpu().numpy())
                else:
                    labels = labels.cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Calculate metrics
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_labels, all_preds)
        
        try:
            pearson, p_value = pearsonr(all_labels, all_preds)
        except:
            pearson = 0.0
            p_value = 1.0
        
        metrics = {
            f"{split}_mse": mse,
            f"{split}_rmse": rmse,
            f"{split}_r2": r2,
            f"{split}_pearson": pearson,
        }
        
        # Log metrics
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        # Save results to CSV if test split
        if split == 'test':
            save_csv_path = os.path.join(self.save_dir, f"results_{self.model_type}.csv")
            results_df = pd.DataFrame({
                'sample_id': [f"sample_{i}" for i in range(len(all_labels))],
                'true_target': all_labels,
                'predicted_target': all_preds,
                'error': all_labels - all_preds,
                'abs_error': np.abs(all_labels - all_preds)
            })
            results_df.to_csv(save_csv_path, index=False)
            self.logger.info(f"Results saved to {save_csv_path}")
        
        return metrics
    
    def save_training_config(self):
        """Save training configuration to JSON file"""
        actual_config = {
            "batch_size": self.config.get('batch_size', 32),
            "learning_rate": self.config.get('learning_rate', 3e-4),
            "batch_sampler": self.config.get('batch_sampler', 'random'),
            "model_type": self.model_type,
            "num_epochs": self.config.get('num_epochs', 100),
            "warmup_epochs": self.config.get('warmup_epochs', 5),
            "weight_decay": self.config.get('weight_decay', 0.01),
            "dropout": self.config.get('dropout', 0.25),
            "patience": self.config.get('patience', 10),
            "hidden_dim": self.config.get('hidden_dim', 256),
            "feature_dimension": self.config.get('feature_dimension'),
            "max_seq_length": self.config.get('max_seq_length'),
            "metric": self.dataset_info.get('metric', 'pearson'),
            "normalize_target": self.normalize_target,
            "normalization_method": self.normalization_method,
            "loss_function": self.loss_function,
        }
        
        dataset_info_dict = {
            "task_type": self.dataset_info.get('task_type'),
            "dataset_name": os.path.basename(os.path.dirname(self.plan_path)),
            "plan_path": self.plan_path,
            "fold": self.fold,
            "evaluation_setting": self.evaluation_setting,
        }
        
        config_path = os.path.join(self.save_dir, 'training_config.json')
        config_dict = {
            "dataset_info": dataset_info_dict,
            "actual_configuration": actual_config,
            "random_seed": self.seed,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Training configuration saved to {config_path}")

