#!/usr/bin/env python3
"""
Survival Trainer for nnMIL
Handles survival tasks with Cox/MSE/MAE loss, batch_size > 1
"""

import os
import sys
import time as time_module
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nnMIL.training.trainers.base_trainer import BaseTrainer
from nnMIL.network_architecture.model_factory import create_mil_model
from nnMIL.utilities.utils import cosine_lr
from nnMIL.utilities.plan_loader import create_dataset_from_plan
from nnMIL.training.samplers.survival_sampler import BalancedSurvivalSampler, StratifiedSurvivalSampler, RiskSetBatchSampler
from nnMIL.training.losses.survival_loss import SurvivalLoss, survival_c_index
from nnMIL.training.callbacks.early_stopping import EarlyStoppingSurvival


class SurvivalTrainer(BaseTrainer):
    """Trainer for survival tasks (Cox/MSE/MAE loss, batch_size > 1)"""
    
    def __init__(self, plan_path, model_type, fold=None, **kwargs):
        super().__init__(plan_path, model_type, fold, **kwargs)
        
        # Survival-specific initialization
        self.survival_loss = kwargs.get('survival_loss', self.config.get('survival_loss', 'cox'))
        self.logger.info(f"Survival task initialized")
        self.logger.info(f"Survival loss: {self.survival_loss}")
    
    def create_model(self):
        """Create survival model (single output)"""
        self.model = create_mil_model(
            model_type=self.model_type,
            input_dim=self.config['feature_dimension'],
            hidden_dim=self.config['hidden_dim'],
            num_classes=1,  # Survival: single risk score output
            dropout=self.config['dropout']
        )
        self.model = self.model.to(self.device)
        self.logger.info(f"Model created: {self.model_type}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        return self.model
    
    def create_data_loaders(self):
        """Create data loaders for survival task"""
        self.logger.info("Loading datasets from plan file...")
        
        train_dataset = create_dataset_from_plan(self.plan_path, split='train', fold=self.fold)
        val_dataset = create_dataset_from_plan(self.plan_path, split='val', fold=self.fold)
        test_dataset = create_dataset_from_plan(self.plan_path, split='test', fold=self.fold)
        
        self.logger.info(f"Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Default collate function
        def default_collate_fn(batch):
            features_list = [item[0] for item in batch]
            coords_list = [item[1] for item in batch]
            bag_sizes = [item[2] for item in batch]
            status = torch.stack([item[3] for item in batch])
            time = torch.stack([item[4] for item in batch])
            patient_ids = [item[5] for item in batch]
            
            # Check if batch has slide_ids (7 items)
            if len(batch[0]) == 7:
                slide_ids = [item[6] for item in batch]
            else:
                slide_ids = None
            
            batch_features = torch.stack(features_list)
            batch_coords = torch.stack(coords_list)
            batch_bag_sizes = torch.stack(bag_sizes)
            
            if slide_ids is not None:
                return batch_features, batch_coords, batch_bag_sizes, status, time, patient_ids, slide_ids
            else:
                return batch_features, batch_coords, batch_bag_sizes, status, time, patient_ids
        
        batch_size = self.config.get('batch_size', 32)
        batch_sampler_type = self.config.get('batch_sampler', None)  # Default to None (random sampling)
        
        # Create train loader with appropriate sampler
        if batch_sampler_type == 'risk_set' and batch_size > 1:
            train_sampler = RiskSetBatchSampler(train_dataset, batch_size, shuffle_within=True, seed=self.seed)
            self.train_loader = DataLoader(
                train_dataset, batch_sampler=train_sampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
                collate_fn=default_collate_fn
            )
            self.logger.info("Using RiskSetBatchSampler")
        elif batch_sampler_type == 'balanced_survival' and batch_size > 1:
            train_sampler = BalancedSurvivalSampler(train_dataset, batch_size, shuffle=True, seed=self.seed)
            self.train_loader = DataLoader(
                train_dataset, batch_sampler=train_sampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
                collate_fn=default_collate_fn
            )
            self.logger.info("Using BalancedSurvivalSampler")
        elif batch_sampler_type == 'stratified_survival' and batch_size > 1:
            train_sampler = StratifiedSurvivalSampler(train_dataset, batch_size, shuffle=True, seed=self.seed)
            self.train_loader = DataLoader(
                train_dataset, batch_sampler=train_sampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
                collate_fn=default_collate_fn
            )
            self.logger.info("Using StratifiedSurvivalSampler")
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
        ], lr=self.config.get('learning_rate', 1e-4))
        
        # Setup loss function
        loss_fn = SurvivalLoss(loss_type=self.survival_loss)
        
        # Setup LR scheduler
        num_epochs = self.config.get('num_epochs', 100)
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = len(self.train_loader) * self.config.get('warmup_epochs', 5)
        lr_scheduler = cosine_lr(optimizer, self.config.get('learning_rate', 1e-4), warmup_steps, total_steps)
        
        # Setup early stopping
        metric = self.dataset_info.get('metric', 'c_index')
        early_stopping = EarlyStoppingSurvival(
            patience=self.config.get('patience', 10),
            verbose=True,
            metric=metric,
            save_dir=self.save_dir,
            model_type=self.model_type,
            logger=self.logger
        )
        
        # Setup mixed precision
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        # For bfloat16, GradScaler is typically not needed due to sufficient dynamic range
        # Disable GradScaler to avoid the "No inf checks were recorded" error
        fp16_scaler = None
        
        # Training loop
        self.model.train()
        global_step = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            epoch_start_time = time_module.time()
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Learning Rate: {current_lr:.2e}")
            if self.writer:
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                # Survival dataset now returns 7 items: features, coords, bag_sizes, status, time, patient_ids, slide_ids
                if len(batch) == 7:
                    features, coords, bag_sizes, status, time, patient_ids, slide_ids = batch
                else:
                    # Old format with 6 items
                    features, coords, bag_sizes, status, time, patient_ids = batch
                
                features = features.to(self.device)
                status = status.to(self.device)
                time = time.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type, dtype=torch.bfloat16):
                    # Verify model is in training mode
                    if not self.model.training:
                        self.logger.warning(f"Model not in training mode at batch {batch_idx}, setting to train mode")
                        self.model.train()
                    
                    # Verify gradients are enabled
                    if not torch.is_grad_enabled():
                        self.logger.error(f"CRITICAL: torch.is_grad_enabled()=False at batch {batch_idx}")
                        torch.set_grad_enabled(True)
                    
                    # For survival models, pass is_cox=True if supported
                    if hasattr(self.model, 'forward') and 'is_cox' in self.model.forward.__code__.co_varnames:
                        output = self.model(features, is_cox=True)
                    else:
                        output = self.model(features)
                    
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
                    
                    logits = logits.squeeze(-1)  # Ensure 1D
                    
                    # Check if logits has gradient - if not, there's a problem with model output
                    if not logits.requires_grad:
                        self.logger.error(f"CRITICAL: logits.requires_grad={logits.requires_grad} at batch {batch_idx}")
                        self.logger.error(f"  Model training mode: {self.model.training}")
                        self.logger.error(f"  torch.is_grad_enabled(): {torch.is_grad_enabled()}")
                        self.logger.error(f"  output type: {type(output)}")
                        # Check if any model parameters require grad
                        params_require_grad = [p.requires_grad for p in self.model.parameters()]
                        self.logger.error(f"  Model params requiring grad: {sum(params_require_grad)}/{len(params_require_grad)}")
                        raise RuntimeError(f"Cannot establish gradient connection for logits at batch {batch_idx}. Model may be detached or in eval mode.")
                    
                    loss = loss_fn(logits, status, time)
                    
                    # Verify loss has gradient connection - if not, it's a bug in loss function
                    if not loss.requires_grad:
                        self.logger.error(f"CRITICAL: loss.requires_grad={loss.requires_grad} at batch {batch_idx}")
                        self.logger.error(f"  logits.requires_grad={logits.requires_grad}, logits.shape={logits.shape}")
                        self.logger.error(f"  status.sum()={status.sum().item()}, batch_size={len(status)}")
                        raise RuntimeError(f"Loss function returned tensor without gradient connection at batch {batch_idx}")
                
                if fp16_scaler is not None:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                lr_scheduler(global_step)
                global_step += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / max(epoch_steps, 1)
            epoch_time = time_module.time() - epoch_start_time
            
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
            
            if self.writer:
                self.writer.add_scalar('Train/Loss', avg_loss, epoch)
            
            # Validation (skip first 2 epochs to allow model to warm up)
            if epoch > 1:
                self.model.eval()
                torch.set_grad_enabled(False)
                val_metrics = self.evaluate('val')
                torch.set_grad_enabled(True)
                self.model.train()
                
                # Save latest model after each validation
                latest_model_path = os.path.join(self.save_dir, f"latest_{self.model_type}.pth")
                torch.save(self.model.state_dict(), latest_model_path)
                self.logger.info(f"Saved latest model to {latest_model_path}")
                
                # Early stopping
                metric_value = val_metrics.get(f'val_{metric}', val_metrics.get('val_c_index', 0.0))
                # Convert to float to ensure it's a number, not a model object
                if isinstance(metric_value, (torch.Tensor, np.ndarray)):
                    metric_value = float(metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value)
                else:
                    metric_value = float(metric_value)
                # EarlyStoppingSurvival.__call__ signature: (val_loss, val_c_index, model)
                # val_loss is not used but required by signature, pass 0.0
                early_stopping(0.0, metric_value, self.model)
                if early_stopping.early_stop:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                # Save latest model even in first 2 epochs (before validation starts)
                latest_model_path = os.path.join(self.save_dir, f"latest_{self.model_type}.pth")
                torch.save(self.model.state_dict(), latest_model_path)
        
        # Save latest model at the end of training
        latest_model_path = os.path.join(self.save_dir, f"latest_{self.model_type}.pth")
        torch.save(self.model.state_dict(), latest_model_path)
        self.logger.info(f"Saved latest model to {latest_model_path}")
        
        # Load best model
        best_model_path = os.path.join(self.save_dir, f'best_{self.model_type}.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info("Loaded best model for final evaluation")
        elif hasattr(early_stopping, 'best_model_state') and early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            self.logger.info("Loaded best model from early stopping state")
        
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
        all_status = []
        all_time = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{prefix} Eval"):
                # Survival dataset now returns 7 items: features, coords, bag_sizes, status, time, patient_ids, slide_ids
                if len(batch) == 7:
                    features, coords, bag_sizes, status, time, patient_ids, slide_ids = batch
                else:
                    # Old format with 6 items
                    features, coords, bag_sizes, status, time, patient_ids = batch
                
                features = features.to(self.device)
                
                if hasattr(self.model, 'forward') and 'is_cox' in self.model.forward.__code__.co_varnames:
                    output = self.model(features, is_cox=True)
                else:
                    output = self.model(features)
                
                if isinstance(output, dict):
                    logits = output['logits']
                else:
                    logits = output
                
                preds = logits.squeeze(-1).cpu().numpy()
                
                # Flatten to ensure 1D arrays for concatenation
                all_preds.append(preds.flatten())
                all_status.append(status.cpu().numpy().flatten())
                all_time.append(time.cpu().numpy().flatten())
                # Handle patient_ids - could be list or single string
                if isinstance(patient_ids, (list, tuple)):
                    all_patient_ids.extend(patient_ids)
                else:
                    all_patient_ids.append(patient_ids)
        
        all_preds = np.concatenate(all_preds)
        all_status = np.concatenate(all_status)
        all_time = np.concatenate(all_time)
        
        # Calculate C-index
        risk_tensor = torch.tensor(all_preds, dtype=torch.float32)
        status_tensor = torch.tensor(all_status, dtype=torch.float32)
        time_tensor = torch.tensor(all_time, dtype=torch.float32)
        
        c_index = survival_c_index(risk_tensor, status_tensor, time_tensor, all_patient_ids)
        
        # Calculate event statistics
        num_events = int(all_status.sum())
        num_censored = int((1 - all_status).sum())
        event_rate = float(all_status.mean())
        mean_time = float(all_time.mean())
        median_time = float(np.median(all_time))
        
        metrics = {
            f"{split}_c_index": c_index if c_index is not None else 0.0,
            f"{split}_events": num_events,
            f"{split}_censored": num_censored,
            f"{split}_event_rate": event_rate,
            f"{split}_mean_time": mean_time,
            f"{split}_median_time": median_time,
        }
        
        # Log metrics
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Save results to CSV if test split
        if split == 'test':
            save_csv_path = os.path.join(self.save_dir, f"results_{self.model_type}.csv")
            results_df = pd.DataFrame({
                'sample_id': [f"sample_{i}" for i in range(len(all_preds))],
                'patient_id': all_patient_ids,
                'status': all_status.astype(int),
                'time': all_time,
                'risk_score': all_preds
            })
            results_df.to_csv(save_csv_path, index=False)
            self.logger.info(f"Results saved to {save_csv_path}")
        
        return metrics
    
    def save_training_config(self):
        """Save training configuration to JSON file"""
        actual_config = {
            "batch_size": self.config.get('batch_size', 32),
            "learning_rate": self.config.get('learning_rate', 1e-4),
            "batch_sampler": self.config.get('batch_sampler', None),  # None means random sampling
            "model_type": self.model_type,
            "num_epochs": self.config.get('num_epochs', 100),
            "warmup_epochs": self.config.get('warmup_epochs', 5),
            "weight_decay": self.config.get('weight_decay', 0.01),
            "dropout": self.config.get('dropout', 0.25),
            "patience": self.config.get('patience', 10),
            "hidden_dim": self.config.get('hidden_dim', 256),
            "feature_dimension": self.config.get('feature_dimension'),
            "max_seq_length": self.config.get('max_seq_length'),
            "metric": self.dataset_info.get('metric', 'c_index'),
            "survival_loss": self.survival_loss,
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

