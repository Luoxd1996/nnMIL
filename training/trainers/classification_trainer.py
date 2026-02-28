#!/usr/bin/env python3
"""
Classification Trainer for nnMIL
Handles classification tasks (binary and multi-class)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nnMIL.training.trainers.base_trainer import BaseTrainer
from nnMIL.network_architecture.model_factory import create_mil_model
from nnMIL.utilities.utils import get_eval_metrics, cosine_lr
from nnMIL.utilities.plan_loader import create_dataset_from_plan
from nnMIL.training.samplers.classification_sampler import BalancedBatchSampler, AUCBatchSampler
from nnMIL.training.callbacks.early_stopping import EarlyStopping


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification tasks"""
    
    def __init__(self, plan_path, model_type, fold=None, **kwargs):
        super().__init__(plan_path, model_type, fold, **kwargs)
        
        # Classification-specific initialization
        self.num_classes = self.config.get('num_classes', 2)
        self.logger.info(f"Number of classes: {self.num_classes}")
    
    def create_model(self):
        """Create classification model"""
        self.model = create_mil_model(
            model_type=self.model_type,
            input_dim=self.config['feature_dimension'],
            hidden_dim=self.config['hidden_dim'],
            num_classes=self.num_classes,
            dropout=self.config['dropout']
        )
        self.model = self.model.to(self.device)
        self.logger.info(f"Model created: {self.model_type}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        return self.model
    
    def create_data_loaders(self):
        """Create data loaders for classification task"""
        self.logger.info("Loading datasets from plan file...")
        
        train_dataset = create_dataset_from_plan(self.plan_path, split='train', fold=self.fold)
        val_dataset = create_dataset_from_plan(self.plan_path, split='val', fold=self.fold)
        test_dataset = create_dataset_from_plan(self.plan_path, split='test', fold=self.fold)
        
        self.logger.info(f"Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Update num_classes from dataset if not in config
        if self.num_classes == 2 and hasattr(train_dataset, 'num_classes'):
            self.num_classes = train_dataset.num_classes
        
        # Determine batch sampler
        batch_sampler_type = self.config.get('batch_sampler', 'random')
        batch_size = self.config.get('batch_size', 32)
        
        # Default collate function for fixed-length sequences
        def default_collate_fn(batch):
            features_list = [item[0] for item in batch]
            coords_list = [item[1] for item in batch]
            bag_sizes = [item[2] for item in batch]
            labels = [item[3] for item in batch]
            
            batch_features = torch.stack(features_list)
            batch_coords = torch.stack(coords_list)
            batch_bag_sizes = torch.stack(bag_sizes)
            batch_labels = torch.stack(labels)
            
            if len(batch[0]) == 6:
                slide_ids = [item[4] for item in batch]
                datasets = [item[5] for item in batch]
                return batch_features, batch_coords, batch_bag_sizes, batch_labels, slide_ids, datasets
            else:
                return batch_features, batch_coords, batch_bag_sizes, batch_labels
        
        # Create train loader with appropriate sampler
        if batch_sampler_type == 'balanced' and batch_size > 1:
            train_sampler = BalancedBatchSampler(train_dataset, batch_size, shuffle=True, seed=self.seed)
            self.train_loader = DataLoader(
                train_dataset, batch_sampler=train_sampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
                collate_fn=default_collate_fn
            )
            self.logger.info("Using BalancedBatchSampler")
        elif batch_sampler_type == 'auc' and batch_size > 1:
            train_sampler = AUCBatchSampler(train_dataset, batch_size, shuffle=True, seed=self.seed)
            self.train_loader = DataLoader(
                train_dataset, batch_sampler=train_sampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
                collate_fn=default_collate_fn
            )
            self.logger.info("Using AUCBatchSampler")
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
        
        # Create val and test loaders (deterministic)
        def _worker_init_fn(worker_id):
            np.random.seed(self.seed + worker_id)
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=4,
            worker_init_fn=_worker_init_fn
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4,
            worker_init_fn=_worker_init_fn
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
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Setup LR scheduler
        num_epochs = self.config.get('num_epochs', 100)
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = len(self.train_loader) * self.config.get('warmup_epochs', 5)
        lr_scheduler = cosine_lr(optimizer, self.config.get('learning_rate', 3e-4), warmup_steps, total_steps)
        
        # Setup early stopping
        metric = self.dataset_info.get('metric', 'bacc')
        early_stopping = EarlyStopping(
            patience=self.config.get('patience', 10),
            verbose=True,
            metric=metric,
            save_dir=self.save_dir,
            model_type=self.model_type,
            logger=self.logger
        )
        
        # Setup mixed precision
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        fp16_scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None
        
        # Training loop
        self.model.train()
        global_step = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
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
                if len(batch) == 6:
                    features, coords, bag_sizes, labels, _, _ = batch
                else:
                    features, coords, bag_sizes, labels = batch
                
                features = features.to(self.device)
                coords = coords.to(self.device)
                bag_sizes = bag_sizes.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type, dtype=torch.bfloat16):
                    output = self.model(features)
                    
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
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
            
            # Save latest model after each validation
            latest_model_path = os.path.join(self.save_dir, f"latest_{self.model_type}.pth")
            torch.save(self.model.state_dict(), latest_model_path)
            self.logger.info(f"Saved latest model to {latest_model_path}")
            
            # Early stopping - extract metrics from val_metrics
            # Note: evaluate() returns prefixed metrics like "val_val/bacc" (split='val' + prefix='val')
            # Check both prefixed and unprefixed keys for compatibility
            val_loss = val_metrics.get('val_val/loss', val_metrics.get('val/loss', 0.0))
            val_bacc = val_metrics.get('val_val/bacc', val_metrics.get('val/bacc', 0.0))
            val_f1 = val_metrics.get('val_val/weighted_f1', val_metrics.get('val/weighted_f1', 0.0))
            val_auc = val_metrics.get('val_val/auroc', val_metrics.get('val/auroc', 0.0))
            val_kappa = val_metrics.get('val_val/kappa', val_metrics.get('val/kappa', None))
            
            early_stopping(val_loss, val_bacc, val_f1, val_auc, self.model, val_kappa=val_kappa)
            if early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save latest model at the end of training
        latest_model_path = os.path.join(self.save_dir, f"latest_{self.model_type}.pth")
        torch.save(self.model.state_dict(), latest_model_path)
        self.logger.info(f"Saved latest model to {latest_model_path}")
        
        # Load best model
        best_model_path = os.path.join(self.save_dir, f'best_{self.model_type}.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info("Loaded best model for final evaluation")
        elif hasattr(early_stopping, 'best_model_state'):
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
        
        all_logits = []
        all_labels = []
        all_probs = []
        all_slide_ids = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{prefix} Eval"):
                if len(batch) == 6:
                    features, coords, bag_sizes, labels, slide_ids, _ = batch
                else:
                    features, coords, bag_sizes, labels = batch
                    slide_ids = None
                
                features = features.to(self.device)
                coords = coords.to(self.device)
                bag_sizes = bag_sizes.to(self.device)
                
                output = self.model(features)
                
                if isinstance(output, dict):
                    logits = output['logits']
                else:
                    logits = output
                
                probs = F.softmax(logits, dim=1)
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                if slide_ids:
                    all_slide_ids.extend(slide_ids)
        
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        # Get predictions from probabilities (argmax)
        all_preds = np.argmax(all_probs, axis=1)
        
        # Calculate metrics
        # Get unique classes from labels (or use self.num_classes to create range)
        if hasattr(self, 'num_classes') and self.num_classes is not None:
            unique_classes = list(range(self.num_classes))
        else:
            unique_classes = sorted(np.unique(all_labels).tolist())
        
        metrics = get_eval_metrics(
            targets_all=all_labels,
            preds_all=all_preds,
            probs_all=all_probs,
            unique_classes=unique_classes,
            prefix=split
        )
        
        # Add prefix
        prefixed_metrics = {f"{split}_{k}": v for k, v in metrics.items()}
        
        # Log metrics
        for key, value in prefixed_metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        # Save results to CSV if test split
        if split == 'test':
            import pandas as pd
            save_csv_path = os.path.join(self.save_dir, f"results_{self.model_type}.csv")
            
            # Create DataFrame with required columns
            results_data = {
                'slide_id': all_slide_ids if all_slide_ids else [f"sample_{i}" for i in range(len(all_labels))],
                'patient_id': [None] * len(all_labels),  # Will be filled if available
                'label': all_labels,
                'prediction': all_preds
            }
            
            # Add probability columns for each class
            if self.num_classes == 2:
                results_data['probability_class_0'] = all_probs[:, 0]
                results_data['probability_class_1'] = all_probs[:, 1]
            else:
                for i in range(self.num_classes):
                    results_data[f'probability_class_{i}'] = all_probs[:, i]
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(save_csv_path, index=False)
            self.logger.info(f"Results saved to {save_csv_path}")
        
        return prefixed_metrics
    
    def save_training_config(self):
        """Save training configuration to JSON file"""
        import json
        
        # Prepare actual configuration
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
            "metric": self.dataset_info.get('metric', 'bacc'),
            "num_classes": self.num_classes,
        }
        
        # Prepare dataset info
        dataset_info_dict = {
            "task_type": self.dataset_info.get('task_type'),
            "dataset_name": os.path.basename(os.path.dirname(self.plan_path)),
            "plan_path": self.plan_path,
            "fold": self.fold,
            "evaluation_setting": self.evaluation_setting,
        }
        
        # Save to JSON
        config_path = os.path.join(self.save_dir, 'training_config.json')
        config_dict = {
            "dataset_info": dataset_info_dict,
            "actual_configuration": actual_config,
            "random_seed": self.seed,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Training configuration saved to {config_path}")
