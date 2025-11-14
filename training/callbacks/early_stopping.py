"""
Early stopping callbacks for different task types.
"""
import os
import torch
import numpy as np


class EarlyStopping:
    """Early stopping with metric from plan file for classification"""
    def __init__(self, patience=7, verbose=False, delta=0, metric='bacc', save_dir=None, model_type=None, logger=None):
        """
        Args:
            patience: Early stopping patience
            verbose: Verbose output
            delta: Minimum change to qualify as improvement
            metric: Primary metric from plan file ('auc', 'bacc', 'f1', 'kappa', etc.)
            save_dir: Directory to save best model
            model_type: Model type name for saving
            logger: Optional logger for logging messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.model_type = model_type
        self.logger = logger
        
        # Use metric from plan file (no hardcoding)
        metric_lower = metric.lower()
        if 'kappa' in metric_lower:
            self.primary_metric = "KAPPA"
        elif 'auc' in metric_lower:
            self.primary_metric = "AUC"
        elif metric_lower in ['bacc', 'balanced_accuracy']:
            self.primary_metric = "BACC"
        elif 'f1' in metric_lower:
            self.primary_metric = "F1"
        else:
            # Default to BACC for classification
            self.primary_metric = "BACC"
        
        msg = f"EarlyStopping: Using {self.primary_metric} as primary metric (from plan: {metric})"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def __call__(self, val_loss, val_bacc, val_f1, val_auc, model, val_kappa=None):
        # Use metric from plan file
        if self.primary_metric == "KAPPA":
            score = val_kappa if val_kappa is not None else 0.0
        elif self.primary_metric == "AUC":
            score = val_auc
        elif self.primary_metric == "F1":
            score = val_f1
        else:
            # BACC or default
            score = val_bacc
        
        # Handle NaN/inf scores
        if np.isnan(score) or np.isinf(score):
            score = 0.0
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_bacc, val_f1, val_auc, model, val_kappa)
            msg = f'EarlyStopping: Initial {self.primary_metric} = {score:.4f}'
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        elif score <= self.best_score + self.delta:
            # Score did not improve (or improved less than delta)
            self.counter += 1
            msg = f'EarlyStopping counter: {self.counter}/{self.patience} ({self.primary_metric}: {score:.4f} <= {self.best_score:.4f} + {self.delta:.4f})'
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            if self.counter >= self.patience:
                self.early_stop = True
                msg = f'Early stopping triggered! No improvement for {self.patience} epochs.'
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
        else:
            # Score improved
            improvement = score - self.best_score
            old_score = self.best_score
            self.best_score = score
            self.save_checkpoint(val_loss, val_bacc, val_f1, val_auc, model, val_kappa)
            self.counter = 0
            msg = f'EarlyStopping: {self.primary_metric} improved from {old_score:.4f} to {self.best_score:.4f} (+{improvement:.4f}). Reset counter.'
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

    def save_checkpoint(self, val_loss, val_bacc, val_f1, val_auc, model, val_kappa=None):
        msg = f'Validation {self.primary_metric} improved. Saving model...'
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        self.best_model_state = model.state_dict().copy()
        
        # Save best model to file
        if self.save_dir and self.model_type:
            best_model_path = os.path.join(self.save_dir, f"best_{self.model_type}.pth")
            torch.save(model.state_dict(), best_model_path)
            msg = f'Saved best model to {best_model_path}'
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)


class RegressionEarlyStopping:
    """Early stopping for regression tasks using metric from plan file"""
    def __init__(self, patience=10, verbose=False, delta=0, metric='pearson', save_dir=None, model_type=None, logger=None):
        """
        Args:
            patience: Early stopping patience
            verbose: Verbose output
            delta: Minimum change to qualify as improvement
            metric: Primary metric from plan file ('pearson', 'r2', 'mse', etc.)
            save_dir: Directory to save best model
            model_type: Model type name for saving
            logger: Optional logger for logging messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.model_type = model_type
        self.logger = logger
        
        # Use metric from plan file
        metric_lower = metric.lower()
        if 'pearson' in metric_lower or 'corr' in metric_lower:
            self.primary_metric = "PEARSON"
        elif 'r2' in metric_lower or 'r_squared' in metric_lower:
            self.primary_metric = "R2"
        elif 'mse' in metric_lower:
            self.primary_metric = "MSE"
            # MSE is lower-is-better, so we'll handle it differently
            self.higher_is_better = False
        else:
            # Default to Pearson
            self.primary_metric = "PEARSON"
            self.higher_is_better = True
        
        # Most regression metrics are higher-is-better, except MSE
        if not hasattr(self, 'higher_is_better'):
            self.higher_is_better = True
        
        msg = f"RegressionEarlyStopping: Using {self.primary_metric} as primary metric (from plan: {metric})"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def __call__(self, val_mse, val_pearson, val_r2, model):
        # Use metric from plan file
        if self.primary_metric == "PEARSON":
            score = val_pearson
        elif self.primary_metric == "R2":
            score = val_r2
        elif self.primary_metric == "MSE":
            score = -val_mse  # Convert to higher-is-better for comparison
        else:
            score = val_pearson  # Default
        
        # Handle NaN/inf scores
        if np.isnan(score) or np.isinf(score):
            score = 0.0 if self.higher_is_better else -1e6
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_mse, val_pearson, val_r2, model)
            msg = f'RegressionEarlyStopping: Initial {self.primary_metric} = {score:.4f}'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
        elif score < self.best_score + self.delta:
            self.counter += 1
            msg = f'EarlyStopping counter: {self.counter}/{self.patience} ({self.primary_metric}: {score:.4f} < {self.best_score:.4f} + {self.delta:.4f})'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
            if self.counter >= self.patience:
                self.early_stop = True
                msg = f'Early stopping triggered! No improvement for {self.patience} epochs.'
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
        else:
            improvement = score - self.best_score
            old_score = self.best_score
            self.best_score = score
            self.save_checkpoint(val_mse, val_pearson, val_r2, model)
            self.counter = 0
            msg = f'RegressionEarlyStopping: {self.primary_metric} improved from {old_score:.4f} to {self.best_score:.4f} (+{improvement:.4f}). Reset counter.'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)

    def save_checkpoint(self, val_mse, val_pearson, val_r2, model):
        msg = f'Validation {self.primary_metric} improved. Saving model...'
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(msg)
        self.best_model_state = model.state_dict().copy()
        
        # Save best model to file
        if self.save_dir and self.model_type:
            best_model_path = os.path.join(self.save_dir, f"best_{self.model_type}.pth")
            torch.save(model.state_dict(), best_model_path)
            msg = f'Saved best model to {best_model_path}'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)


class EarlyStoppingSurvival:
    """Early stopping for survival analysis using metric from plan file"""
    def __init__(self, patience=10, verbose=False, delta=0, metric='c_index', save_dir=None, model_type=None, logger=None):
        """
        Args:
            patience: Early stopping patience
            verbose: Verbose output
            delta: Minimum change to qualify as improvement
            metric: Primary metric from plan file ('c_index', 'cindex', etc.)
            save_dir: Directory to save best model
            model_type: Model type name for saving
            logger: Optional logger
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.model_type = model_type
        self.logger = logger
        
        # Use metric from plan file (usually c_index for survival)
        metric_lower = metric.lower()
        if 'c_index' in metric_lower or 'cindex' in metric_lower:
            self.primary_metric = "C-index"
        else:
            # Default to C-index for survival
            self.primary_metric = "C-index"
        
        msg = f"EarlyStopping: Using {self.primary_metric} as primary metric for survival analysis (from plan: {metric})"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def __call__(self, val_loss, val_c_index, model):
        score = val_c_index
        
        # Handle NaN/inf scores
        if np.isnan(score) or np.isinf(score):
            score = 0.0
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_c_index, model)
            msg = f'EarlyStopping: Initial {self.primary_metric} = {score:.4f}'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
        elif score < self.best_score + self.delta:
            # No improvement (score < best_score + delta) or worse
            self.counter += 1
            msg = f'EarlyStopping counter: {self.counter}/{self.patience} ({self.primary_metric}: {score:.4f} < {self.best_score:.4f} + {self.delta:.4f})'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
            if self.counter >= self.patience:
                self.early_stop = True
                msg = f'Early stopping triggered! No improvement for {self.patience} epochs.'
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
        else:
            # Improvement (score > best_score + delta)
            improvement = score - self.best_score
            old_score = self.best_score
            self.best_score = score
            self.save_checkpoint(val_loss, val_c_index, model)
            self.counter = 0
            msg = f'EarlyStopping: {self.primary_metric} improved from {old_score:.4f} to {self.best_score:.4f} (+{improvement:.4f}). Reset counter.'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)

    def save_checkpoint(self, val_loss, val_c_index, model):
        msg = f'Validation {self.primary_metric} improved ({val_c_index:.4f}). Saving model...'
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(msg)
        self.best_model_state = model.state_dict().copy()
        
        # Save best model to file
        if self.save_dir and self.model_type:
            best_model_path = os.path.join(self.save_dir, f"best_{self.model_type}.pth")
            torch.save(model.state_dict(), best_model_path)
            msg = f'Saved best model to {best_model_path}'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
    
    def load_best_model(self, model):
        """Load the best model weights"""
        if hasattr(self, 'best_model_state'):
            model.load_state_dict(self.best_model_state)
            return True
        return False



