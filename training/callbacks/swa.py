"""
Stochastic Weight Averaging (SWA) from Best Checkpoint
Maintains SWA model starting from the best checkpoint.
"""
import os
import torch
import torch.nn as nn
import copy


class SWAFromBest:
    """
    Stochastic Weight Averaging starting from best checkpoint.
    
    This class maintains an SWA model that starts averaging weights
    from the best checkpoint onwards, updating every N epochs.
    """
    
    def __init__(self, model, update_freq=2, start_after_best=True, 
                 min_epoch=None, min_c_index=None, logger=None):
        """
        Args:
            model: The model to average weights for
            update_freq: Update SWA every N epochs (default: 2)
            start_after_best: If True, only start SWA after best checkpoint is found
            min_epoch: Minimum epoch to start SWA (e.g., wait for warmup to finish)
            min_c_index: Minimum C-index threshold to start SWA (e.g., 0.5)
            logger: Optional logger for logging messages
        """
        self.model = model
        self.update_freq = update_freq
        self.start_after_best = start_after_best
        self.min_epoch = min_epoch  # Wait until this epoch to start SWA
        self.min_c_index = min_c_index  # Require C-index above this threshold
        self.logger = logger
        
        # SWA state
        self.swa_model = None
        self.swa_n = 0  # Number of models averaged
        self.best_found = False
        self.epochs_since_best = 0
        self.last_update_epoch = -1
        
        # Track best SWA checkpoint
        self.best_swa_c_index = None
        self.best_swa_state = None
        
    def initialize_from_best(self, best_model_state, epoch=None, val_c_index=None):
        """
        Initialize SWA model from best checkpoint.
        Only initializes once - subsequent best checkpoints do not reset SWA.
        
        Args:
            best_model_state: State dict of the best model
            epoch: Current epoch number (for min_epoch check)
            val_c_index: Current validation C-index (for min_c_index check)
        """
        # Only initialize once - SWA should continue accumulating even if best updates
        if self.swa_model is None:
            # Check if we should wait (min_epoch or min_c_index threshold)
            should_wait = False
            wait_reason = []
            
            if self.min_epoch is not None and epoch is not None:
                if epoch < self.min_epoch:
                    should_wait = True
                    wait_reason.append(f"epoch < {self.min_epoch}")
            
            if self.min_c_index is not None and val_c_index is not None:
                if val_c_index < self.min_c_index:
                    should_wait = True
                    wait_reason.append(f"c_index < {self.min_c_index}")
            
            if should_wait:
                msg = f"SWA initialization delayed: {', '.join(wait_reason)} (epoch={epoch}, val_c_index={val_c_index:.4f})"
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
                return False
            
            # Create SWA model as a copy of the current model
            self.swa_model = copy.deepcopy(self.model)
            # Initialize with best checkpoint weights
            self.swa_model.load_state_dict(best_model_state)
            self.swa_n = 1
            self.best_found = True
            self.epochs_since_best = 0
            
            msg = f"SWA initialized from best checkpoint (n={self.swa_n}"
            if epoch is not None:
                msg += f", epoch={epoch+1}"
            if val_c_index is not None:
                msg += f", val_c_index={val_c_index:.4f}"
            msg += ")"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            return True
        else:
            # SWA already initialized - just log that best checkpoint updated
            # SWA will continue accumulating from previous state
            msg = f"Best checkpoint updated, but SWA continues accumulating (n={self.swa_n})"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            return False
    
    def update(self, current_model_state, epoch, val_c_index=None):
        """
        Update SWA model with current model weights.
        
        Args:
            current_model_state: Current model state dict
            epoch: Current epoch number
            val_c_index: Optional validation C-index for tracking best SWA
        """
        # Only update if best checkpoint has been found (if start_after_best=True)
        if self.start_after_best and not self.best_found:
            return
        
        # Only update every update_freq epochs
        if epoch - self.last_update_epoch < self.update_freq:
            return
        
        if self.swa_model is None:
            # Initialize SWA from current model if not initialized
            self.swa_model = copy.deepcopy(self.model)
            self.swa_model.load_state_dict(current_model_state)
            self.swa_n = 1
            self.last_update_epoch = epoch
            
            msg = f"SWA initialized from epoch {epoch+1} (n={self.swa_n})"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            return
        
        # Update SWA: swa_model = (n * swa_model + current_model) / (n + 1)
        # Use state_dict for reliable parameter matching
        swa_state = self.swa_model.state_dict()
        with torch.no_grad():
            for param_name, current_val in current_model_state.items():
                if param_name in swa_state:
                    swa_param = swa_state[param_name]
                    if isinstance(swa_param, torch.Tensor) and isinstance(current_val, torch.Tensor):
                        # Update: swa_param = (n * swa_param + current_val) / (n + 1)
                        swa_param.mul_(self.swa_n).add_(current_val).div_(self.swa_n + 1)
        
        # Load updated state back to model
        self.swa_model.load_state_dict(swa_state)
        
        self.swa_n += 1
        self.last_update_epoch = epoch
        
        # Track best SWA checkpoint based on validation C-index
        if val_c_index is not None:
            if self.best_swa_c_index is None or val_c_index > self.best_swa_c_index:
                self.best_swa_c_index = val_c_index
                self.best_swa_state = copy.deepcopy(self.swa_model.state_dict())
        
        msg = f"SWA updated at epoch {epoch+1} (n={self.swa_n})"
        if val_c_index is not None:
            msg += f", val_c_index={val_c_index:.4f}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def get_swa_model(self):
        """Get the SWA model"""
        return self.swa_model
    
    def get_best_swa_model(self):
        """Get the best SWA model based on validation C-index"""
        if self.best_swa_state is not None:
            best_swa_model = copy.deepcopy(self.model)
            best_swa_model.load_state_dict(self.best_swa_state)
            return best_swa_model
        return self.swa_model
    
    def save_swa_model(self, save_path):
        """Save SWA model to file"""
        if self.swa_model is not None:
            torch.save(self.swa_model.state_dict(), save_path)
            msg = f"SWA model saved to {save_path} (n={self.swa_n})"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
    
    def save_best_swa_model(self, save_path):
        """Save best SWA model to file"""
        if self.best_swa_state is not None:
            torch.save(self.best_swa_state, save_path)
            msg = f"Best SWA model saved to {save_path} (val_c_index={self.best_swa_c_index:.4f}, n={self.swa_n})"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        elif self.swa_model is not None:
            self.save_swa_model(save_path)

