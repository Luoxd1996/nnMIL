"""
Exponential Moving Average (EMA) for Model Weights
Maintains EMA model starting from a specified epoch.
"""
import os
import torch
import torch.nn as nn
import copy


class EMAFromEpoch:
    """
    Exponential Moving Average starting from a specified epoch.
    
    This class maintains an EMA model that starts averaging weights
    from a specified epoch onwards, updating every iteration (batch).
    """
    
    def __init__(self, model, start_epoch=10, decay=0.999, steps_per_epoch=None, logger=None):
        """
        Args:
            model: The model to average weights for
            start_epoch: Start EMA from this epoch (0-indexed, so epoch 10 means after 10 epochs)
            decay: EMA decay factor (default: 0.999, higher = more smoothing)
            steps_per_epoch: Number of iterations per epoch (for calculating start step)
            logger: Optional logger for logging messages
        """
        self.model = model
        self.start_epoch = start_epoch
        self.decay = decay
        self.steps_per_epoch = steps_per_epoch
        self.logger = logger
        
        # EMA state
        self.ema_model = None
        self.ema_started = False
        self.start_step = None  # Can be overridden (e.g., when resetting at new best)
        self.min_start_step = None  # Determined from start_epoch and steps_per_epoch
        
        if self.steps_per_epoch is not None:
            self.min_start_step = self.start_epoch * self.steps_per_epoch
            # Default start step follows warmup configuration
            self.start_step = self.min_start_step
        
    def set_steps_per_epoch(self, steps_per_epoch):
        """Set steps per epoch and calculate start step"""
        self.steps_per_epoch = steps_per_epoch
        self.min_start_step = self.start_epoch * steps_per_epoch
        if self.start_step is None:
            self.start_step = self.min_start_step
        
    def update(self, current_model_state, global_step):
        """
        Update EMA model with current model weights.
        Called every iteration (batch).
        
        Args:
            current_model_state: Current model state dict
            global_step: Current global step (iteration number, 0-indexed)
        """
        # Determine the earliest step when EMA should start
        start_threshold = self.start_step
        if start_threshold is None and self.min_start_step is not None:
            start_threshold = self.min_start_step
        
        # Only start EMA after start_step
        if start_threshold is not None and global_step < start_threshold:
            return
        
        # Initialize EMA on first update
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.load_state_dict(current_model_state)
            self.ema_started = True
            
            # Calculate which epoch we're in
            current_epoch = global_step // self.steps_per_epoch if self.steps_per_epoch else None
            msg = f"EMA initialized at step {global_step}"
            if current_epoch is not None:
                msg += f" (epoch {current_epoch+1})"
            msg += f" (decay={self.decay})"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            return
        
        # Update EMA: ema_model = decay * ema_model + (1 - decay) * current_model
        ema_state = self.ema_model.state_dict()
        with torch.no_grad():
            for param_name, current_val in current_model_state.items():
                if param_name in ema_state:
                    ema_param = ema_state[param_name]
                    if isinstance(ema_param, torch.Tensor) and isinstance(current_val, torch.Tensor):
                        # EMA update: ema_param = decay * ema_param + (1 - decay) * current_val
                        ema_param.mul_(self.decay).add_(current_val, alpha=1 - self.decay)
        
        # Also update buffers (like BatchNorm running stats) if they exist
        # Note: SimpleMIL doesn't use BatchNorm, but this handles other models
        model_buffers = {name: buf for name, buf in self.model.named_buffers()}
        if model_buffers:
            for buffer_name, current_buffer in model_buffers.items():
                if buffer_name in ema_state:
                    ema_buffer = ema_state[buffer_name]
                    if isinstance(ema_buffer, torch.Tensor) and isinstance(current_buffer, torch.Tensor):
                        ema_buffer.mul_(self.decay).add_(current_buffer, alpha=1 - self.decay)
        
        # Load updated state back to model
        self.ema_model.load_state_dict(ema_state)
    
    def reset_to_model_state(self, current_model_state, global_step, reason=None):
        """
        Reinitialize EMA from the provided model state and resume updating from current step.
        Useful when we want EMA to track from the most recent best checkpoint onward.
        """
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(self.model)
        self.ema_model.load_state_dict(current_model_state)
        self.ema_started = True
        # Ensure EMA does not start before the configured warmup period
        min_step = self.min_start_step if self.min_start_step is not None else 0
        self.start_step = max(global_step, min_step)
        
        if self.logger:
            msg = f"EMA reset at step {global_step} (start_step={self.start_step}, decay={self.decay})"
            if reason:
                msg += f" due to {reason}"
            self.logger.info(msg)
        else:
            print(f"EMA reset at step {global_step} (start_step={self.start_step}, decay={self.decay})" + (f" due to {reason}" if reason else ""))
    
    def get_ema_model(self):
        """Get the EMA model"""
        return self.ema_model
    
    def save_ema_model(self, save_path):
        """Save EMA model to file"""
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), save_path)
            msg = f"EMA model saved to {save_path} (decay={self.decay})"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

