"""
Regression loss functions for MIL training.
"""
import torch
import torch.nn as nn


class CombinedRegressionLoss(nn.Module):
    """Combined loss function: MSE + L1 + Smooth L1"""
    def __init__(self, mse_weight=1.0, l1_weight=0.1, smooth_l1_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.smooth_l1_weight = smooth_l1_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=1.0)
    
    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        l1 = self.l1_loss(predictions, targets)
        smooth_l1 = self.smooth_l1_loss(predictions, targets)
        
        total_loss = (self.mse_weight * mse + 
                     self.l1_weight * l1 + 
                     self.smooth_l1_weight * smooth_l1)
        
        return total_loss

