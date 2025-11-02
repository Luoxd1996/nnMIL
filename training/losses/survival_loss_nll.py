import torch
import torch.nn as nn


class NLLSurvLoss(nn.Module):
    """
    Discrete-time negative log-likelihood survival loss (Zadeh & Schmid, 2020).
    Compatible with batch_size=1 since it operates per-sample with class (time-bin) dimension.
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, c):
        # Ensure 1D tensors for y and c even when coming as scalars
        if y.dim() == 0:
            y = y.view(1)
        if c.dim() == 0:
            c = c.view(1)
        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps, reduction=self.reduction)


def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    else:
        loss = neg_l

    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


