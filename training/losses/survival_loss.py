import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import concordance_index_censored from scikit-survival
try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    # Fallback to lifelines if scikit-survival is not available
    try:
        from lifelines.utils import concordance_index
        
        def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-08):
            """Wrapper for lifelines concordance_index"""
            import numpy as np
            
            # Convert to numpy arrays
            event_indicator = np.asarray(event_indicator)
            event_time = np.asarray(event_time)
            estimate = np.asarray(estimate)
            
            # Use lifelines implementation (note: lifelines expects risk scores in opposite direction)
            return concordance_index(event_time, -estimate, event_indicator)
            
    except ImportError:
        # Manual implementation as last resort
        def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-08):
            """Manual implementation of concordance index for survival analysis"""
            import numpy as np
            
            # Convert to numpy arrays
            event_indicator = np.asarray(event_indicator)
            event_time = np.asarray(event_time)
            estimate = np.asarray(estimate)
            
            n = len(estimate)
            concordant_pairs = 0
            total_pairs = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    # Skip if both samples are censored
                    if event_indicator[i] == 0 and event_indicator[j] == 0:
                        continue
                    
                    # Determine which sample has the event
                    if event_indicator[i] == 1 and event_indicator[j] == 1:
                        # Both events: compare times
                        if event_time[i] < event_time[j]:
                            # i has shorter survival, should have higher risk
                            if estimate[i] > estimate[j]:
                                concordant_pairs += 1
                            total_pairs += 1
                        elif event_time[j] < event_time[i]:
                            # j has shorter survival, should have higher risk
                            if estimate[j] > estimate[i]:
                                concordant_pairs += 1
                            total_pairs += 1
                    elif event_indicator[i] == 1 and event_indicator[j] == 0:
                        # i has event, j is censored
                        if event_time[i] <= event_time[j]:
                            # i has shorter survival, should have higher risk
                            if estimate[i] > estimate[j]:
                                concordant_pairs += 1
                            total_pairs += 1
                    elif event_indicator[i] == 0 and event_indicator[j] == 1:
                        # j has event, i is censored
                        if event_time[j] <= event_time[i]:
                            # j has shorter survival, should have higher risk
                            if estimate[j] > estimate[i]:
                                concordant_pairs += 1
                            total_pairs += 1
            
            if total_pairs == 0:
                return 0.5  # Random performance
            
            return concordant_pairs / total_pairs


class SurvivalLoss(nn.Module):
    """
    Survival loss functions for MIL models adapted to survival analysis
    """
    def __init__(self, loss_type="cox", reduction="mean", ties="breslow", **kwargs):
        super(SurvivalLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.ties = ties  # 'breslow' or 'efron'
        
        if self.loss_type == "cox":
            self.loss_fn = self._cox_loss
        elif self.loss_type == "mse":
            self.loss_fn = self._mse_loss
        elif self.loss_type == "mae":
            self.loss_fn = self._mae_loss
        elif self.loss_type == "nllsurv":
            # Placeholder; not used via this wrapper's forward.
            # Use NLLSurvLoss directly in training for clarity.
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, logits, status, time):
        """
        Compute survival loss
        
        Args:
            logits: Model output logits (batch_size, 1) or (batch_size,)
            status: Event indicator (batch_size,) - 0=censored, 1=event
            time: Survival time (batch_size,)
            
        Returns:
            loss: Survival loss value
        """
        if self.loss_type == "nllsurv":
            raise RuntimeError("Use NLLSurvLoss(h, y, c) directly for 'nllsurv'.")
        return self.loss_fn(logits, status, time)
    
    def _cox_loss(self, logits, status, time, strata=None, eps: float = 1e-12):
        # Check if logits has gradient - if not, raise error (should never happen if called correctly)
        if not logits.requires_grad:
            raise RuntimeError(f"logits.requires_grad=False in _cox_loss - logits must come from model output with gradient tracking")
        
        if logits.dim() > 1:
            logits = logits.squeeze(-1)

        # Keep original logits for gradient connection fallback - MUST keep reference before any masking
        # Use identity operation to maintain reference without breaking gradient
        original_logits = logits + 0.0  # Identity operation that preserves gradient

        # Filter NaN/Inf values
        mask = torch.isfinite(logits) & torch.isfinite(time)
        if strata is not None:
            mask = mask & torch.isfinite(strata.float())
        
        # If all samples are filtered out, return zero connected to original logits
        if not mask.any():
            return original_logits.sum() * 0.0
        
        logits = logits[mask]
        status = status[mask]
        time   = time[mask]
        strata = (strata[mask] if strata is not None else None)

        # No events: return zero with gradient connection
        if status.sum() == 0:
            # Return a zero tensor with requires_grad=True to maintain gradient connection
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if strata is None:
            loss, n_events = self._cox_one_stratum(logits, status, time, eps, original_logits)
        else:
            # Initialize loss using original method
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            n_events = 0
            # Compute loss for each stratum independently and sum
            for sid in torch.unique(strata):
                idx = (strata == sid)
                l, ne = self._cox_one_stratum(logits[idx], status[idx], time[idx], eps, original_logits)
                loss = loss + l
                n_events += ne

        if self.reduction == "mean":
            return loss / (float(n_events) + eps)
        elif self.reduction == "sum":
            return loss
        else:
            # 'none': return scalar (typically not used for Cox loss)
            return loss

    def _cox_one_stratum(self, logits, status, time, eps: float, original_logits_for_grad=None):
        """
        Compute Cox partial log-likelihood for one stratum
        """
        # Use provided original_logits_for_grad, or fallback to logits
        if original_logits_for_grad is None:
            original_logits_for_grad = logits
        
        # Sort by time in descending order (latest -> earliest), risk set can be computed via prefix sum
        order = torch.argsort(time, descending=True)
        x = logits[order]
        e = status[order].long()
        t = time[order]

        exp_x = torch.exp(x)
        # cumsum doesn't have deterministic CUDA implementation
        # With warn_only=True in base_trainer, this will only warn, not error
        cum_exp = torch.cumsum(exp_x, dim=0)  # Risk set denominator (without Efron adjustment)

        # Group samples with the same unique time into consecutive segments [s:ed]
        uniq_t, counts = torch.unique_consecutive(t, return_counts=True)
        # cumsum doesn't have deterministic CUDA implementation
        # With warn_only=True in base_trainer, this will only warn, not error
        ends = torch.cumsum(counts, dim=0) - 1
        starts = ends - counts + 1

        n_events = int(e.sum().item())

        if n_events == 0:
            # Return a zero tensor connected to the computation graph
            # Use original approach: create tensor with requires_grad=True on same device
            return torch.tensor(0.0, device=x.device, requires_grad=True), 0

        # Use original approach: initialize with requires_grad=True and accumulate
        # This ensures gradient connection through addition operations
        total_nll = torch.tensor(0.0, device=x.device, requires_grad=True)

        for s, ed in zip(starts.tolist(), ends.tolist()):
            # All samples at this time point
            e_seg = e[s:ed+1]
            m = int(e_seg.sum().item())  # Number of events at this time (ties)
            if m == 0:
                continue

            x_events_sum = x[s:ed+1][e_seg.bool()].sum()
            denom = cum_exp[ed] + eps

            # Choose calculation method based on ties parameter
            if m == 1:
                # Single event: Breslow and Efron are equivalent
                nll = -(x_events_sum - torch.log(denom))
            else:
                # Multiple events: choose method based on ties parameter
                if self.ties == "efron":
                    # Efron method for handling ties
                    exp_x_events_sum = exp_x[s:ed+1][e_seg.bool()].sum()
                    logs = []
                    # Efron: gradually subtract contributions from tied events
                    for l in range(m):
                        adj = denom - (l / m) * exp_x_events_sum
                        logs.append(torch.log(adj + eps))
                    nll = -(x_events_sum - torch.stack(logs).sum())
                else:
                    # Breslow method (default)
                    nll = -(x_events_sum - m * torch.log(denom))

            # Accumulate using addition - this preserves gradient connection
            total_nll = total_nll + nll

        return total_nll, n_events
    
    def _mse_loss(self, logits, status, time):
        """
        Mean Squared Error loss - treat as regression problem
        """
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        
        # Use log-transformed time for better numerical stability
        log_time = torch.log(time + 1e-8)
        
        # Only use events for training (status=1)
        event_mask = (status == 1)
        
        if not event_mask.any():
            return logits.sum() * 0.0
        
        event_logits = logits[event_mask]
        event_log_time = log_time[event_mask]
        
        return F.mse_loss(event_logits, event_log_time)
    
    def _mae_loss(self, logits, status, time):
        """
        Mean Absolute Error loss - treat as regression problem
        """
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        
        # Use log-transformed time for better numerical stability
        log_time = torch.log(time + 1e-8)
        
        # Only use events for training (status=1)
        event_mask = (status == 1)
        
        if not event_mask.any():
            return logits.sum() * 0.0
        
        event_logits = logits[event_mask]
        event_log_time = log_time[event_mask]
        
        return F.l1_loss(event_logits, event_log_time)


def survival_c_index(logits, status, time, patient_ids=None):
    """
    Compute Harrell's C-index for survival prediction
    
    Args:
        logits: Model output logits (batch_size, 1) or (batch_size,)
        status: Event indicator (batch_size,) - 0=censored, 1=event
        time: Survival time (batch_size,)
        patient_ids: Patient IDs (batch_size,) - if provided, aggregate at patient level
        
    Returns:
        c_index: Concordance index (0-1, higher is better)
    """
    # Convert inputs to numpy (support both torch.Tensor and numpy.ndarray)
    if isinstance(logits, torch.Tensor):
        risk_scores = logits.detach().cpu().numpy()
    else:
        risk_scores = np.asarray(logits)
    
    # Ensure risk_scores is 1D - flatten regardless of initial shape
    if risk_scores.ndim > 1:
        risk_scores = risk_scores.flatten()
    
    if isinstance(status, torch.Tensor):
        status = status.detach().cpu().numpy().astype(bool)
    else:
        status = np.asarray(status).astype(bool)
    
    if isinstance(time, torch.Tensor):
        time = time.detach().cpu().numpy()
    else:
        time = np.asarray(time)
    
    # If patient_ids provided, aggregate at patient level
    if patient_ids is not None:
        patient_ids = np.array(patient_ids)
        # Ensure patient_ids is 1D
        if patient_ids.ndim > 1:
            patient_ids = patient_ids.flatten()
        
        # Group by patient_id and take mean risk score
        unique_patients = np.unique(patient_ids)
        patient_risk_scores = []
        patient_status = []
        patient_time = []
        
        for patient_id in unique_patients:
            patient_mask = (patient_ids == patient_id)
            patient_risk_scores.append(np.mean(risk_scores[patient_mask]))
            # For patient-level, use the first occurrence (should be same for all samples from same patient)
            patient_status.append(status[patient_mask][0])
            patient_time.append(time[patient_mask][0])
        
        risk_scores = np.array(patient_risk_scores)
        status = np.array(patient_status)
        time = np.array(patient_time)
    
    # Use concordance_index_censored
    result = concordance_index_censored(status, time, risk_scores, tied_tol=1e-08)
    # Handle different return types: sksurv returns tuple, lifelines returns scalar
    if isinstance(result, tuple):
        c_index = result[0]
    else:
        c_index = result
    
    return c_index