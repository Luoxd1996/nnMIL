import torch
import torch.nn as nn
import torch.nn.functional as F
            
class SimpleMIL(nn.Module):
    """
    Minimal version:
    - Training: randomly sample 256 dims from D dims for attention (recorded in self.last_train_idx), 
      pooling and classification use full-dim x
    - Validation: evenly divide D dims into 256-dim chunks, average logits across chunks 
      (chunk count recorded in self.eval_chunks)
    """
    COVER_SHUFFLE = True
    # COVER_SHUFFLE = False
    COVER_SEED = 42

    def __init__(self, input_dim=768, hidden_dim=256, pred_num=1,
                 activation='softmax', dropout=False):
        super().__init__()
        self.D = input_dim
        self.H = hidden_dim
        self.pred_num = pred_num
        self.act = activation
        # self.norm = nn.LayerNorm(self.D)

        self.V = nn.Linear(self.D, self.H)    # path a
        self.U = nn.Linear(self.D, self.H)    # path b
        self.w = nn.Linear(self.H, 1)         # attention scoring
        self.drop = nn.Dropout(0.25) if dropout else nn.Identity()
        self.cls = nn.Linear(self.D, pred_num)

        # Debug information
        self.last_train_idx = None  # Last training sampled channel indices (Tensor[256])
        self.eval_chunks = 0        # Number of chunks used in last validation
        
    @staticmethod
    def _first_linear_with_idx(lin: nn.Linear, x: torch.Tensor, idx: torch.Tensor):
        x_sub = x.index_select(-1, idx)                 # [B,N,keep]
        w_sub = lin.weight.index_select(1, idx)         # [H,keep]
        return F.linear(x_sub, w_sub, lin.bias)         # [B,N,H]

    def _activate(self, A_raw):
        if self.act == 'softmax':   return F.softmax(A_raw, dim=1)
        if self.act == 'sigmoid':   return torch.sigmoid(A_raw)
        if self.act == 'relu':      return F.relu(A_raw)
        if self.act == 'leaky_relu':return F.leaky_relu(A_raw)
        raise NotImplementedError
    
    @staticmethod
    def _cover_indices(D, keep, device, shuffle=True, seed=42, stride_divisor=4):
        """
        Generate chunk indices for feature selection during evaluation.
        
        Args:
            stride_divisor: Controls stride size (1, 2, 4, 8, 16)
                           stride = keep // stride_divisor
        """
        if keep >= D:  # One chunk covers all channels
            return [torch.arange(D, device=device)]
        if shuffle:
            g = torch.Generator(device=device); g.manual_seed(seed)
            perm = torch.randperm(D, generator=g, device=device)
        else:
            perm = torch.arange(D, device=device)
            
        stride = max(1, keep // stride_divisor)  # Adjustable via stride_divisor parameter
        starts = list(range(0, max(1, D - keep + 1), stride))
        if starts[-1] != (D - keep):
            starts.append(D - keep)  # Pad the last chunk to cover the end

        return [perm[i:i+keep] for i in starts]
    
    def _attend(self, x, idx=None):
        # Attention first layer uses sub-channels (if idx is not None)
        if idx is None:
            a = torch.tanh(self.V(x))
            b = torch.sigmoid(self.U(x))
        else:
            a = torch.tanh(self._first_linear_with_idx(self.V, x, idx))
            b = torch.sigmoid(self._first_linear_with_idx(self.U, x, idx))
        a = self.drop(a); b = self.drop(b)
        A_raw = self.w(a * b)  # [B,N,1]
        A = self._activate(A_raw)   # [B,N,1] (softmax normalizes along instance dimension)
        return A

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False, is_cox=False, no_feature_select=False, stride_divisor=4):
        """
        Forward pass of SimpleMIL model.
        
        Args:
            x: Input features [B, N, D]
            return_WSI_attn: Whether to return attention weights
            return_WSI_feature: Whether to return aggregated features
            is_cox: Whether using Cox proportional hazards (for survival analysis)
            no_feature_select: If True, disable subsampling and chunking, use ABMIL-style attention
            stride_divisor: Controls stride during eval (stride = keep // stride_divisor), options: 1, 2, 4, 8, 16
            
        Returns:
            dict with 'logits' key (consistent with other MIL models)
        """
        forward_return = {}
        B, N, D = x.shape
        assert D == self.D, f"expect D={self.D}, got {D}"

        # =============== Training ===============
        if self.training:
            if no_feature_select:
                # No subsampling, use ABMIL-style attention: A_raw = w(tanh(V(x)))
                idx = None
                self.last_train_idx = None
                z = torch.tanh(self.V(x))              # [B,N,H]
                z = self.drop(z)
                A_raw = self.w(z)                      # [B,N,1]
                A = self._activate(A_raw)              # [B,N,1]
            else:
                # Original SimpleMIL: subsampling + gated attention
                keep = min(self.H, D)
                idx = torch.randperm(D, device=x.device)[:keep]
                self.last_train_idx = idx
                A = self._attend(x, idx)               # [B,N,1]

            feat = torch.bmm(A.transpose(1, 2), x).squeeze(1)   # [B,D] pooling on original x
            logits = self.cls(feat)                             # [B,pred_num]

            forward_return['logits'] = logits
            if return_WSI_feature:
                forward_return['WSI_feature'] = feat
            if return_WSI_attn:
                forward_return['WSI_attn'] = A.squeeze(-1)
            return forward_return

        # =============== Eval/Test ===============
        keep = min(self.H, D)
        if no_feature_select:
            # No chunking, ABMIL attention, full-dim single pass
            self.eval_chunks = 1
            z = torch.tanh(self.V(x))                  # [B,N,H]
            z = self.drop(z)
            A_raw = self.w(z)                          # [B,N,1]
            A = self._activate(A_raw)                  # [B,N,1]

            feat = torch.bmm(A.transpose(1, 2), x).squeeze(1)  # [B,D]
            final_logits = self.cls(feat)                      # [B,pred_num] or [B] if pred_num==1
            if is_cox and final_logits.dim() == 2 and final_logits.size(-1) == 1:
                final_logits = final_logits.squeeze(-1)

            forward_return['logits'] = final_logits
            if return_WSI_feature:
                forward_return['WSI_feature'] = feat
            if return_WSI_attn:
                forward_return['WSI_attn'] = A.squeeze(-1)
            return forward_return

        # Original chunking logic: each chunk uses SimpleMIL gated attention
        idx_chunks = self._cover_indices(D, keep, x.device, shuffle=self.COVER_SHUFFLE, seed=self.COVER_SEED, stride_divisor=stride_divisor)
        self.eval_chunks = len(idx_chunks)

        logit_sum, feat_sum, attn_sum = [], [], []
        for idx in idx_chunks:
            A_blk = self._attend(x, None if keep >= D else idx)         # [B,N,1]
            feat_blk = torch.bmm(A_blk.transpose(1, 2), x).squeeze(1)   # [B,D]
            logits_blk = self.cls(feat_blk)                              # [B,pred_num]
            logit_sum.append(logits_blk)
            if return_WSI_feature:
                feat_sum.append(feat_blk)
            if return_WSI_attn:
                attn_sum.append(A_blk.squeeze(-1))

        if is_cox:
            eta_stack = torch.stack(logit_sum, dim=0)        # [K,B] (pred_num=1) or [K,B,pred]
            final_logits = torch.logsumexp(eta_stack, dim=0) - torch.log(
                torch.tensor(len(eta_stack), dtype=torch.float32, device=eta_stack.device)
            )
            var_logits = torch.var(eta_stack, dim=0)

            MI = None
            H_mean = None
            H_each = eta_stack
        else:
            # Classification task: average logits as final prediction (preserve original logic)
            logits_stack = torch.stack(logit_sum, dim=0)  # [K, B, pred_num]
            final_logits = logits_stack.mean(dim=0)  # [B, pred_num]
            
            # Uncertainty computation: convert to probability distribution first
            probs_stack = F.softmax(logits_stack, dim=-1)  # [K, B, pred_num]
            
            # Compute average probability distribution
            mean_p = probs_stack.mean(dim=0)  # [B, pred_num]
            
            # Compute entropy: H(mean_p) - entropy of average prediction (total uncertainty)
            H_mean = -torch.sum(mean_p * torch.log(mean_p + 1e-8), dim=-1)  # [B]
            
            # Compute entropy of each chunk prediction (aleatoric uncertainty)
            H_each = -torch.sum(probs_stack * torch.log(probs_stack + 1e-8), dim=-1)  # [K, B]
            
            # Mutual information = H(mean_p) - E[H(p_k)] (epistemic uncertainty)
            MI = H_mean - H_each.mean(dim=0)  # [B]
            
            # Variance of probabilities (in probability space)
            var_logits = torch.var(probs_stack, dim=0)  # [B, pred_num]

        forward_return['logits'] = final_logits  # Return logits (without softmax), external code will apply softmax
        forward_return['var_logits'] = var_logits
        if MI is not None:
            forward_return['MI'] = MI
        if H_mean is not None:
            forward_return['H_mean'] = H_mean
        if H_each is not None:
            forward_return['H_each'] = H_each
        if return_WSI_feature:
            forward_return['WSI_feature'] = torch.stack(feat_sum).mean(dim=0)
        if return_WSI_attn:
            forward_return['WSI_attn'] = torch.stack(attn_sum).mean(dim=0)
            forward_return['WSI_attn_std'] = torch.stack(attn_sum).std(dim=0)
            
        return forward_return
