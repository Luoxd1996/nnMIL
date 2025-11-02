import torch
import torch.nn as nn
import torch.nn.functional as F
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class AB_MIL(nn.Module):
    def __init__(self,L = 512,D = 128,num_classes = 2,dropout=0,act= nn.ReLU() ,in_dim = 512, rrt = None):
        super(AB_MIL, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        self.feature = [nn.Linear(in_dim, self.L)]
        
        self.feature += [act]

        if dropout:
            self.feature += [nn.Dropout(dropout)]
            
        if self.rrt != None:
            self.feature += [self.rrt]

        self.feature = nn.Sequential(*self.feature)
        self.drop = nn.Dropout(dropout)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.in_dim, self.num_classes),
        )

        self.apply(initialize_weights)
    def forward(self, x, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        feature = self.feature(x)  # [B, N, L]
        B, N, L = feature.shape
        
        # Unified batch processing (works for any batch_size including 1)
        # Reshape for batch attention computation: [B, N, L] -> [B*N, L]
        feature_flat = feature.view(-1, L)  # [B*N, L]
        A_flat = self.attention(feature_flat)  # [B*N, K]
        A = A_flat.view(B, N, self.K)  # [B, N, K]
        
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # [B, K, N]
        A = F.softmax(A, dim=-1)  # softmax over N dimension
        
        # Attention on original features (keeping full pretrained representations)
        M = torch.bmm(A, x)  # [B, K, in_dim] - attention acts on original x
        M = M.view(B, self.in_dim)  # [B, in_dim] for classifier (K=1)
        
        M = self.drop(M)
        
        logits = self.classifier(M)  # [B, num_classes]
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori.transpose(-1, -2)  # [B, N, K]
            
        return forward_return



