import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperGlueMatchingModule(nn.Module):
    """SuperGlue-style graph neural network for feature matching"""

    def __init__(self, feature_dim=256, n_layers=9):
        super().__init__()

        self.layers = nn.ModuleList([
            AttentionalGNN(feature_dim) for _ in range(n_layers)
        ])

        self.final_proj = nn.Linear(feature_dim, feature_dim)
        self.bin_score = nn.Linear(feature_dim, 1)

    def forward(self, desc0, desc1, keypoints0=None, keypoints1=None):
        # Normalize descriptors
        desc0 = F.normalize(desc0, p=2, dim=-1)
        desc1 = F.normalize(desc1, p=2, dim=-1)

        # Message passing
        for layer in self.layers:
            desc0, desc1 = layer(desc0, desc1)

        # Project to final descriptor space
        mdesc0 = self.final_proj(desc0)
        mdesc1 = self.final_proj(desc1)

        # Compute matching scores
        scores = torch.einsum('bnd,bmd->bnm', mdesc0, mdesc1)

        # Dustbin scores (for unmatched features)
        bin_score0 = self.bin_score(desc0)
        bin_score1 = self.bin_score(desc1)

        # Add dustbin scores
        scores = torch.cat([scores, bin_score0.expand(-1, -1, 1)], dim=2)
        scores = torch.cat([scores, bin_score1.expand(-1, -1, 1).transpose(1, 2)], dim=1)

        # Apply log softmax
        scores = F.log_softmax(scores, dim=2)
        scores = F.log_softmax(scores, dim=1)

        return scores


class AttentionalGNN(nn.Module):
    """Attentional Graph Neural Network layer"""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        self.attn = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, desc0, desc1):
        # Self-attention
        desc0 = desc0 + self.attn(desc0, desc0, desc0, need_weights=False)[0]
        desc1 = desc1 + self.attn(desc1, desc1, desc1, need_weights=False)[0]

        # Cross-attention
        desc0 = desc0 + self.attn(desc0, desc1, desc1, need_weights=False)[0]
        desc1 = desc1 + self.attn(desc1, desc0, desc0, need_weights=False)[0]

        # MLP
        desc0 = desc0 + self.mlp(torch.cat([desc0, desc0], dim=-1))
        desc1 = desc1 + self.mlp(torch.cat([desc1, desc1], dim=-1))

        return desc0, desc1