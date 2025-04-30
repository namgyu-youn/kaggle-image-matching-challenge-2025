import torch
import torch.nn as nn
from einops import rearrange
from .utils import PositionEncodingSine


class LoFTRFeatureMatcher(nn.Module):
    """LoFTR-style coarse-to-fine feature matching"""

    def __init__(self, feature_dim=256, n_heads=8, n_layers=6):
        super().__init__()

        # Local feature CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1)
        )

        # Positional encoding
        self.pos_encoding = PositionEncodingSine(feature_dim, normalize=True)

        # Self-attention and cross-attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])

        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])

        # Coarse matching head
        self.coarse_matching = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

        # Fine matching head
        self.fine_matching = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 1, kernel_size=1)
        )

    def forward(self, img1, img2):
        # Extract features
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # Add positional encoding
        pos1 = self.pos_encoding(feat1)
        pos2 = self.pos_encoding(feat2)

        # Flatten spatial dimensions
        b, c, h, w = feat1.shape
        feat1_flat = rearrange(feat1, 'b c h w -> b (h w) c')
        feat2_flat = rearrange(feat2, 'b c h w -> b (h w) c')
        pos1_flat = rearrange(pos1, 'b c h w -> b (h w) c')
        pos2_flat = rearrange(pos2, 'b c h w -> b (h w) c')

        # Self-attention
        for layer in self.self_attn_layers:
            feat1_flat = layer(feat1_flat + pos1_flat)
            feat2_flat = layer(feat2_flat + pos2_flat)

        # Cross-attention
        for layer in self.cross_attn_layers:
            feat1_flat = layer(feat1_flat, feat2_flat + pos2_flat)
            feat2_flat = layer(feat2_flat, feat1_flat + pos1_flat)

        # Coarse matching
        feat1_expanded = feat1_flat.unsqueeze(2).expand(-1, -1, feat2_flat.size(1), -1)
        feat2_expanded = feat2_flat.unsqueeze(1).expand(-1, feat1_flat.size(1), -1, -1)
        combined = torch.cat([feat1_expanded, feat2_expanded], dim=-1)

        coarse_scores = self.coarse_matching(combined).squeeze(-1)

        return coarse_scores, feat1_flat, feat2_flat