import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DINOv2FeatureExtractor(nn.Module):
    """Feature extractor using DINOv2 for better performance"""

    def __init__(self, model_name='dinov2_vitb14', feature_dim=512):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.feature_channels = self.backbone.embed_dim

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.feature_channels, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        # Extract features from DINOv2
        features = self.backbone.forward_features(x)

        # Global pooling for class token
        class_token = features[:, 0]

        # Project to feature space
        projected = self.projection(class_token)
        projected = F.normalize(projected, p=2, dim=1)

        return projected


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


class PositionEncodingSine(nn.Module):
    """Sinusoidal position encoding for transformer"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones_like(x[0, 0, :, :])
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        pos = pos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos