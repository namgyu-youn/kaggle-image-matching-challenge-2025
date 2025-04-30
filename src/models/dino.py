import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2FeatureExtractor(nn.Module):
    """Feature extractor using DINOv2 for better performance"""

    def __init__(self, model_name='dinov2_vitb14', feature_dim=512):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.feature_channels = self.backbone.embed_dim

        # Get patch size
        self.patch_size = self.backbone.patch_embed.patch_size[0]  # Normally 14 or 16

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.feature_channels, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def _resize_if_needed(self, x):
        """Resize image if dimensions are not multiples of patch size"""
        B, C, H, W = x.shape

        # Check if resize is needed
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            new_H = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
            new_W = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size

            # Resize the image
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)

        return x

    def forward(self, x):
        # Handle dictionary input
        if isinstance(x, dict):
            if 'image1' in x and 'image2' in x:
                img1, img2 = x['image1'], x['image2']
            elif 'img1' in x and 'img2' in x:
                img1, img2 = x['img1'], x['img2']
            else:
                raise KeyError(f"Required image keys not found in input dictionary. Available keys: {list(x.keys())}")

            # Resize images if needed
            img1 = self._resize_if_needed(img1)
            img2 = self._resize_if_needed(img2)

            # Process first image
            features1 = self.backbone.forward_features(img1)
            class_token1 = features1[:, 0]
            feat1 = self.projection(class_token1)
            feat1 = F.normalize(feat1, p=2, dim=1)

            # Process second image
            features2 = self.backbone.forward_features(img2)
            class_token2 = features2[:, 0]
            feat2 = self.projection(class_token2)
            feat2 = F.normalize(feat2, p=2, dim=1)

            # Compute similarity
            similarity = torch.sum(feat1 * feat2, dim=1, keepdim=True)

            return {
                'feat1': feat1,
                'feat2': feat2,
                'similarity': similarity
            }

        # Single image processing
        x = self._resize_if_needed(x)
        features = self.backbone.forward_features(x)
        class_token = features[:, 0]

        # Project to feature space
        projected = self.projection(class_token)
        projected = F.normalize(projected, p=2, dim=1)

        return projected