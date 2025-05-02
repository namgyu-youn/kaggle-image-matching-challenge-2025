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

            # DINOv2 returns a dict - extracting the correct feature
            if isinstance(features1, dict):
                if 'x_norm_clstoken' in features1:
                    # Use CLS token if available
                    feat1_token = features1['x_norm_clstoken']
                elif 'x_norm_patchtokens' in features1:
                    # Use average of patch tokens if CLS token not available
                    feat1_token = features1['x_norm_patchtokens'].mean(dim=1)
                else:
                    # Fallback to first available feature
                    feat1_token = list(features1.values())[0]
                    if len(feat1_token.shape) > 2:
                        feat1_token = feat1_token.mean(dim=1)
            else:
                # If not a dict, assume it's the traditional transformer output with CLS token
                feat1_token = features1[:, 0] if features1.ndim > 2 else features1

            # Project features
            feat1 = self.projection(feat1_token)
            feat1 = F.normalize(feat1, p=2, dim=1)

            # Process second image
            features2 = self.backbone.forward_features(img2)

            # Extract features from second image using same approach
            if isinstance(features2, dict):
                if 'x_norm_clstoken' in features2:
                    feat2_token = features2['x_norm_clstoken']
                elif 'x_norm_patchtokens' in features2:
                    feat2_token = features2['x_norm_patchtokens'].mean(dim=1)
                else:
                    feat2_token = list(features2.values())[0]
                    if len(feat2_token.shape) > 2:
                        feat2_token = feat2_token.mean(dim=1)
            else:
                feat2_token = features2[:, 0] if features2.ndim > 2 else features2

            # Project features
            feat2 = self.projection(feat2_token)
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

        # Extract features using the same approach as above
        if isinstance(features, dict):
            if 'x_norm_clstoken' in features:
                token = features['x_norm_clstoken']
            elif 'x_norm_patchtokens' in features:
                token = features['x_norm_patchtokens'].mean(dim=1)
            else:
                token = list(features.values())[0]
                if len(token.shape) > 2:
                    token = token.mean(dim=1)
        else:
            token = features[:, 0] if features.ndim > 2 else features

        # Project to feature space
        projected = self.projection(token)
        projected = F.normalize(projected, p=2, dim=1)

        return projected