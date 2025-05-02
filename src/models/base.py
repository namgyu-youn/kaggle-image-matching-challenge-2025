import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeatureExtractor(nn.Module):
    """Feature extractor based on pre-trained CNN backbone"""

    def __init__(self, backbone='resnet50', pretrained=True, feature_dim=512):
        super().__init__()

        if backbone == 'resnet50':
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            else:
                weights = None
            base_model = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_channels = 2048
        elif backbone == 'efficientnet_b3':
            if pretrained:
                weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            else:
                weights = None
            base_model = models.efficientnet_b3(weights=weights)
            self.features = base_model.features
            self.feature_channels = 1536
        elif backbone == 'vit_b_16':
            if pretrained:
                weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            else:
                weights = None
            base_model = models.vit_b_16(weights=weights)
            self.features = base_model
            self.feature_channels = 768
        elif backbone == 'dinov2':
            # DINOv2 feature extractor (if torch.hub is available)
            try:
                self.features = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                self.feature_channels = 768
            except Exception as e:
                print(f"DINOv2 model not available, falling back to ResNet50: {e}")
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Enhanced projection layer with residual connection
        self.projection = nn.Sequential(
            nn.Linear(self.feature_channels, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, x):
        if self.backbone == 'dinov2':
            # DINOv2 specific forward pass
            feat = self.features(x)
            # Check if feat is a dictionary or tensor
            if isinstance(feat, dict):
                if 'x_norm_patchtokens' in feat:
                    x = feat['x_norm_patchtokens'].mean(dim=1)  # Use patch tokens, global pooling
                elif 'x_norm_clstoken' in feat:
                    x = feat['x_norm_clstoken']  # Use CLS token
                else:
                    # Fallback to first available feature
                    x = next(iter(feat.values()))
                    if len(x.shape) > 2:
                        x = x.mean(dim=1)  # Apply global pooling if needed
            else:
                # Handle case when feat is a tensor (not a dictionary)
                if len(feat.shape) == 4:  # [B, C, H, W]
                    x = feat.mean(dim=(2, 3))  # Global average pooling
                elif len(feat.shape) == 3:  # [B, L, C]
                    x = feat.mean(dim=1)  # Average across sequence length
                else:
                    x = feat  # Already in expected format
        elif self.backbone.startswith('vit'):
            # Vision Transformer
            x = self.features.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.features.encoder(x)
            x = x[:, 0]  # class token
        else:
            # CNN backbone
            x = self.features(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)

        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)

        return x


class SimilarityNetwork(nn.Module):
    """Network to compute similarity between image pairs"""

    def __init__(self, feature_dim=512):
        super().__init__()

        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)

        # Enhanced similarity prediction head with non-linearities
        self.similarity = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        """Extract features from input images"""
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        """Compute similarity between two images"""
        feat1 = self.extract_features(x1)
        feat2 = self.extract_features(x2)

        # Concatenate features and their element-wise product
        combined = torch.cat([
            feat1,
            feat2,
            feat1 * feat2,  # Element-wise product for better similarity
            torch.abs(feat1 - feat2)  # Absolute difference
        ], dim=1)

        # Predict similarity
        similarity = self.similarity(combined)

        return similarity, feat1, feat2


class PoseEstimator(nn.Module):
    """Network to estimate camera pose"""

    def __init__(self, feature_dim=512):
        super().__init__()

        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)

        # Enhanced pose estimation head
        self.pose_regressor = nn.Sequential(
            nn.Linear(feature_dim * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Separate heads for rotation and translation
        self.rotation_head = nn.Linear(256, 9)  # 3x3 rotation matrix
        self.translation_head = nn.Linear(256, 3)  # 3D translation vector

        # Confidence score for prediction quality
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        """Estimate relative pose between two images"""
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)

        # Enhanced feature combination
        combined = torch.cat([
            feat1,
            feat2,
            feat1 * feat2,  # Element-wise product
            torch.abs(feat1 - feat2)  # Absolute difference
        ], dim=1)

        # Estimate pose
        pose_feat = self.pose_regressor(combined)

        # Predict rotation and translation
        rotation = self.rotation_head(pose_feat)
        rotation = rotation.view(-1, 3, 3)

        # Ensure proper rotation matrix with orthogonalization
        U, _, V = torch.svd(rotation)
        rotation = torch.bmm(U, V.transpose(1, 2))

        # Ensure determinant is 1
        det = torch.det(rotation).view(-1, 1, 1)
        rotation = rotation / torch.pow(torch.abs(det) + 1e-6, 1/3)

        translation = self.translation_head(pose_feat)

        # Confidence score
        confidence = self.confidence_head(pose_feat)

        return rotation, translation, confidence


class ImageMatchingModel(nn.Module):
    """Combined model for clustering and pose estimation"""

    def __init__(self, feature_dim=512, backbone='resnet50'):
        super().__init__()

        self.similarity_net = SimilarityNetwork(feature_dim=feature_dim)

        # Use the same backbone type for both networks
        self.similarity_net.feature_extractor.backbone = backbone

        self.pose_estimator = PoseEstimator(feature_dim=feature_dim)
        self.pose_estimator.feature_extractor.backbone = backbone

        # Optional: Share weights between feature extractors
        # self.pose_estimator.feature_extractor = self.similarity_net.feature_extractor

    def forward(self, x):
        """Forward pass based on mode"""
        # Handle dictionary input
        if isinstance(x, dict):
            if 'image1' in x and 'image2' in x:
                x1, x2 = x['image1'], x['image2']
            elif 'img1' in x and 'img2' in x:
                x1, x2 = x['img1'], x['img2']
            else:
                raise KeyError(f"Required image keys not found in input dictionary. Available keys: {list(x.keys())}")

            # Get similarity
            similarity, feat1, feat2 = self.similarity_net(x1, x2)

            # Get pose if needed
            if x.get('mode') == 'pose' or x.get('mode') == 'full':
                rotation, translation, confidence = self.pose_estimator(x1, x2)
                return {
                    'similarity': similarity,
                    'feat1': feat1,
                    'feat2': feat2,
                    'rotation': rotation,
                    'translation': translation,
                    'confidence': confidence
                }
            else:
                return {
                    'similarity': similarity,
                    'feat1': feat1,
                    'feat2': feat2
                }
        else:
            # Legacy support for direct input
            x1, x2 = x
            similarity, feat1, feat2 = self.similarity_net(x1, x2)
            return similarity, feat1, feat2