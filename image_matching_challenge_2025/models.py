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
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.feature_channels, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        if self.backbone.startswith('vit'):
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

        # Similarity prediction head
        self.similarity = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        """Extract features from input images"""
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        """Compute similarity between two images"""
        feat1 = self.extract_features(x1)
        feat2 = self.extract_features(x2)

        # Concatenate features
        combined = torch.cat([feat1, feat2], dim=1)

        # Predict similarity
        similarity = self.similarity(combined)

        return similarity, feat1, feat2


class PoseEstimator(nn.Module):
    """Network to estimate camera pose"""

    def __init__(self, feature_dim=512):
        super().__init__()

        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)

        # Pose estimation head
        self.pose_regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Separate heads for rotation and translation
        self.rotation_head = nn.Linear(256, 9)  # 3x3 rotation matrix
        self.translation_head = nn.Linear(256, 3)  # 3D translation vector

    def forward(self, x1, x2):
        """Estimate relative pose between two images"""
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)

        # Concatenate features
        combined = torch.cat([feat1, feat2], dim=1)

        # Estimate pose
        pose_feat = self.pose_regressor(combined)

        # Predict rotation and translation
        rotation = self.rotation_head(pose_feat)
        rotation = rotation.view(-1, 3, 3)

        # Ensure proper rotation matrix
        U, _, V = torch.svd(rotation)
        rotation = torch.bmm(U, V.transpose(1, 2))

        translation = self.translation_head(pose_feat)

        return rotation, translation


class ImageMatchingModel(nn.Module):
    """Combined model for clustering and pose estimation"""

    def __init__(self, feature_dim=512):
        super().__init__()

        self.similarity_net = SimilarityNetwork(feature_dim=feature_dim)
        self.pose_estimator = PoseEstimator(feature_dim=feature_dim)

    def forward(self, x1, x2, mode='similarity'):
        """Forward pass based on mode"""
        if mode == 'similarity':
            return self.similarity_net(x1, x2)
        elif mode == 'pose':
            return self.pose_estimator(x1, x2)
        else:
            # Both
            similarity, _, _ = self.similarity_net(x1, x2)
            rotation, translation = self.pose_estimator(x1, x2)
            return similarity, rotation, translation


class SuperPointDetector(nn.Module):
    """SuperPoint-style keypoint detector"""

    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Detector head
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor head
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Shared encoder
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))

        # Detector
        cPa = F.relu(self.convPa(x))
        score = self.convPb(cPa)
        score = torch.sigmoid(score)

        # Descriptor
        cDa = F.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2, dim=1)

        return score, desc


class AdvancedMatchingModel(nn.Module):
    """Advanced model with keypoint detection and matching"""

    def __init__(self, feature_dim=512):
        super().__init__()

        self.keypoint_detector = SuperPointDetector()
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)

        # Graph neural network for matching
        self.gnn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=256, nhead=8)
            for _ in range(4)
        ])

        # Final prediction heads
        self.similarity_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.pose_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.rotation_head = nn.Linear(256, 9)
        self.translation_head = nn.Linear(256, 3)

    def forward(self, x1, x2):
        # Extract features
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)

        # Detect keypoints
        score1, desc1 = self.keypoint_detector(x1)
        score2, desc2 = self.keypoint_detector(x2)

        # Simple feature matching (can be improved)
        b, c, h, w = desc1.shape
        desc1_flat = desc1.view(b, c, -1).transpose(1, 2)
        desc2_flat = desc2.view(b, c, -1).transpose(1, 2)

        # Compute similarity matrix
        sim_matrix = torch.bmm(desc1_flat, desc2_flat.transpose(1, 2))

        # Aggregate matching features
        matching_feat = torch.cat([
            sim_matrix.max(dim=2)[0].mean(dim=1, keepdim=True),
            sim_matrix.max(dim=1)[0].mean(dim=1, keepdim=True)
        ], dim=1)

        # Combine with global features
        combined_feat = torch.cat([feat1, feat2, matching_feat], dim=1)

        # Predict similarity
        similarity = self.similarity_head(combined_feat)

        # Predict pose
        pose_feat = self.pose_head(combined_feat)
        rotation = self.rotation_head(pose_feat).view(-1, 3, 3)
        translation = self.translation_head(pose_feat)

        # Ensure proper rotation matrix
        U, _, V = torch.svd(rotation)
        rotation = torch.bmm(U, V.transpose(1, 2))

        return similarity, rotation, translation