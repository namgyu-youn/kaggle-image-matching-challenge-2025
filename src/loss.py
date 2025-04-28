import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        """
        Args:
            feat1: Features of first image
            feat2: Features of second image
            label: 0 for negative pair, 1 for positive pair
        """
        distance = F.pairwise_distance(feat1, feat2)

        loss = label * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss for similarity learning"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor features
            positive: Positive features (same scene)
            negative: Negative features (different scene)
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)

        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class PoseLoss(nn.Module):
    """Loss for pose estimation"""

    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def geodesic_loss(self, pred_rot, gt_rot):
        """Geodesic loss for rotation matrices"""
        # Compute the relative rotation matrix
        relative_rot = torch.bmm(pred_rot.transpose(1, 2), gt_rot)

        # Compute the trace
        trace = torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(-1)

        # Compute the geodesic distance
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)
        theta = torch.acos(cos_theta)

        return theta.mean()

    def forward(self, pred_rot, pred_trans, gt_rot=None, gt_trans=None):
        """
        Args:
            pred_rot: Predicted rotation matrix (B, 3, 3)
            pred_trans: Predicted translation vector (B, 3)
            gt_rot: Ground truth rotation matrix (B, 3, 3)
            gt_trans: Ground truth translation vector (B, 3)
        """
        loss = 0.0

        if gt_rot is not None:
            # Rotation loss (geodesic distance)
            rot_loss = self.geodesic_loss(pred_rot, gt_rot)
            loss += self.rotation_weight * rot_loss

        if gt_trans is not None:
            # Translation loss (L2 distance)
            trans_loss = F.mse_loss(pred_trans, gt_trans)
            loss += self.translation_weight * trans_loss

        return loss


class CombinedLoss(nn.Module):
    """Combined loss for the full model"""

    def __init__(self, similarity_weight=1.0, pose_weight=1.0,
                 contrastive_margin=1.0, rotation_weight=1.0,
                 translation_weight=1.0):
        super().__init__()

        self.similarity_weight = similarity_weight
        self.pose_weight = pose_weight

        self.similarity_criterion = nn.BCELoss()
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        self.pose_loss = PoseLoss(rotation_weight=rotation_weight,
                                 translation_weight=translation_weight)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict containing model outputs
            targets: Dict containing ground truth values
        """
        total_loss = 0.0
        loss_dict = {}

        # Similarity loss
        if 'similarity' in predictions and 'similarity' in targets:
            sim_loss = self.similarity_criterion(
                predictions['similarity'].squeeze(),
                targets['similarity'].float()
            )
            total_loss += self.similarity_weight * sim_loss
            loss_dict['similarity_loss'] = sim_loss.item()

        # Contrastive loss on features
        if 'feat1' in predictions and 'feat2' in predictions and 'similarity' in targets:
            contrastive_loss = self.contrastive_loss(
                predictions['feat1'],
                predictions['feat2'],
                targets['similarity']
            )
            total_loss += self.similarity_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.item()

        # Pose loss
        if 'rotation' in predictions and 'translation' in predictions:
            pose_loss = self.pose_loss(
                predictions['rotation'],
                predictions['translation'],
                targets.get('rotation', None),
                targets.get('translation', None)
            )
            total_loss += self.pose_weight * pose_loss
            loss_dict['pose_loss'] = pose_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class MetricLearningLoss(nn.Module):
    """Advanced metric learning loss"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Feature embeddings (B, D)
            labels: Scene labels (B,)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature

        # Create mask for positive pairs
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float()

        # Remove diagonal elements
        mask.fill_diagonal_(0)

        # Compute log probabilities
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # Loss
        loss = -mean_log_prob_pos.mean()

        return loss


class GeometricConsistencyLoss(nn.Module):
    """Geometric consistency loss for pose estimation"""

    def __init__(self):
        super().__init__()

    def forward(self, R1, t1, R2, t2):
        """
        Check if R1, t1 and R2, t2 form a consistent transformation
        Args:
            R1, t1: First transformation
            R2, t2: Second transformation
        """
        # Compose transformations
        R_composed = torch.bmm(R2, R1)
        t_composed = torch.bmm(R2, t1.unsqueeze(-1)).squeeze(-1) + t2

        # Check if composed transformation is close to identity
        identity = torch.eye(3, device=R1.device).unsqueeze(0).repeat(R1.size(0), 1, 1)

        rotation_consistency = F.mse_loss(R_composed, identity)
        translation_consistency = F.mse_loss(t_composed, torch.zeros_like(t_composed))

        return rotation_consistency + translation_consistency