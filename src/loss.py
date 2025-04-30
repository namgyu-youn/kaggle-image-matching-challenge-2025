import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic Losses
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
        if 'similarity' in predictions and 'label' in targets:
            sim_loss = self.similarity_criterion(
                predictions['similarity'].squeeze(),
                targets['label'].float()
            )
            total_loss += self.similarity_weight * sim_loss
            loss_dict['similarity_loss'] = sim_loss.item()

        # Contrastive loss on features
        if 'feat1' in predictions and 'feat2' in predictions and 'label' in targets:
            contrastive_loss = self.contrastive_loss(
                predictions['feat1'],
                predictions['feat2'],
                targets['label']
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
            loss_dict['pose_loss'] = float(pose_loss)

        loss_dict['total_loss'] = total_loss.item()

        return total_loss


# Advanced Metric Learning Losses
class ArcFaceLoss(nn.Module):
    """ArcFace loss for better angular discrimination"""

    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Feature embeddings (batch_size, embedding_size)
            labels: Class labels (batch_size)
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarities
        cos_theta = F.linear(embeddings, weights)

        # Apply margin
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)

        # Create one-hot targets
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Add margin to only target logits
        output = torch.where(one_hot > 0, target_logits, cos_theta)

        # Scale logits
        output = output * self.scale

        # Compute cross entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


class CircleLoss(nn.Module):
    """Circle loss for better similarity learning"""

    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, pos_sims, neg_sims):
        """
        Args:
            pos_sims: Positive pair similarities (batch_size, n_pos)
            neg_sims: Negative pair similarities (batch_size, n_neg)
        """
        # For positive pairs, we want to maximize similarity (minimize -sim)
        ap = torch.clamp_min(- pos_sims + 1 + self.m, min=0)

        # For negative pairs, we want to minimize similarity (maximize sim)
        an = torch.clamp_min(neg_sims + self.m, min=0)

        # Apply weights based on similarity
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (pos_sims - delta_p) * self.gamma
        logit_n = an * (neg_sims - delta_n) * self.gamma

        # Compute Circle Loss
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))

        return loss.mean()


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity loss for better image matching"""

    def __init__(self, alpha=2.0, beta=50.0, lambda_val=0.5, epsilon=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_val = lambda_val
        self.epsilon = epsilon

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Feature embeddings (batch_size, embedding_dim)
            labels: Class labels (batch_size)
        """
        batch_size = embeddings.size(0)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = embeddings.mm(embeddings.t())

        # Create mask for positive and negative pairs
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_mask = label_matrix.float()
        neg_mask = 1.0 - pos_mask

        # Remove diagonal elements (self-similarity)
        eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        pos_mask[eye] = 0
        neg_mask[eye] = 0

        # Get positive and negative similarities
        pos_sims = sim_matrix * pos_mask
        neg_sims = sim_matrix * neg_mask

        # For each anchor, find hardest positive and negative
        hard_pos = torch.min(pos_sims + self.epsilon, dim=1)[0]
        hard_neg = torch.max(neg_sims - self.epsilon, dim=1)[0]

        # For each anchor, compute loss for all positives and negatives
        loss_pos = 0
        loss_neg = 0

        for i in range(batch_size):
            # Positive pairs
            positives = pos_sims[i][pos_mask[i].bool()]
            if len(positives) > 0:
                weights = torch.exp(-self.alpha * (positives - hard_neg[i]))
                loss_pos += torch.sum(weights * torch.log(1 + torch.exp(-self.beta * positives)))

            # Negative pairs
            negatives = neg_sims[i][neg_mask[i].bool()]
            if len(negatives) > 0:
                weights = torch.exp(self.alpha * (negatives - hard_pos[i]))
                loss_neg += torch.sum(weights * torch.log(1 + torch.exp(self.beta * negatives)))

        loss = (loss_pos + loss_neg) / batch_size
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: Feature embeddings (batch_size, embedding_dim)
            labels: Class labels (batch_size)
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Create similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask.fill_diagonal_(0)

        # Compute log probabilities
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()  # For numerical stability

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss


class TripletBatchHardLoss(nn.Module):
    """Triplet loss with batch hard mining"""

    def __init__(self, margin=0.3, soft=False):
        super().__init__()
        self.margin = margin
        self.soft = soft

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Feature embeddings (batch_size, embedding_dim)
            labels: Class labels (batch_size)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Create mask for positive and negative pairs
        same_label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        # For each anchor, find hardest positive
        pos_mask = same_label_mask.float() - torch.eye(embeddings.size(0), device=embeddings.device)
        hardest_positive_dist = torch.max(dist_matrix * pos_mask.clamp(min=0), dim=1)[0]

        # For each anchor, find hardest negative
        neg_mask = (1.0 - same_label_mask).float()
        hardest_negative_dist = torch.min(dist_matrix * neg_mask + (1.0 - neg_mask) * 1e5, dim=1)[0]

        # Compute triplet loss
        if self.soft:
            # Soft triplet loss (smooth approximation)
            loss = torch.log(1 + torch.exp(hardest_positive_dist - hardest_negative_dist))
        else:
            # Hard triplet loss with margin
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        # Return mean loss over valid triplets
        return loss.mean()


class ProxyNCALoss(nn.Module):
    """Proxy-NCA loss for metric learning"""

    def __init__(self, embedding_size, num_classes, scale=10.0):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.scale = scale

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Feature embeddings (batch_size, embedding_size)
            labels: Class labels (batch_size)
        """
        # Normalize embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)

        # Compute distances to proxies
        dist_matrix = torch.cdist(embeddings, proxies, p=2)

        # Convert distances to similarities
        sim_matrix = -dist_matrix * self.scale

        # Compute NCA loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


def get_metric_learning_loss(loss_type, **kwargs):
    """Factory function to get different metric learning losses"""
    if loss_type == 'arcface':
        return ArcFaceLoss(**kwargs)
    elif loss_type == 'circle':
        return CircleLoss(**kwargs)
    elif loss_type == 'multi_similarity':
        return MultiSimilarityLoss(**kwargs)
    elif loss_type == 'supcon':
        return SupConLoss(**kwargs)
    elif loss_type == 'triplet_hard':
        return TripletBatchHardLoss(**kwargs)
    elif loss_type == 'proxy_nca':
        return ProxyNCALoss(**kwargs)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(**kwargs)
    elif loss_type == 'triplet':
        return TripletLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")