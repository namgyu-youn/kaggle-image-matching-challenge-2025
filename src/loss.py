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


class CircleLoss(nn.Module):
    """Circle loss for better similarity learning"""

    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, features, labels):
        """
        Args:
            features: Feature embeddings (batch_size, embedding_dim)
            labels: Class labels (batch_size)
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T)

        # Create masks for positive and negative pairs
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()
        pos_mask.fill_diagonal_(0)  # Remove self-similarity
        neg_mask = 1.0 - pos_mask

        # Handle case with no positive pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # Extract positive and negative similarities
        pos_sims = sim_matrix * pos_mask
        neg_sims = sim_matrix * neg_mask

        # For positive pairs (minimize -sim)
        ap = torch.clamp_min(- pos_sims.masked_fill(pos_mask == 0, -1e12).max(dim=1)[0] + 1 + self.m, min=0)

        # For negative pairs (maximize sim)
        an = torch.clamp_min(neg_sims.masked_fill(neg_mask == 0, -1e12).max(dim=1)[0] + self.m, min=0)

        # Apply weights based on similarity
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (pos_sims.masked_fill(pos_mask == 0, -1e12).max(dim=1)[0] - delta_p) * self.gamma
        logit_n = an * (neg_sims.masked_fill(neg_mask == 0, -1e12).max(dim=1)[0] - delta_n) * self.gamma

        # Compute Circle Loss
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class EnhancedCombinedLoss(nn.Module):
    """Combined loss fuction (SupCon & CircleLoss)"""

    def __init__(self, similarity_weight=1.0, pose_weight=1.0,
                 contrastive_margin=1.0, rotation_weight=1.0,
                 translation_weight=1.0, temperature=0.07,
                 use_adaptive_weights=True, use_focal_loss=True):
        super().__init__()

        self.similarity_weight = similarity_weight
        self.pose_weight = pose_weight
        self.temperature = temperature
        self.use_adaptive_weights = use_adaptive_weights
        self.use_focal_loss = use_focal_loss

        # Base loss functions
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        self.pose_loss = PoseLoss(rotation_weight=rotation_weight,
                                 translation_weight=translation_weight)

        # Combine SupCon with CircleLoss
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.circle_loss = CircleLoss(m=0.25, gamma=256)

        # Focal Loss 파라미터
        self.focal_gamma = 2.0
        self.focal_alpha = 0.25

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict containing model outputs
            targets: Dict containing ground truth values
        """
        total_loss = 0.0
        loss_dict = {}

        # 1. Calculate simiarity loss (Base + Focal Loss)
        if 'similarity' in predictions and 'label' in targets:
            if self.use_focal_loss:
                sim_loss = self.focal_bce_loss(
                    predictions['similarity'].squeeze(),
                    targets['label'].float()
                )
            else:
                sim_loss = F.binary_cross_entropy_with_logits(
                    predictions['similarity'].squeeze(),
                    targets['label'].float()
                )

            # Adaptive loss : Higher weight for higher loss
            if self.use_adaptive_weights:
                sim_weight = self.similarity_weight * (1.0 + sim_loss.detach())
            else:
                sim_weight = self.similarity_weight

            total_loss += sim_weight * sim_loss
            loss_dict['similarity_loss'] = sim_loss.item()

        # 2. contrastive_loss for feature vectors
        if 'feat1' in predictions and 'feat2' in predictions and 'label' in targets:
            contrastive_loss = self.contrastive_loss(
                predictions['feat1'],
                predictions['feat2'],
                targets['label']
            )

            total_loss += self.similarity_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.item()

            # 3. SupCon Loss - Use both positive/negative samples in batch
            if torch.sum(targets['label']) > 0:  # Positive sample exist
                # Feature label connection
                features = torch.cat([predictions['feat1'], predictions['feat2']], dim=0)
                batch_size = predictions['feat1'].size(0)
                labels = torch.cat([targets['label'], targets['label']], dim=0)

                supcon_loss = self.supcon_loss(features, labels)
                total_loss += 0.5 * self.similarity_weight * supcon_loss
                loss_dict['supcon_loss'] = supcon_loss.item()

        # 4. Pose loss - Apply uncertaint weight
        if 'rotation' in predictions and 'translation' in predictions:
            pose_loss = self.pose_loss(
                predictions['rotation'],
                predictions['translation'],
                targets.get('rotation', None),
                targets.get('translation', None)
            )

            # Pose estimation based on confidency
            if 'confidence' in predictions and self.use_adaptive_weights:
                confidence = predictions['confidence'].detach()
                pose_weight = self.pose_weight * (2.0 - confidence.mean())
            else:
                pose_weight = self.pose_weight

            total_loss += pose_weight * pose_loss
            loss_dict['pose_loss'] = float(pose_loss)

        # 5. Final loss
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def focal_bce_loss(self, pred, target):
        """Focal Loss for binary classification - Higher weight for harder samples"""
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)

        # Base BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Apply focal loss
        focal_weight = (1 - p_t) ** self.focal_gamma
        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()


def get_enhanced_loss(loss_type='enhanced', **kwargs):
    """Factory function to get different metric learning losses"""

    if loss_type == 'enhanced':
        return EnhancedCombinedLoss(**kwargs)
    elif loss_type == 'supcon':
        return SupConLoss(**kwargs)
    elif loss_type == 'circle':
        return CircleLoss(**kwargs)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")