import torch
import torch.nn as nn
import torch.nn.functional as F


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
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")