import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torch.nn.functional as F


class MixedPrecisionTrainer:
    """Trainer with mixed precision training for faster computation"""

    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()

    def train_step(self, batch, criterion):
        self.model.train()

        # Mixed precision training
        with autocast():
            outputs = self.model(batch)
            loss = criterion(outputs, batch['targets'])

        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()


class HardNegativeMining:
    """Hard negative mining for contrastive learning"""

    def __init__(self, num_negative=3, temperature=0.07):
        self.num_negative = num_negative
        self.temperature = temperature

    def mine_hard_negatives(self, anchor_features, positive_features, negative_features):
        """Find hard negatives for each anchor"""
        # Compute similarities
        anchor_norm = F.normalize(anchor_features, p=2, dim=1)
        negative_norm = F.normalize(negative_features, p=2, dim=1)

        similarities = torch.matmul(anchor_norm, negative_norm.t())

        # Find hardest negatives (highest similarity)
        hard_negative_indices = torch.topk(similarities, self.num_negative, dim=1).indices

        return hard_negative_indices

    def compute_loss(self, anchor, positive, negatives):
        """Compute contrastive loss with hard negatives"""
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negatives_norm = F.normalize(negatives, p=2, dim=1)

        # Positive similarity
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature

        # Negative similarities
        neg_sim = torch.matmul(anchor_norm.unsqueeze(1), negatives_norm.transpose(1, 2))
        neg_sim = neg_sim.squeeze(1) / self.temperature

        # Combine and compute loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss

class CurriculumLearning:
    """Curriculum learning strategy"""

    def __init__(self, initial_difficulty=0.0, max_difficulty=1.0, warmup_epochs=10):
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.warmup_epochs = warmup_epochs
        self.current_difficulty = initial_difficulty

    def update_difficulty(self, epoch):
        """Update difficulty level based on epoch"""
        if epoch < self.warmup_epochs:
            self.current_difficulty = self.initial_difficulty + \
                (self.max_difficulty - self.initial_difficulty) * (epoch / self.warmup_epochs)
        else:
            self.current_difficulty = self.max_difficulty

    def should_use_sample(self, sample_difficulty):
        """Determine if a sample should be used based on its difficulty"""
        return sample_difficulty <= self.current_difficulty


class EMAModel:
    """Exponential Moving Average of model parameters"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                   (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class AdversarialTraining:
    """Adversarial training for robustness"""

    def __init__(self, epsilon=0.01, alpha=0.005, num_steps=3):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def generate_adversarial_example(self, model, x, y, criterion):
        """Generate adversarial example using PGD"""
        x_adv = x.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad = True
            outputs = model(x_adv)
            loss = criterion(outputs, y)

            # Compute gradients
            grad = torch.autograd.grad(loss, x_adv)[0]

            # Update adversarial example
            x_adv = x_adv.detach() + self.alpha * grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
            x_adv = torch.clamp(x + delta, min=0, max=1).detach()

        return x_adv


class LabelSmoothing(nn.Module):
    """Label smoothing for better generalization"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (batch_size, n_classes)
            target: Targets (batch_size)
        """
        n_classes = pred.size(-1)

        # Convert to one-hot
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

        # Apply smoothing
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Compute loss
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -(smoothed * log_prob).sum(dim=-1).mean()

        return loss


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * alpha
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                param_group['lr'] = self.min_lr + (base_lr - self.min_lr) * cosine_decay


def get_advanced_optimizer(model, config):
    """Get advanced optimizer with different parameter groups"""
    # Separate backbone and head parameters
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name or 'feature_extractor' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Different learning rates for backbone and head
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['learning_rate'] * 0.1},
        {'params': head_params, 'lr': config['learning_rate']}
    ], weight_decay=config['weight_decay'])

    return optimizer