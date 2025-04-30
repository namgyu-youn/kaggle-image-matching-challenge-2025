import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import logging
from tqdm import tqdm


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
            loss = criterion(outputs, batch)

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

    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                # Forward pass
                with autocast():
                    outputs = self.model(batch)
                    loss = criterion(outputs, batch)

                val_loss += loss.item()

        return val_loss / len(val_loader)


class HardNegativeMining:
    """Hard negative mining for contrastive learning"""

    def __init__(self, num_negative=3, temperature=0.07):
        self.num_negative = num_negative
        self.temperature = temperature

    def mine_hard_negatives(self, anchor_features, positive_features, negative_features):
        """Find hard negatives for each anchor"""
        # Compute similarities
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        negative_features = F.normalize(negative_features, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(anchor_features, negative_features.t())

        # Get indices of hardest negatives
        _, hard_indices = torch.topk(sim_matrix, k=self.num_negative, dim=1)

        # Gather hard negatives
        batch_size = anchor_features.size(0)
        hard_negatives = torch.zeros(
            batch_size, self.num_negative, anchor_features.size(1),
            device=anchor_features.device
        )

        for i in range(batch_size):
            hard_negatives[i] = negative_features[hard_indices[i]]

        return hard_negatives

    def compute_loss(self, anchor, positive, negatives):
        """Compute contrastive loss with hard negatives"""
        # Normalize features
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)

        # Compute positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature

        # Compute negative similarities
        neg_sim = torch.bmm(
            anchor.unsqueeze(1),
            negatives.transpose(1, 2)
        ).squeeze(1) / self.temperature

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class CurriculumLearning:
    """Curriculum learning for gradually increasing task difficulty"""

    def __init__(self, initial_difficulty=0.0, max_difficulty=1.0, warmup_epochs=10):
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.warmup_epochs = warmup_epochs
        self.current_difficulty = initial_difficulty

    def update_difficulty(self, epoch):
        """Update difficulty based on current epoch"""
        if self.warmup_epochs > 0:
            progress = min(1.0, epoch / self.warmup_epochs)
            self.current_difficulty = self.initial_difficulty + \
                                     progress * (self.max_difficulty - self.initial_difficulty)
        else:
            self.current_difficulty = self.max_difficulty

        return self.current_difficulty

    def get_difficulty(self):
        """Get current difficulty level"""
        return self.current_difficulty


class EMAModel:
    """Exponential Moving Average of model parameters"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to the model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters to the model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""

    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, final_lr=0, warmup_start_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0

    def step(self, epoch=None):
        """Update learning rate based on epoch"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_epoch)

    def get_lr(self, epoch):
        """Calculate learning rate for current epoch"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            return self.warmup_start_lr + alpha * (self.base_lr - self.warmup_start_lr)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return self.final_lr + 0.5 * (self.base_lr - self.final_lr) * \
                   (1 + np.cos(np.pi * progress))


def get_advanced_optimizer(model, lr=1e-3, weight_decay=1e-4, optimizer_type='adam'):
    """Get optimizer with different parameter groups"""
    # Separate parameters for different learning rates
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name or 'feature_extractor' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': lr}             # Higher LR for heads
    ]

    # Create optimizer
    if optimizer_type == 'adam':
        return optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


class Trainer:
    """Main trainer class for image matching models"""

    def __init__(self, model, config):
        """
        Initialize trainer with model and configuration

        Args:
            model: The model to train
            config: Dictionary with training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = get_advanced_optimizer(
            model,
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4),
            optimizer_type=config.get('optimizer', 'adamw')
        )

        # Setup scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            max_epochs=config.get('epochs', 100),
            base_lr=config.get('learning_rate', 1e-4),
            final_lr=config.get('min_learning_rate', 1e-6),
            warmup_start_lr=config.get('warmup_start_lr', 1e-6)
        )

        # Setup mixed precision training
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        if self.use_mixed_precision:
            self.trainer = MixedPrecisionTrainer(self.model, self.optimizer, self.device)

        # Setup EMA
        self.use_ema = config.get('use_ema', False)
        if self.use_ema:
            self.ema = EMAModel(self.model, decay=config.get('ema_decay', 0.999))

        # Setup logging
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Setup checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Setup logger
        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger()

    def train(self, train_loader, val_loader, criterion):
        """
        Train the model

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
        """
        num_epochs = self.config.get('epochs', 100)

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                # Training step
                if self.use_mixed_precision:
                    loss = self.trainer.train_step(batch, criterion)
                else:
                    # Standard training
                    self.optimizer.zero_grad()
                    outputs = self.model(batch)
                    loss = criterion(outputs, batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    loss = loss.item()

                train_loss += loss
                progress_bar.set_postfix({'loss': loss})

                # Update EMA model
                if self.use_ema:
                    self.ema.update()

            # Calculate average training loss
            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            # Apply EMA for validation
            if self.use_ema:
                self.ema.apply_shadow()

            with torch.no_grad():
                for batch in val_loader:
                    # Move batch to device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)

                    # Forward pass
                    outputs = self.model(batch)
                    loss = criterion(outputs, batch)
                    val_loss += loss.item()

            # Restore original model
            if self.use_ema:
                self.ema.restore()

            # Calculate average validation loss
            val_loss /= len(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Log metrics
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save checkpoint
            self._save_checkpoint(epoch, val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_loss, is_best=True)

            # Check for early stopping
            patience = self.config.get('patience', 10)
            if epoch - self.best_epoch >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_latest.pth')

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_best.pth')
            self.logger.info(f"Saved best model checkpoint with val_loss: {val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.best_epoch = checkpoint.get('epoch', 0)

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0)