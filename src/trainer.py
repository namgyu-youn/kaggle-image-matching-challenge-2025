import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from src.models import ImageMatchingModel, AdvancedMatchingModel
from src.loss import CombinedLoss
from src.dataset import get_dataloaders


class Trainer:
    """Trainer for Image Matching Challenge models"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Model
        if config.get('model_type', 'basic') == 'advanced':
            self.model = AdvancedMatchingModel(feature_dim=config.get('feature_dim', 512))
        else:
            self.model = ImageMatchingModel(feature_dim=config.get('feature_dim', 512))

        self.model = self.model.to(self.device)

        # Loss
        self.criterion = CombinedLoss(
            similarity_weight=config.get('similarity_weight', 1.0),
            pose_weight=config.get('pose_weight', 1.0),
            contrastive_margin=config.get('contrastive_margin', 1.0),
            rotation_weight=config.get('rotation_weight', 1.0),
            translation_weight=config.get('translation_weight', 1.0)
        )

        # Optimizer
        self.optimizer = self._get_optimizer()

        # Scheduler
        self.scheduler = self._get_scheduler()

        # Data loaders
        data_dir = Path(config['data_dir']).absolute()
        print(f"Loading data from: {data_dir}")
        self.train_loader, self.val_loader = get_dataloaders(
            str(data_dir),  # Convert to string for compatibility
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4)
        )

        # Logging
        self.writer = SummaryWriter(config.get('log_dir', 'logs'))
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        # Load checkpoint if specified
        if config.get('resume', None):
            self.load_checkpoint(config['resume'])

    def _get_optimizer(self):
        """Create optimizer"""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        if opt_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _get_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('factor', 0.1),
                patience=self.config.get('patience', 10),
                min_lr=self.config.get('min_lr', 1e-6)
            )
        else:
            return None

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_dict_sum = {}

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            label = batch['label'].to(self.device)

            # Forward pass
            if isinstance(self.model, AdvancedMatchingModel):
                similarity, rotation, translation = self.model(image1, image2)
                predictions = {
                    'similarity': similarity,
                    'rotation': rotation,
                    'translation': translation
                }
            else:
                similarity, feat1, feat2 = self.model(image1, image2, mode='similarity')
                rotation, translation = self.model(image1, image2, mode='pose')
                predictions = {
                    'similarity': similarity,
                    'feat1': feat1,
                    'feat2': feat2,
                    'rotation': rotation,
                    'translation': translation
                }

            targets = {
                'similarity': label
            }

            # Calculate loss
            loss, loss_dict = self.criterion(predictions, targets)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_dict_sum:
                    loss_dict_sum[k] = 0
                loss_dict_sum[k] += v

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to tensorboard
            if batch_idx % self.config.get('log_interval', 10) == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Train/{k}', v, step)

        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for k in loss_dict_sum:
            loss_dict_sum[k] /= len(self.train_loader)

        return avg_loss, loss_dict_sum

    def validate(self, epoch):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        loss_dict_sum = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                image1 = batch['image1'].to(self.device)
                image2 = batch['image2'].to(self.device)
                label = batch['label'].to(self.device)

                # Forward pass
                if isinstance(self.model, AdvancedMatchingModel):
                    similarity, rotation, translation = self.model(image1, image2)
                    predictions = {
                        'similarity': similarity,
                        'rotation': rotation,
                        'translation': translation
                    }
                else:
                    similarity, feat1, feat2 = self.model(image1, image2, mode='similarity')
                    rotation, translation = self.model(image1, image2, mode='pose')
                    predictions = {
                        'similarity': similarity,
                        'feat1': feat1,
                        'feat2': feat2,
                        'rotation': rotation,
                        'translation': translation
                    }

                targets = {
                    'similarity': label
                }

                # Calculate loss
                loss, loss_dict = self.criterion(predictions, targets)

                # Update metrics
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    if k not in loss_dict_sum:
                        loss_dict_sum[k] = 0
                    loss_dict_sum[k] += v

        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        for k in loss_dict_sum:
            loss_dict_sum[k] /= len(self.val_loader)

        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        for k, v in loss_dict_sum.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)

        return avg_loss, loss_dict_sum

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest checkpoint
        path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)

        # Save periodic checkpoint
        if epoch % self.config.get('save_interval', 10) == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def train(self):
        """Full training loop"""
        print(f"Starting training on device: {self.device}")

        for epoch in range(self.start_epoch, self.config.get('epochs', 100)):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate(epoch)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, is_best)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Metrics: {train_metrics}")
            print(f"Val Metrics: {val_metrics}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if self.config.get('early_stopping', None):
                if epoch - self.best_epoch > self.config['early_stopping']:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

            if is_best:
                self.best_epoch = epoch

        print("Training completed!")
        self.writer.close()