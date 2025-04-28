import argparse
import json
from pathlib import Path
import torch
import logging
from datetime import datetime
import random
import numpy as np

from src.dataset import get_dataloaders
from src.models import ImageMatchingModel
from src_v2.models import DINOv2FeatureExtractor, LoFTRFeatureMatcher
from src_v2.trainer import (
    MixedPrecisionTrainer, EMAModel, WarmupCosineScheduler, get_advanced_optimizer
)
from src_v2.augmentation import get_augmentation_transform
from src_v2.metric_learning import get_metric_learning_loss


class AdvancedTrainer:
    """Advanced trainer for Image Matching Challenge 2025"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Set random seeds for reproducibility
        random.seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        torch.manual_seed(config.get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.get('seed', 42))

        # Setup logging
        self.logger = self._setup_logging()

        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Create EMA model if enabled
        if config.get('use_ema', True):
            self.ema_model = EMAModel(self.model, decay=config.get('ema_decay', 0.999))
        else:
            self.ema_model = None

        # Setup optimizer and scheduler
        self.optimizer = get_advanced_optimizer(self.model, config)

        # Setup learning rate scheduler
        total_epochs = config.get('epochs', 100)
        warmup_epochs = config.get('warmup_epochs', 10)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=config.get('min_lr', 1e-6)
        )

        # Setup loss function
        self.criterion = self._create_loss_function()

        # Mixed precision training
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        if self.use_mixed_precision:
            self.mp_trainer = MixedPrecisionTrainer(self.model, self.optimizer, self.device)

        # Setup data loaders with advanced augmentation
        self.train_loader, self.val_loader = self._create_dataloaders()

        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Resume from checkpoint if specified
        if config.get('resume', None):
            self.load_checkpoint(config['resume'])

    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger('advanced_trainer')
        logger.setLevel(logging.INFO)

        # Create file handler
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _create_model(self):
        """Create model based on configuration"""
        model_type = self.config.get('model_type', 'advanced')
        feature_dim = self.config.get('feature_dim', 512)

        if model_type == 'dino':
            # Create model with DINOv2 feature extractor
            return DINOv2FeatureExtractor(
                model_name=self.config.get('dino_model', 'dinov2_vitb14'),
                feature_dim=feature_dim
            )
        elif model_type == 'loftr':
            # Create LoFTR-style matcher
            return LoFTRFeatureMatcher(
                feature_dim=feature_dim,
                n_heads=self.config.get('n_heads', 8),
                n_layers=self.config.get('n_layers', 6)
            )
        else:
            # Default to base ImageMatchingModel
            return ImageMatchingModel(feature_dim=feature_dim)

    def _create_loss_function(self):
        """Create loss function based on configuration"""
        loss_type = self.config.get('loss_type', 'combined')

        if loss_type == 'arcface':
            return get_metric_learning_loss(
                'arcface',
                embedding_size=self.config.get('feature_dim', 512),
                num_classes=self.config.get('num_classes', 100),
                margin=self.config.get('arcface_margin', 0.5),
                scale=self.config.get('arcface_scale', 64)
            )
        elif loss_type == 'triplet':
            return get_metric_learning_loss(
                'triplet_hard',
                margin=self.config.get('triplet_margin', 0.3),
                soft=self.config.get('triplet_soft', False)
            )
        elif loss_type == 'supcon':
            return get_metric_learning_loss(
                'supcon',
                temperature=self.config.get('temperature', 0.07)
            )
        else:
            # Default combined loss from base version
            from image_matching_challenge_2025.loss import CombinedLoss
            return CombinedLoss(
                similarity_weight=self.config.get('similarity_weight', 1.0),
                pose_weight=self.config.get('pose_weight', 1.0),
                contrastive_margin=self.config.get('contrastive_margin', 1.0),
                rotation_weight=self.config.get('rotation_weight', 1.0),
                translation_weight=self.config.get('translation_weight', 1.0)
            )

    def _create_dataloaders(self):
        """Create dataloaders with advanced augmentation"""
        data_dir = Path(self.config['data_dir'])

        # Get train transform with advanced augmentation
        strong_aug = self.config.get('strong_augmentation', True)
        transform = get_augmentation_transform(mode='train', strong_aug=strong_aug)

        # Use base dataloaders but with our advanced transformations
        return get_dataloaders(
            str(data_dir),
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', 4),
            transform=transform
        )

    def train_epoch(self, epoch):
        """Train for one epoch with advanced techniques"""
        self.model.train()
        total_loss = 0

        # Progress bar
        pbar = self.train_loader
        if self.config.get('use_tqdm', True):
            from tqdm import tqdm
            pbar = tqdm(pbar, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Forward and backward pass with mixed precision if enabled
            if self.use_mixed_precision:
                loss = self.mp_trainer.train_step(batch, self.criterion)
            else:
                # Regular training
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.get('gradient_clip', None):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )

                self.optimizer.step()
                loss = loss.item()

            # Update EMA model if enabled
            if self.ema_model is not None:
                self.ema_model.update()

            # Update progress bar
            if self.config.get('use_tqdm', True):
                pbar.set_postfix({'loss': loss})

            # Log to logger
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss:.6f}')

            total_loss += loss

        # Average loss
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def validate(self, epoch):
        """Validation loop with EMA model if available"""
        # Use EMA model for validation if available
        if self.ema_model is not None:
            self.ema_model.apply_shadow()

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)

                total_loss += loss.item()

        # Restore original model weights if using EMA
        if self.ema_model is not None:
            self.ema_model.restore()

        # Average loss
        avg_loss = total_loss / len(self.val_loader)

        self.logger.info(f'Validation Epoch: {epoch} Average loss: {avg_loss:.6f}')

        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save EMA model state if available
        if self.ema_model is not None:
            self.ema_model.apply_shadow()
            checkpoint['ema_state_dict'] = self.model.state_dict()
            self.ema_model.restore()

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
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load EMA state if available
        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.apply_shadow()
            self.model.load_state_dict(checkpoint['ema_state_dict'])
            self.ema_model.restore()

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']

        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def train(self):
        """Full training loop with advanced features"""
        self.logger.info(f"Starting advanced training on device: {self.device}")
        self.logger.info(f"Model type: {self.config.get('model_type', 'advanced')}")
        self.logger.info(f"Total epochs: {self.config.get('epochs', 100)}")

        for epoch in range(self.start_epoch, self.config.get('epochs', 100)):
            # Update learning rate with warmup scheduler
            self.scheduler.step(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch}: learning rate = {current_lr:.6f}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Print epoch summary
            self.logger.info(f"Epoch {epoch} Summary:")
            self.logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            self.logger.info(f"Best Val Loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch})")

            # Early stopping
            if self.config.get('early_stopping', None):
                if epoch - self.best_epoch > self.config['early_stopping']:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

        self.logger.info("Training completed!")
        self.logger.info(f"Best model at epoch {self.best_epoch} with validation loss {self.best_val_loss:.6f}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Training for Image Matching Challenge')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the training data directory')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='dino',
                        choices=['dino', 'loftr', 'advanced'],
                        help='Type of advanced model to use')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension')
    parser.add_argument('--dino_model', type=str, default='dinov2_vitb14',
                        help='DINOv2 model variant')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # Loss parameters
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['combined', 'arcface', 'triplet', 'supcon'],
                        help='Type of loss function')

    # Advanced features
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use exponential moving average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')
    parser.add_argument('--strong_augmentation', action='store_true',
                        help='Use strong data augmentation')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')

    # Paths
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')

    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (JSON)')

    return parser.parse_args()


def load_config(args):
    """Load configuration from file if provided"""
    config = vars(args)

    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)

        # Update config with file values
        config.update(file_config)

    return config


def main():
    # Parse arguments
    args = parse_arguments()

    # Load config
    config = load_config(args)

    # Set device
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config['device'] = 'cpu'

    print("Advanced Configuration:")
    print(json.dumps(config, indent=2))

    # Create directories
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = AdvancedTrainer(config)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()