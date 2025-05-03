import torch
import torchvision.transforms as transforms
from pathlib import Path
import logging
import random
import numpy as np
import os
import json
import argparse
import yaml

from src.dataset import get_dataloaders
from src.models import (
    ImageMatchingModel,
    DINOv2FeatureExtractor,
    LoFTRFeatureMatcher,
    SuperGlueMatchingModule
)
from src.trainer import (
    Trainer
)
from src.loss import get_metric_learning_loss, CombinedLoss
from src.evaluation import evaluate_model


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainingPipeline:
    """Training pipeline for image matching models"""

    def __init__(self, config):
        """
        Initialize training pipeline

        Args:
            config: Dictionary with training configuration
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Set random seed
        set_seed(config.get('seed', 42))

        # Create directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.log_dir = Path(config.get('log_dir', 'logs'))

        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Setup logging
        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger()

        # Create model
        self.model = self._create_model()

        # Create dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()

        # Create loss function
        self.criterion = self._create_loss()

        # Create trainer
        self.trainer = Trainer(self.model, self.config)

    def _create_model(self):
        """Create model based on configuration"""
        model_type = self.config.get('model_type', 'dino')
        feature_dim = self.config.get('feature_dim', 512)

        if model_type == 'dino':
            model = DINOv2FeatureExtractor(feature_dim=feature_dim)
        elif model_type == 'loftr':
            model = LoFTRFeatureMatcher()
        elif model_type == 'superglue':
            model = SuperGlueMatchingModule()
        elif model_type == 'advanced':
            backbone = self.config.get('backbone', 'resnet50')
            model = ImageMatchingModel(feature_dim=feature_dim, backbone=backbone)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Resume from checkpoint if specified
        if 'resume' in self.config and self.config['resume']:
            checkpoint_path = self.config['resume']
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load model weights
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                self.logger.warning(f"Checkpoint not found: {checkpoint_path}")

        return model

    def _create_dataloaders(self):
        """Create dataloaders with minimal augmentation"""
        data_dir = Path(self.config['data_dir'])

        # Basic augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop((1280, 1280), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Get dataloaders
        train_loader, val_loader = get_dataloaders(
            data_dir,
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', 4),
            transform=transform
        )

        return train_loader, val_loader

    def _create_loss(self):
        """Create loss function based on configuration"""
        loss_type = self.config.get('loss_type', 'combined')

        if loss_type == 'metric_learning':
            # Advanced metric learning loss
            return get_metric_learning_loss(
                self.config.get('metric_loss', 'supcon'),
                temperature=self.config.get('temperature', 0.07)
            )
        else:
            # Default combined loss
            return CombinedLoss(
                similarity_weight=self.config.get('similarity_weight', 1.0),
                pose_weight=self.config.get('pose_weight', 1.0),
                contrastive_margin=self.config.get('contrastive_margin', 1.0),
                rotation_weight=self.config.get('rotation_weight', 1.0),
                translation_weight=self.config.get('translation_weight', 1.0)
            )

    def train(self):
        """Train the model"""
        self.logger.info("Starting training...")
        self.logger.info(f"Model type: {self.config.get('model_type', 'dino')}")
        self.logger.info(f"Batch size: {self.config.get('batch_size', 32)}")
        self.logger.info(f"Learning rate: {self.config.get('learning_rate', 1e-4)}")
        self.logger.info(f"Epochs: {self.config.get('epochs', 100)}")

        # Train the model
        self.trainer.train(self.train_loader, self.val_loader, self.criterion)

        # Evaluate the model
        self.logger.info("Evaluating best model...")
        best_model_path = self.checkpoint_dir / 'checkpoint_best.pth'

        if os.path.exists(best_model_path):
            # Load best model
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Evaluate
            metrics = evaluate_model(
                self.model,
                self.val_loader,
                self.device,
                output_dir=self.log_dir
            )

            # Log metrics
            self.logger.info("Evaluation metrics:")
            for k, v in metrics.items():
                self.logger.info(f"  {k}: {v}")

        self.logger.info("Training completed!")

        return self.model


def parse_args():
    """Parse command line arguments for non-performance-related parameters only"""
    parser = argparse.ArgumentParser(description='Train image matching model')

    # Non-performance-related arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    return parser.parse_args()


def load_config():
    """Load configuration from config.yml file"""
    config_path = Path('config.yml')

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert nested dictionary to flat dictionary
    flat_config = {}

    # Model parameters
    flat_config['model_type'] = config['model']['type']  # Map 'type' to 'model_type'
    flat_config['feature_dim'] = config['model']['feature_dim']
    flat_config['backbone'] = config['model']['backbone']

    # Training parameters
    flat_config['batch_size'] = config['training']['batch_size']
    flat_config['epochs'] = config['training']['epochs']
    flat_config['learning_rate'] = config['training']['learning_rate']
    flat_config['weight_decay'] = config['training']['weight_decay']
    flat_config['optimizer'] = config['training']['optimizer']
    flat_config['warmup_epochs'] = config['training']['warmup_epochs']
    flat_config['min_learning_rate'] = config['training']['min_learning_rate']
    flat_config['seed'] = config['training']['seed']

    # Loss parameters
    flat_config['loss_type'] = config['loss']['type']
    flat_config['metric_loss'] = config['loss']['metric_loss']
    flat_config['temperature'] = config['loss']['temperature']
    flat_config['similarity_weight'] = config['loss']['similarity_weight']
    flat_config['pose_weight'] = config['loss']['pose_weight']
    flat_config['contrastive_margin'] = config['loss']['contrastive_margin']
    flat_config['rotation_weight'] = config['loss']['rotation_weight']
    flat_config['translation_weight'] = config['loss']['translation_weight']

    # Advanced parameters
    flat_config['use_mixed_precision'] = config['advanced']['use_mixed_precision']
    flat_config['use_ema'] = config['advanced']['use_ema']
    flat_config['ema_decay'] = config['advanced']['ema_decay']

    return flat_config


def main():
    """Main function"""
    # Parse command line arguments for non-performance-related parameters
    args = parse_args()

    # Load performance-related parameters from config.yml
    config = load_config()

    # Add non-performance-related parameters from command line
    config['data_dir'] = args.data_dir
    config['checkpoint_dir'] = args.checkpoint_dir
    config['log_dir'] = args.log_dir
    config['num_workers'] = args.num_workers
    config['resume'] = args.resume

    # Save configuration
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(exist_ok=True, parents=True)

    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Create training pipeline
    pipeline = TrainingPipeline(config)

    # Train model
    model = pipeline.train()

    return model


if __name__ == '__main__':
    main()