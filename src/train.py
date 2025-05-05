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
from src.trainer import Trainer
from src.loss import get_metric_learning_loss, get_enhanced_loss, CombinedLoss
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
        """Initialize training pipeline"""
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Set random seed
        seed = config.get('training', {}).get('seed')
        set_seed(seed)

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
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'dino')
        feature_dim = model_config.get('feature_dim', 512)

        # Advanced parameters
        num_layers = model_config.get('num_layers')
        multi_scale = model_config.get('multi_scale', False)
        attention_heads = model_config.get('attention_heads', 8)
        backbone = model_config.get('backbone', 'resnet50')

        if model_type == 'advanced':
            model = ImageMatchingModel(
                feature_dim=feature_dim,
                backbone=backbone,
                num_layers=num_layers,
                multi_scale=multi_scale,
                attention_heads=attention_heads
            )
            self.logger.info(f"Created advanced model with backbone={backbone}")
        elif model_type == 'loftr':
            model = LoFTRFeatureMatcher()
        elif model_type == 'superglue':
            model = SuperGlueMatchingModule()
        else:  # Default to dino
            model = DINOv2FeatureExtractor()

        # Torch compile
        advanced_config = self.config.get('advanced', {})
        if advanced_config.get('torch_compile', False):
            try:
                self.logger.info("Compiling model with torch.compile...")
                # Mode options: 'default', 'reduce-overhead', 'max-autotune'
                model = torch.compile(model, mode='reduce-overhead')
                self.logger.info("Model compilation successful")
            except Exception as e:
                self.logger.warning(f"Failed to compile model: {e}")

        # Resume from checkpoint if specified
        if self.config.get('resume'):
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
        """Create dataloaders with optimized settings"""
        data_dir = Path(self.config['data_dir'])
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size')

        # Configs for optimizing data loader
        num_workers = self.config.get('num_workers', min(os.cpu_count(), 16))
        prefetch_factor = 2
        pin_memory, persistent_workers = True, True

        # Basic transform
        transform = transforms.Compose([
            transforms.RandomResizedCrop((1024, 1024), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

        # Get dataloaders
        train_loader, val_loader = get_dataloaders(
            data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

        # Log dataloader info
        self.logger.info(f"Created dataloaders with batch_size={batch_size}, num_workers={num_workers}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}, validation samples: {len(val_loader.dataset)}")

        return train_loader, val_loader

    def _create_loss(self):
        """Create loss function based on configuration"""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'combined')

        if loss_type == 'metric_learning':
            return get_metric_learning_loss(
                loss_config.get('metric_loss'),
                temperature=loss_config.get('temperature')
            )
        elif loss_type == 'enhanced':
            return get_enhanced_loss(
                similarity_weight=loss_config.get('similarity_weight', 1.5),
                pose_weight=loss_config.get('pose_weight', 0.5),
                contrastive_margin=loss_config.get('contrastive_margin', 1.0),
                rotation_weight=loss_config.get('rotation_weight', 1.0),
                translation_weight=loss_config.get('translation_weight', 1.0),
                temperature=loss_config.get('temperature', 0.07),
                use_adaptive_weights=loss_config.get('use_adaptive_weights', True),
                use_focal_loss=loss_config.get('use_focal_loss', True)
            )
        else:
            return CombinedLoss(
                similarity_weight=loss_config.get('similarity_weight'),
                pose_weight=loss_config.get('pose_weight'),
                contrastive_margin=loss_config.get('contrastive_margin'),
                rotation_weight=loss_config.get('rotation_weight'),
                translation_weight=loss_config.get('translation_weight')
            )

    def train(self):
        """Train the model"""
        model_config = self.config.get('model')
        training_config = self.config.get('training')

        self.logger.info("Starting training...")
        self.logger.info(f"Model type: {model_config.get('type', 'dino')}")

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
    """Minimal argument parsing with hardcoded paths"""
    parser = argparse.ArgumentParser(description='Train image matching model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config():
    """Load configuration from config.yml file"""
    config_path = Path('config.yml')

    if not config_path.exists():
        print(f"Warning: Configuration file not found: {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def main():
    """Main function"""
    # Get args related to resume
    args = parse_args()

    # Load performance-related parameters from config.yml
    config = load_config()

    # Set hardcoded paths
    config['data_dir'] = './data'
    config['checkpoint_dir'] = 'checkpoints'
    config['log_dir'] = 'logs'
    config['num_workers'] = 4

    # Add resume if provided
    if args.resume:
        config['resume'] = args.resume

    # Save configuration
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(exist_ok=True, parents=True)

    try:
        with open(log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save configuration: {e}")

    # Create and run training pipeline
    pipeline = TrainingPipeline(config)
    model = pipeline.train()

    return model


if __name__ == '__main__':
    main()