import argparse
import json
from pathlib import Path
import torch
# import yaml  # Optional if you want to use YAML configs

from image_matching_challenge_2025.trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Image Matching Model')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the training data directory')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='advanced',
                        choices=['basic', 'advanced'],
                        help='Type of model to use')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau'],
                        help='Learning rate scheduler')

    # Loss weights
    parser.add_argument('--similarity_weight', type=float, default=1.0,
                        help='Weight for similarity loss')
    parser.add_argument('--pose_weight', type=float, default=1.0,
                        help='Weight for pose estimation loss')
    parser.add_argument('--contrastive_margin', type=float, default=1.0,
                        help='Margin for contrastive loss')
    parser.add_argument('--rotation_weight', type=float, default=1.0,
                        help='Weight for rotation loss')
    parser.add_argument('--translation_weight', type=float, default=1.0,
                        help='Weight for translation loss')

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval for tensorboard')

    # Paths
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')

    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (YAML or JSON)')

    return parser.parse_args()


def load_config(args):
    """Load configuration from file if provided"""
    config = vars(args)

    if args.config:
        ext = Path(args.config).suffix.lower()
        with open(args.config, 'r') as f:
            if ext == '.yaml' or ext == '.yml':
                try:
                    import yaml
                    file_config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML not installed. Please install with 'pip install pyyaml' or use JSON configs.")
            elif ext == '.json':
                file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {ext}")

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

    print("Configuration:")
    print(json.dumps(config, indent=2))

    # Create directories
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = Trainer(config)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()