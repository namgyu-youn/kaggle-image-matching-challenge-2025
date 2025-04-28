import torch
import cv2
import logging

from src.inference import InferencePipeline as BaseInferencePipeline
from src_v2.models import DINOv2FeatureExtractor, LoFTRFeatureMatcher

class AdvancedInferencePipeline(BaseInferencePipeline):
    """Advanced inference pipeline for the Image Matching Challenge"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Set up logging
        self.logger = self._setup_logging()

        # Load models (potentially multiple for ensemble)
        self.models = self._load_models()

        # Classical feature extractor
        self.sift = cv2.SIFT_create()

        # Test-time augmentation
        self.use_tta = config.get('use_tta', False)
        self.tta_transforms = self._get_tta_transforms() if self.use_tta else None

    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger('advanced_inference')
        logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(console_handler)

        return logger

    def _load_models(self):
        """Load one or multiple models for ensemble"""
        model_paths = self.config.get('model_paths', [self.config.get('checkpoint_path')])
        loaded_models = []

        for path in model_paths:
            self.logger.info(f"Loading model from {path}")

            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            model_config = checkpoint.get('config', {})

            # Determine model type
            model_type = model_config.get('model_type', self.config.get('model_type', 'advanced'))
            feature_dim = model_config.get('feature_dim', self.config.get('feature_dim', 512))

            # Create model based on type
            if model_type == 'dino':
                model = DINOv2FeatureExtractor(
                    model_name=model_config.get('dino_model', 'dinov2_vitb14'),
                    feature_dim=feature_dim
                )
            elif model_type == 'loftr':
                model = LoFTRFeatureMatcher(
                    feature_dim=feature_dim,
                    n_heads=model_config.get('n_heads', 8),
                    n_layers=model_config.get('n_layers', 6)
                )
            else:
                # Default to base model
                from src.models import ImageMatchingModel
                model = ImageMatchingModel(feature_dim=feature_dim)

            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            loaded_models.append(model)

        return loaded_models