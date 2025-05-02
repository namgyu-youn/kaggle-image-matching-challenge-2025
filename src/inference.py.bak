import torch
import logging
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class InferencePipeline:
    """Inference pipeline for image matching models"""

    def __init__(self, model, config):
        """
        Initialize inference pipeline

        Args:
            model: Trained model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set up transforms
        self.transform = self._get_transform()

        # Set up logging
        self.logger = self._setup_logging()

        # Test-time augmentation
        self.use_tta = config.get('use_tta', False)
        self.tta_transforms = self._get_tta_transforms() if self.use_tta else None

    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger('inference')
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

    def _get_transform(self):
        """Get image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _get_tta_transforms(self):
        """Get test-time augmentation transforms"""
        return [
            # Original
            transforms.Compose([
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((480, 640)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ]),
            # Color jitter
            transforms.Compose([
                transforms.Resize((480, 640)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        ]

    def load_image(self, image_path):
        """
        Load and preprocess an image

        Args:
            image_path: Path to image

        Returns:
            tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

    def predict_similarity(self, image1_path, image2_path):
        """
        Predict similarity between two images

        Args:
            image1_path: Path to first image
            image2_path: Path to second image

        Returns:
            similarity: Similarity score between 0 and 1
        """
        # Load images
        img1 = self.load_image(image1_path).to(self.device)
        img2 = self.load_image(image2_path).to(self.device)

        # Create batch
        batch = {
            'image1': img1,
            'image2': img2
        }

        # Predict
        with torch.no_grad():
            outputs = self.model(batch)

            # Get similarity score
            if isinstance(outputs, dict):
                similarity = outputs['similarity'].squeeze().cpu().item()
            else:
                similarity = outputs[0].squeeze().cpu().item()

        return similarity

    def predict_similarity_batch(self, image_pairs):
        """
        Predict similarity for a batch of image pairs

        Args:
            image_pairs: List of (image1_path, image2_path) tuples

        Returns:
            similarities: List of similarity scores
        """
        batch_size = self.config.get('batch_size', 16)
        similarities = []

        # Process in batches
        for i in range(0, len(image_pairs), batch_size):
            batch_pairs = image_pairs[i:i+batch_size]

            # Load images
            img1_batch = []
            img2_batch = []

            for img1_path, img2_path in batch_pairs:
                img1 = self.load_image(img1_path).to(self.device)
                img2 = self.load_image(img2_path).to(self.device)

                img1_batch.append(img1)
                img2_batch.append(img2)

            # Stack tensors
            img1_batch = torch.cat(img1_batch, dim=0)
            img2_batch = torch.cat(img2_batch, dim=0)

            # Create batch
            batch = {
                'image1': img1_batch,
                'image2': img2_batch
            }

            # Predict
            with torch.no_grad():
                outputs = self.model(batch)

                # Get similarity scores
                if isinstance(outputs, dict):
                    batch_similarities = outputs['similarity'].squeeze().cpu().numpy()
                else:
                    batch_similarities = outputs[0].squeeze().cpu().numpy()

                # Handle single item case
                if len(batch_pairs) == 1:
                    batch_similarities = [batch_similarities.item()]

                similarities.extend(batch_similarities.tolist())

        return similarities

    def extract_features(self, image_path):
        """
        Extract features from an image

        Args:
            image_path: Path to image

        Returns:
            features: Feature vector
        """
        # Load image
        img = self.load_image(image_path).to(self.device)

        # Extract features
        with torch.no_grad():
            if hasattr(self.model, 'extract_features'):
                features = self.model.extract_features(img)
            elif hasattr(self.model, 'feature_extractor'):
                features = self.model.feature_extractor(img)
            else:
                # Try to get features from forward pass
                outputs = self.model({'image1': img, 'image2': img})
                if isinstance(outputs, dict) and 'feat1' in outputs:
                    features = outputs['feat1']
                else:
                    raise ValueError("Model doesn't support feature extraction")

        return features.cpu()

    def extract_features_batch(self, image_paths):
        """
        Extract features from a batch of images

        Args:
            image_paths: List of image paths

        Returns:
            features: List of feature vectors
        """
        batch_size = self.config.get('batch_size', 16)
        all_features = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]

            # Load images
            img_batch = []

            for img_path in batch_paths:
                img = self.load_image(img_path).to(self.device)
                img_batch.append(img)

            # Stack tensors
            img_batch = torch.cat(img_batch, dim=0)

            # Extract features
            with torch.no_grad():
                if hasattr(self.model, 'extract_features'):
                    features = self.model.extract_features(img_batch)
                elif hasattr(self.model, 'feature_extractor'):
                    features = self.model.feature_extractor(img_batch)
                else:
                    # Try to get features from forward pass
                    outputs = self.model({'image1': img_batch, 'image2': img_batch})
                    if isinstance(outputs, dict) and 'feat1' in outputs:
                        features = outputs['feat1']
                    else:
                        raise ValueError("Model doesn't support feature extraction")

            all_features.append(features.cpu())

        # Concatenate features
        all_features = torch.cat(all_features, dim=0)

        return all_features

    def predict_with_tta(self, image1_path, image2_path):
        """
        Predict with test-time augmentation

        Args:
            image1_path: Path to first image
            image2_path: Path to second image

        Returns:
            similarity: Similarity score between 0 and 1
        """
        # Load images
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')

        similarities = []

        # Apply each TTA transform
        for transform in self.tta_transforms:
            img1_tensor = transform(img1).unsqueeze(0).to(self.device)
            img2_tensor = transform(img2).unsqueeze(0).to(self.device)

            # Create batch
            batch = {
                'image1': img1_tensor,
                'image2': img2_tensor
            }

            # Predict
            with torch.no_grad():
                outputs = self.model(batch)

                # Get similarity score
                if isinstance(outputs, dict):
                    similarity = outputs['similarity'].squeeze().cpu().item()
                else:
                    similarity = outputs[0].squeeze().cpu().item()

                similarities.append(similarity)

        # Average similarities
        return np.mean(similarities)

    def visualize_similarity(self, image1_path, image2_path, save_path=None):
        """
        Visualize similarity prediction

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            save_path: Path to save visualization
        """
        # Load images
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')

        # Predict similarity
        similarity = self.predict_similarity(image1_path, image2_path)

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Display images
        ax[0].imshow(img1)
        ax[1].imshow(img2)

        # Remove axes
        ax[0].axis('off')
        ax[1].axis('off')

        # Set title
        fig.suptitle(f'Similarity: {similarity:.4f}', fontsize=16)

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def process_dataset(self, dataset_dir, output_dir=None):
        """
        Process a dataset of images

        Args:
            dataset_dir: Directory containing images
            output_dir: Directory to save results

        Returns:
            results: Dictionary with results
        """
        dataset_dir = Path(dataset_dir)

        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

        # Find all images
        image_paths = list(dataset_dir.glob('**/*.png')) + list(dataset_dir.glob('**/*.jpg'))
        self.logger.info(f"Found {len(image_paths)} images in {dataset_dir}")

        # Extract features
        self.logger.info("Extracting features...")
        features = self.extract_features_batch(image_paths)

        # Compute similarity matrix
        self.logger.info("Computing similarity matrix...")
        similarity_matrix = torch.matmul(features, features.t()).cpu().numpy()

        # Save results
        results = {
            'image_paths': [str(p) for p in image_paths],
            'features': features.numpy(),
            'similarity_matrix': similarity_matrix
        }

        if output_dir:
            np.save(output_dir / 'features.npy', features.numpy())
            np.save(output_dir / 'similarity_matrix.npy', similarity_matrix)

            # Save image paths
            with open(output_dir / 'image_paths.txt', 'w') as f:
                for path in image_paths:
                    f.write(f"{path}\n")

        return results


def load_model_for_inference(model_path, model_type='dino', device='cuda'):
    """
    Load a trained model for inference

    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('dino', 'loftr', 'superglue', 'advanced')
        device: Device to load model on

    Returns:
        model: Loaded model
    """
    # Import models
    from src.models import (
        DINOv2FeatureExtractor,
        LoFTRFeatureMatcher,
        SuperGlueMatchingModule,
        ImageMatchingModel
    )

    # Create model based on type
    if model_type == 'dino':
        model = DINOv2FeatureExtractor(feature_dim=512)
    elif model_type == 'loftr':
        model = LoFTRFeatureMatcher()
    elif model_type == 'superglue':
        model = SuperGlueMatchingModule()
    elif model_type == 'advanced':
        model = ImageMatchingModel(feature_dim=512, backbone='resnet50')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model


def create_inference_pipeline(model_path, model_type='dino', config=None):
    """
    Create an inference pipeline

    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('dino', 'loftr', 'superglue', 'advanced')
        config: Configuration dictionary

    Returns:
        pipeline: Inference pipeline
    """
    # Default config
    if config is None:
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': 16,
            'use_tta': False
        }

    # Load model
    model = load_model_for_inference(model_path, model_type, config['device'])

    # Create pipeline
    pipeline = InferencePipeline(model, config)

    return pipeline