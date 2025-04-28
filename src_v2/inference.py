import torch
import cv2
import logging
import numpy as np

from src.inference import InferencePipeline as BaseInferencePipeline
from src_v2.models import DINOv2FeatureExtractor, LoFTRFeatureMatcher
from src_v2.evaluation import evaluate_submission
from src_v2.clustering import cluster_images
from src_v2.pose_estimation import reconstruct_scene
from src_v2.submission import create_submission_file

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

    def cluster_and_reconstruct(self, features, image_ids, matches):
        """
        Cluster images and reconstruct 3D scenes.

        Args:
            features: Feature vectors for images
            image_ids: List of image ids
            matches: Dict of matches between image pairs

        Returns:
            clusters: Dict mapping cluster ids to lists of image ids
            poses: Dict mapping image ids to camera poses
            outliers: List of outlier image ids
        """
        # Compute similarity matrix
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        features_norm = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)
        similarity_matrix = np.dot(features_norm, features_norm.T)

        # Cluster images
        self.logger.info("Clustering images...")
        clusters, outliers = cluster_images(
            features_np,
            image_ids,
            method=self.config.get('clustering_method', 'hierarchical'),
            similarity_matrix=similarity_matrix,
            threshold=self.config.get('clustering_threshold', 0.7),
            outlier_threshold=self.config.get('outlier_threshold', 0.6)
        )

        self.logger.info(f"Found {len(clusters)} clusters and {len(outliers)} outliers")

        # Reconstruct each scene
        poses = {}
        intrinsics = self.config.get('camera_intrinsics', None)

        if intrinsics is None:
            # Default camera intrinsics (can be adjusted)
            fx = fy = 1000
            cx = 640 / 2
            cy = 480 / 2
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

        for cluster_id, cluster_imgs in clusters.items():
            self.logger.info(f"Reconstructing cluster {cluster_id} with {len(cluster_images)} images...")

            # Filter matches for this cluster
            cluster_matches = {}
            for (img1, img2), match_data in matches.items():
                if img1 in cluster_images and img2 in cluster_images:
                    cluster_matches[(img1, img2)] = match_data

            # Extract features for this cluster
            cluster_features = {img_id: features_np[image_ids.index(img_id)]
                                for img_id in cluster_images}

            # Reconstruct scene
            cluster_poses, _, _ = reconstruct_scene(
                cluster_images,
                cluster_features,
                cluster_matches,
                intrinsics,
                min_inliers=self.config.get('min_inliers', 15)
            )

            # Add poses to global poses
            poses.update(cluster_poses)

        return clusters, poses, outliers

    def create_submission(self, dataset_results, output_path):
        """
        Create submission file from results.

        Args:
            dataset_results: Dict mapping dataset ids to results
            output_path: Path to save submission file
        """
        create_submission_file(dataset_results, output_path)
        self.logger.info(f"Submission saved to {output_path}")

    def evaluate_results(self, submission_path, gt_path):
        """
        Evaluate results against ground truth.

        Args:
            submission_path: Path to submission file
            gt_path: Path to ground truth file

        Returns:
            score: Overall score
            dataset_scores: Scores for each dataset
        """
        self.logger.info("Evaluating submission...")
        score, dataset_scores = evaluate_submission(submission_path, gt_path)

        self.logger.info(f"Overall score: {score:.4f}")

        for dataset_id, scores in dataset_scores.items():
            self.logger.info(f"Dataset {dataset_id}: mAA={scores['mAA']:.4f}, "
                            f"clustering={scores['clustering']:.4f}, "
                            f"combined={scores['combined']:.4f}")

        return score, dataset_scores