import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import argparse
import json
from collections import defaultdict

from image_matching_challenge_2025.models import ImageMatchingModel, AdvancedMatchingModel
from image_matching_challenge_2025.dataset import ImageMatchingDataset
from image_matching_challenge_2025.functions import (
    extract_sift_features, match_features, estimate_fundamental_matrix,
    compute_similarity_matrix, spectral_clustering, detect_outliers,
    estimate_camera_intrinsics,
    compute_pose_from_essential, filter_matches_epipolar
)


class InferencePipeline:
    """Inference pipeline for the Image Matching Challenge"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Classical feature extractor
        self.sift = cv2.SIFT_create()

    def _load_model(self):
        """Load trained model"""
        if self.config.get('model_type', 'basic') == 'advanced':
            model = AdvancedMatchingModel(feature_dim=self.config.get('feature_dim', 512))
        else:
            model = ImageMatchingModel(feature_dim=self.config.get('feature_dim', 512))

        checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        return model

    def extract_features(self, dataset):
        """Extract features for all images"""
        features = []
        image_paths = []

        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Extracting features"):
                item = dataset[i]
                image = item['image'].unsqueeze(0).to(self.device)

                if hasattr(self.model, 'feature_extractor'):
                    feat = self.model.feature_extractor(image)
                else:
                    feat = self.model.similarity_net.feature_extractor(image)

                features.append(feat.cpu().numpy())
                image_paths.append(item['path'])

        features = np.vstack(features)
        return features, image_paths

    def compute_similarity_scores(self, dataset):
        """Compute pairwise similarity scores"""
        n_images = len(dataset)
        similarity_matrix = np.zeros((n_images, n_images))

        with torch.no_grad():
            for i in tqdm(range(n_images), desc="Computing similarities"):
                for j in range(i+1, n_images):
                    item_i = dataset[i]
                    item_j = dataset[j]

                    if item_i['dataset_id'] != item_j['dataset_id']:
                        continue

                    img_i = item_i['image'].unsqueeze(0).to(self.device)
                    img_j = item_j['image'].unsqueeze(0).to(self.device)

                    if isinstance(self.model, AdvancedMatchingModel):
                        similarity, _, _ = self.model(img_i, img_j)
                    else:
                        similarity, _, _ = self.model(img_i, img_j, mode='similarity')

                    sim_score = similarity.item()
                    similarity_matrix[i, j] = sim_score
                    similarity_matrix[j, i] = sim_score

        return similarity_matrix

    def cluster_images(self, features, similarity_matrix, dataset):
        """Cluster images by dataset"""
        clusters = {}
        outliers = {}

        dataset_indices = defaultdict(list)
        for i, item in enumerate(dataset.dataset_ids):
            dataset_indices[item].append(i)

        for dataset_id, indices in dataset_indices.items():
            # Extract sub-matrix for this dataset
            sub_features = features[indices]
            sub_similarity = similarity_matrix[indices][:, indices]

            # Detect outliers
            outlier_mask = detect_outliers(sub_features, sub_similarity,
                                         threshold=self.config.get('outlier_threshold', 0.1))

            # Cluster non-outlier images
            non_outlier_indices = np.where(~outlier_mask)[0]

            if len(non_outlier_indices) > 0:
                # Perform spectral clustering
                non_outlier_similarity = sub_similarity[non_outlier_indices][:, non_outlier_indices]

                labels, n_clusters = spectral_clustering(
                    non_outlier_similarity,
                    n_clusters=self.config.get('n_clusters', None),
                    max_clusters=self.config.get('max_clusters', 10)
                )

                # Map back to original indices
                cluster_assignments = {}
                for i, label in enumerate(labels):
                    orig_idx = indices[non_outlier_indices[i]]
                    cluster_assignments[orig_idx] = label

                clusters[dataset_id] = cluster_assignments
            else:
                clusters[dataset_id] = {}

            # Store outliers
            outlier_indices = np.where(outlier_mask)[0]
            outliers[dataset_id] = [indices[i] for i in outlier_indices]

        return clusters, outliers

    def estimate_poses(self, dataset, clusters):
        """Estimate camera poses for clustered images"""
        poses = {}

        for dataset_id, cluster_assignments in clusters.items():
            poses[dataset_id] = {}

            # Group images by cluster
            cluster_to_images = defaultdict(list)
            for img_idx, cluster_id in cluster_assignments.items():
                cluster_to_images[cluster_id].append(img_idx)

            # Process each cluster
            for cluster_id, image_indices in cluster_to_images.items():
                if len(image_indices) < 2:
                    # Cannot estimate pose for single image
                    for idx in image_indices:
                        poses[dataset_id][idx] = {
                            'rotation': None,
                            'translation': None
                        }
                    continue

                # Use first image as reference
                ref_idx = image_indices[0]
                ref_item = dataset[ref_idx]
                ref_image = ref_item['image']

                # Initialize reference pose
                poses[dataset_id][ref_idx] = {
                    'rotation': np.eye(3),
                    'translation': np.zeros(3)
                }

                # Estimate poses for other images relative to reference
                for idx in image_indices[1:]:
                    item = dataset[idx]
                    image = item['image']

                    # Extract SIFT features
                    kp1, desc1 = extract_sift_features(ref_image)
                    kp2, desc2 = extract_sift_features(image)

                    # Match features
                    matches = match_features(desc1, desc2)

                    if len(matches) < 8:
                        # Not enough matches
                        poses[dataset_id][idx] = {
                            'rotation': None,
                            'translation': None
                        }
                        continue

                    # Estimate fundamental matrix
                    F, mask = estimate_fundamental_matrix(kp1, kp2, matches)

                    if F is None:
                        poses[dataset_id][idx] = {
                            'rotation': None,
                            'translation': None
                        }
                        continue

                    # Filter matches with epipolar constraint
                    good_matches = filter_matches_epipolar(kp1, kp2, matches, F)

                    # Estimate camera intrinsics
                    K = estimate_camera_intrinsics(ref_image.shape)

                    # Compute essential matrix
                    E = K.T @ F @ K

                    # Recover pose
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                    R, t = compute_pose_from_essential(E, pts1, pts2, K)

                    poses[dataset_id][idx] = {
                        'rotation': R,
                        'translation': t
                    }

        return poses

    def predict_with_model(self, dataset, clusters):
        """Use neural network to predict poses"""
        poses = {}

        with torch.no_grad():
            for dataset_id, cluster_assignments in clusters.items():
                poses[dataset_id] = {}

                # Group images by cluster
                cluster_to_images = defaultdict(list)
                for img_idx, cluster_id in cluster_assignments.items():
                    cluster_to_images[cluster_id].append(img_idx)

                # Process each cluster
                for cluster_id, image_indices in cluster_to_images.items():
                    if len(image_indices) < 2:
                        for idx in image_indices:
                            poses[dataset_id][idx] = {
                                'rotation': None,
                                'translation': None
                            }
                        continue

                    # Use first image as reference
                    ref_idx = image_indices[0]
                    ref_item = dataset[ref_idx]
                    ref_image = ref_item['image'].unsqueeze(0).to(self.device)

                    # Initialize reference pose
                    poses[dataset_id][ref_idx] = {
                        'rotation': np.eye(3),
                        'translation': np.zeros(3)
                    }

                    # Estimate poses for other images
                    for idx in image_indices[1:]:
                        item = dataset[idx]
                        image = item['image'].unsqueeze(0).to(self.device)

                        # Predict relative pose
                        if isinstance(self.model, AdvancedMatchingModel):
                            _, rotation, translation = self.model(ref_image, image)
                        else:
                            rotation, translation = self.model(ref_image, image, mode='pose')

                        poses[dataset_id][idx] = {
                            'rotation': rotation.squeeze().cpu().numpy(),
                            'translation': translation.squeeze().cpu().numpy()
                        }

        return poses

    def run_inference(self, data_dir, output_path):
        """Run full inference pipeline"""
        print("Loading dataset...")
        dataset = ImageMatchingDataset(data_dir, mode='test')

        print("Extracting features...")
        features, image_paths = self.extract_features(dataset)

        print("Computing similarities...")
        if self.config.get('use_neural_similarity', True):
            similarity_matrix = self.compute_similarity_scores(dataset)
        else:
            similarity_matrix = compute_similarity_matrix(features)

        print("Clustering images...")
        clusters, outliers = self.cluster_images(features, similarity_matrix, dataset)

        print("Estimating poses...")
        if self.config.get('use_neural_pose', True):
            poses = self.predict_with_model(dataset, clusters)
        else:
            poses = self.estimate_poses(dataset, clusters)

        print("Creating submission...")
        submission_data = self.create_submission(dataset, clusters, outliers, poses)

        # Save submission
        df = pd.DataFrame(submission_data)
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

    def create_submission(self, dataset, clusters, outliers, poses):
        """Create submission file"""
        submission_data = []

        for i in range(len(dataset)):
            item = dataset[i]
            dataset_id = item['dataset_id']
            img_name = item['filename']

            # Check if outlier
            if i in outliers.get(dataset_id, []):
                submission_data.append({
                    'dataset': dataset_id,
                    'scene': 'outliers',
                    'image': img_name,
                    'rotation_matrix': ';'.join(['nan'] * 9),
                    'translation_vector': ';'.join(['nan'] * 3)
                })
            else:
                # Get cluster ID
                cluster_id = clusters[dataset_id].get(i, None)
                if cluster_id is None:
                    scene = 'outliers'
                    rotation_str = ';'.join(['nan'] * 9)
                    translation_str = ';'.join(['nan'] * 3)
                else:
                    scene = f"cluster{cluster_id}"
                    pose = poses[dataset_id].get(i, {})

                    if pose.get('rotation') is not None:
                        rotation_str = ';'.join([f"{val:.6f}" for val in pose['rotation'].flatten()])
                    else:
                        rotation_str = ';'.join(['nan'] * 9)

                    if pose.get('translation') is not None:
                        translation_str = ';'.join([f"{val:.6f}" for val in pose['translation']])
                    else:
                        translation_str = ';'.join(['nan'] * 3)

                submission_data.append({
                    'dataset': dataset_id,
                    'scene': scene,
                    'image': img_name,
                    'rotation_matrix': rotation_str,
                    'translation_vector': translation_str
                })

        return submission_data


def main():
    parser = argparse.ArgumentParser(description='Image Matching Challenge Inference')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output file path')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'model_type': 'advanced',
            'feature_dim': 512,
            'use_neural_similarity': True,
            'use_neural_pose': True,
            'outlier_threshold': 0.1,
            'max_clusters': 10
        }

    config['checkpoint_path'] = args.checkpoint
    config['device'] = args.device

    # Run inference
    pipeline = InferencePipeline(config)
    pipeline.run_inference(args.data_dir, args.output)


if __name__ == '__main__':
    main()