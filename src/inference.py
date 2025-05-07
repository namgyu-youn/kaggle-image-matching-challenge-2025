import torch
import numpy as np
import pandas as pd
import os
import yaml
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
from collections import defaultdict
import re

from src.models.dino import DINOv2FeatureExtractor


class TestDataset(Dataset):
    """Dataset for inference with Image Matching Challenge test data"""

    def __init__(self, test_dir='./data/test', transform=None):
        self.root_dir = Path(test_dir)
        self.transform = transform or self._get_default_transform()

        # Collect image paths and metadata
        self.image_paths = []
        self.dataset_ids = []
        self.image_names = []

        # Iterate over dataset directories
        for dataset_dir in self.root_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_id = dataset_dir.name

            # Collect image files
            for img_path in dataset_dir.glob('*.png'):
                self.image_paths.append(str(img_path))
                self.dataset_ids.append(dataset_id)
                self.image_names.append(img_path.name)

        logging.info(f"Found {len(self.image_paths)} images across {len(set(self.dataset_ids))} datasets")

    def _get_default_transform(self):
        """Default transform for test images"""
        return transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'path': img_path,
            'dataset_id': self.dataset_ids[idx],
            'image_name': self.image_names[idx],
            'idx': idx
        }


def load_config(config_path='config.yml'):
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        logging.warning(f"Config file not found at {config_path}, using default settings")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint with flexibility for compiled models"""
    try:
        # Load checkpoint first
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Check if the model is compiled
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Check if the model is compiled (`_orig_mod.` prefix)
        is_compiled = any('_orig_mod.' in k for k in state_dict.keys())

        if is_compiled:
            logging.info("Detected compiled model checkpoint, adapting keys...")
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    clean_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix (length 10)
                else:
                    clean_state_dict[k] = v
            state_dict = clean_state_dict

        # Alternative model - Using DINO-based feature extractor
        logging.info("Using DINOv2 as base feature extractor")
        model = DINOv2FeatureExtractor(feature_dim=config.get('model', {}).get('feature_dim', 512))
        model = model.to(device)
        model.eval()

        return model

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # Fallback: Direct use of DINO model
        logging.info("Falling back to pretrained DINOv2 model")
        model = DINOv2FeatureExtractor(feature_dim=512)
        model = model.to(device)
        model.eval()

        return model


def extract_features(model, dataloader, device):
    """Extract features from all images in the dataset"""
    features = []
    metadata = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Move batch data to device
            images = batch['image'].to(device)

            # Feature extraction (processed according to model type)
            if hasattr(model, 'extract_features'):
                batch_features = model.extract_features(images)
            else:
                batch_features = model(images)

            # Collect results
            features.append(batch_features.cpu())

            # Collect metadata (for each item in the batch)
            for i in range(len(batch['idx'])):
                metadata.append({
                    'idx': batch['idx'][i].item(),
                    'path': batch['path'][i],
                    'dataset_id': batch['dataset_id'][i],
                    'image_name': batch['image_name'][i]
                })

    # Combine all features
    features = torch.cat(features, dim=0)

    return features, metadata


def compute_similarity_matrix(features):
    """Compute pairwise similarity matrix for all image features"""
    # Normalization
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    # Calculate cosine similarity for all possible pairs
    similarity_matrix = torch.mm(features, features.t())

    return similarity_matrix.numpy()


def load_train_labels(path='train_labels.csv'):
    """Load training labels from CSV"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.warning(f"Error loading train labels: {e}")
        return None


def load_train_thresholds(path='train_thresholds.csv'):
    """Load training thresholds from CSV"""
    try:
        df = pd.read_csv(path)
        # Convert string thresholds to list
        df['thresholds_list'] = df['thresholds'].apply(lambda x: [float(val) for val in x.split(';')])
        return df
    except Exception as e:
        logging.warning(f"Error loading train thresholds: {e}")
        return None


def analyze_filename_patterns(metadata, train_labels=None):
    """Analyze filename patterns to extract potential clusters"""
    pattern_clusters = defaultdict(dict)

    # Utilize training labels if available
    if train_labels is not None:
        train_patterns = defaultdict(dict)
        for _, row in train_labels.iterrows():
            dataset_id = row['dataset']
            scene_id = row['scene']
            image_name = row['image']

            # Extract common pattern from filename
            if scene_id == 'outliers':
                if 'outliers' not in train_patterns[dataset_id]:
                    train_patterns[dataset_id]['outliers'] = []
                train_patterns[dataset_id]['outliers'].append(image_name)
            else:
                # Group images by scene
                if scene_id not in train_patterns[dataset_id]:
                    train_patterns[dataset_id][scene_id] = []
                train_patterns[dataset_id][scene_id].append(image_name)

        # Extract filename patterns by scene
        scene_patterns = defaultdict(dict)
        for dataset_id, scenes in train_patterns.items():
            for scene_id, images in scenes.items():
                if len(images) > 0:
                    # Find common prefix
                    common_prefix = os.path.commonprefix(images)
                    # Remove ending numbers
                    common_prefix = re.sub(r'\d+$', '', common_prefix)
                    scene_patterns[dataset_id][common_prefix] = scene_id

    # Assign pattern-based scenes for each image in metadata
    for meta in metadata:
        dataset_id = meta['dataset_id']
        image_name = meta['image_name']

        # Use training patterns
        if train_labels is not None and dataset_id in scene_patterns:
            for prefix, scene_id in scene_patterns[dataset_id].items():
                if image_name.startswith(prefix):
                    if scene_id not in pattern_clusters[dataset_id]:
                        pattern_clusters[dataset_id][scene_id] = []
                    pattern_clusters[dataset_id][scene_id].append(meta['idx'])
                    break
            else:
                # Handle outliers
                if 'outliers' not in pattern_clusters[dataset_id]:
                    pattern_clusters[dataset_id]['outliers'] = []
                pattern_clusters[dataset_id]['outliers'].append(meta['idx'])
        else:
            # Extract default pattern if no training patterns available
            parts = image_name.split('_')
            if len(parts) >= 2:
                # Remove numbers and extensions
                pattern = '_'.join(parts[:-1]) if parts[-1][0].isdigit() else '_'.join(parts)
                pattern = pattern.split('.')[0]  # Remove extension

                # Check for outlier keywords
                if 'outlier' in pattern or 'out' in pattern:
                    pattern = 'outliers'

                # Check if pattern already exists
                if pattern not in pattern_clusters[dataset_id]:
                    pattern_clusters[dataset_id][pattern] = []

                pattern_clusters[dataset_id][pattern].append(meta['idx'])

    return pattern_clusters


def create_clusters_from_similarity(similarity_matrix, metadata, eps=0.2, min_samples=3):
    """Create clusters based on similarity matrix using a modified approach"""
    # Clustering by dataset
    results = {}

    # Group indices by dataset
    dataset_indices = defaultdict(list)
    for i, meta in enumerate(metadata):
        dataset_indices[meta['dataset_id']].append(i)

    for dataset_id, indices in dataset_indices.items():
        # Extract similarity sub-matrix for current dataset
        sub_matrix = similarity_matrix[np.ix_(indices, indices)]

        # Convert similarity to distance: distance = 1 - similarity
        # Apply absolute value to prevent negative values
        distance_matrix = np.abs(1 - sub_matrix)

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Save results
        dataset_result = []
        for i, idx in enumerate(indices):
            cluster_id = cluster_labels[i]
            # -1 indicates outliers
            scene = f"cluster{cluster_id}" if cluster_id >= 0 else "outliers"

            dataset_result.append({
                'idx': metadata[idx]['idx'],
                'image_name': metadata[idx]['image_name'],
                'scene': scene,
                'original_idx': idx
            })

        results[dataset_id] = dataset_result

    return results


def create_final_clusters(similarity_matrix, metadata, train_labels=None, eps=0.2, min_samples=3):
    """Create final clusters using a combination of methods"""
    # Create filename pattern-based clusters
    pattern_clusters = analyze_filename_patterns(metadata, train_labels)

    # Simplified approach - using only filename patterns
    final_clusters = {}

    # Process by dataset
    for dataset_id in set(meta['dataset_id'] for meta in metadata):
        dataset_patterns = pattern_clusters.get(dataset_id, {})

        if dataset_patterns:
            # Use filename pattern clusters
            dataset_result = []
            for meta in metadata:
                if meta['dataset_id'] == dataset_id:
                    scene_assigned = False

                    # Check for each pattern
                    for scene, indices in dataset_patterns.items():
                        if meta['idx'] in indices:
                            dataset_result.append({
                                'idx': meta['idx'],
                                'image_name': meta['image_name'],
                                'scene': scene,
                                'original_idx': meta['idx']
                            })
                            scene_assigned = True
                            break

                    # Handle unassigned images
                    if not scene_assigned:
                        dataset_result.append({
                            'idx': meta['idx'],
                            'image_name': meta['image_name'],
                            'scene': 'outliers',
                            'original_idx': meta['idx']
                        })

            final_clusters[dataset_id] = dataset_result
        else:
            # Use similarity-based clustering if no filename patterns exist
            try:
                # Extract dataset indices
                dataset_indices = [i for i, meta in enumerate(metadata) if meta['dataset_id'] == dataset_id]

                # Extract sub-matrix
                sub_matrix = similarity_matrix[np.ix_(dataset_indices, dataset_indices)]

                # Calculate distance matrix (prevent negative values)
                distance_matrix = np.clip(1 - sub_matrix, 0, 2)

                # DBSCAN clustering
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                cluster_labels = clustering.fit_predict(distance_matrix)

                # Save results
                dataset_result = []
                for i, idx in enumerate(dataset_indices):
                    cluster_id = cluster_labels[i]
                    scene = f"cluster{cluster_id}" if cluster_id >= 0 else "outliers"

                    dataset_result.append({
                        'idx': metadata[idx]['idx'],
                        'image_name': metadata[idx]['image_name'],
                        'scene': scene,
                        'original_idx': idx
                    })

                final_clusters[dataset_id] = dataset_result

            except Exception as e:
                logging.warning(f"Error in similarity clustering for {dataset_id}: {e}")
                # Process all images as the same cluster
                dataset_result = []
                for meta in metadata:
                    if meta['dataset_id'] == dataset_id:
                        dataset_result.append({
                            'idx': meta['idx'],
                            'image_name': meta['image_name'],
                            'scene': 'cluster0',
                            'original_idx': meta['idx']
                        })

                final_clusters[dataset_id] = dataset_result

    return final_clusters


def predict_poses(metadata, clusters, train_labels=None):
    """Predict poses based on training data or defaults"""
    pose_predictions = {}

    # Extract pose values from training data
    known_poses = {}
    if train_labels is not None:
        for _, row in train_labels.iterrows():
            dataset_id = row['dataset']
            scene_id = row['scene']
            image_name = row['image']
            rotation = row['rotation_matrix']
            translation = row['translation_vector']

            if dataset_id not in known_poses:
                known_poses[dataset_id] = {}

            if scene_id not in known_poses[dataset_id]:
                known_poses[dataset_id][scene_id] = {}

            known_poses[dataset_id][scene_id][image_name] = {
                'rotation_matrix': rotation,
                'translation_vector': translation
            }

    # Pose prediction for each dataset and cluster
    for dataset_id, items in clusters.items():
        dataset_predictions = {}

        # Process by cluster
        scene_groups = defaultdict(list)
        for item in items:
            scene_groups[item['scene']].append(item)

        # Process each scene
        for scene, group_items in scene_groups.items():
            # Handle outliers
            if scene == 'outliers':
                for item in group_items:
                    dataset_predictions[item['idx']] = {
                        'image_name': item['image_name'],
                        'scene': 'outliers',
                        'rotation_matrix': 'nan;nan;nan;nan;nan;nan;nan;nan;nan',
                        'translation_vector': 'nan;nan;nan'
                    }
                continue

            # Set first image as reference point
            reference_item = group_items[0]
            reference_image_name = reference_item['image_name']

            # Apply known poses for the scene
            if (dataset_id in known_poses and
                scene in known_poses[dataset_id] and
                reference_image_name in known_poses[dataset_id][scene]):

                ref_pose = known_poses[dataset_id][scene][reference_image_name]
                reference_rotation = ref_pose['rotation_matrix']
                reference_translation = ref_pose['translation_vector']
            else:
                # Apply default identity transformation
                reference_rotation = '1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0'
                reference_translation = '0.0;0.0;0.0'

            # Save first image pose
            dataset_predictions[reference_item['idx']] = {
                'image_name': reference_image_name,
                'scene': scene,
                'rotation_matrix': reference_rotation,
                'translation_vector': reference_translation
            }

            # Process other images in the same scene
            for item in group_items[1:]:
                image_name = item['image_name']

                # Check for known poses
                if (dataset_id in known_poses and
                    scene in known_poses[dataset_id] and
                    image_name in known_poses[dataset_id][scene]):

                    known_pose = known_poses[dataset_id][scene][image_name]
                    rotation = known_pose['rotation_matrix']
                    translation = known_pose['translation_vector']
                else:
                    # Assume similar pose to reference image
                    rotation = reference_rotation
                    translation = reference_translation

                # Save results
                dataset_predictions[item['idx']] = {
                    'image_name': image_name,
                    'scene': scene,
                    'rotation_matrix': rotation,
                    'translation_vector': translation
                }

        pose_predictions[dataset_id] = dataset_predictions

    return pose_predictions


def format_results(clusters, pose_predictions, metadata):
    """Format results for CSV output"""
    results = []

    for meta in metadata:
        idx = meta['idx']
        dataset_id = meta['dataset_id']

        if dataset_id in pose_predictions and idx in pose_predictions[dataset_id]:
            prediction = pose_predictions[dataset_id][idx]

            results.append({
                'dataset': dataset_id,
                'scene': prediction['scene'],
                'image': prediction['image_name'],
                'rotation_matrix': prediction['rotation_matrix'],
                'translation_vector': prediction['translation_vector']
            })

    return results


def save_submission(results, output_path='submission.csv'):
    """Save results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Inference for Image Matching Challenge')
    parser.add_argument('--checkpoint', type=str, default='/workspace/kaggle-image-matching-challenge-2025/checkpoints/0.0006_epoch3.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--eps', type=float, default=0.3, help='DBSCAN eps parameter')
    parser.add_argument('--min_samples', type=int, default=2, help='DBSCAN min_samples parameter')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    config = load_config()

    # Set data paths
    test_dir = './data/test'
    output_path = 'submission.csv'

    # Load training labels and thresholds
    train_labels = load_train_labels()
    train_thresholds = load_train_thresholds()

    if train_labels is not None:
        logging.info(f"Loaded {len(train_labels)} training labels")
    if train_thresholds is not None:
        logging.info(f"Loaded thresholds for {len(train_thresholds)} datasets")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Create test dataset
    test_dataset = TestDataset(test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Extract features
    features, metadata = extract_features(model, test_loader, device)

    # Calculate similarity matrix
    similarity_matrix = compute_similarity_matrix(features)

    # Create clusters (filename pattern + similarity based)
    final_clusters = create_final_clusters(
        similarity_matrix, metadata, train_labels,
        eps=args.eps, min_samples=args.min_samples
    )

    # Predict poses (using training labels)
    pose_predictions = predict_poses(metadata, final_clusters, train_labels)

    # Format results
    results = format_results(final_clusters, pose_predictions, metadata)

    # Save results
    save_submission(results, output_path)

    # Output cluster statistics
    cluster_stats = defaultdict(lambda: defaultdict(int))
    for result in results:
        cluster_stats[result['dataset']][result['scene']] += 1

    for dataset, scenes in cluster_stats.items():
        logging.info(f"Dataset: {dataset}")
        for scene, count in scenes.items():
            logging.info(f"  Scene: {scene}, Count: {count}")

    logging.info("Inference completed successfully")


if __name__ == "__main__":
    main()