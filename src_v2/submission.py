import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

def format_rotation_matrix(R):
    """Format rotation matrix as semicolon-separated string"""
    if R is None:
        return ';'.join(['nan'] * 9)
    return ';'.join([f"{val:.6f}" for val in R.flatten()])


def format_translation_vector(t):
    """Format translation vector as semicolon-separated string"""
    if t is None:
        return ';'.join(['nan'] * 3)
    return ';'.join([f"{val:.6f}" for val in t.flatten()])


def create_submission_file(datasets, output_path):
    """
    Create submission file in the required format.

    Args:
        datasets: Dict mapping dataset ids to clusters and poses
        output_path: Path to save the submission file
    """
    # Initialize rows for the submission file
    rows = []

    for dataset_id, data in datasets.items():
        clusters = data['clusters']
        outliers = data.get('outliers', [])
        poses = data.get('poses', {})

        # Process each cluster
        for cluster_id, image_ids in clusters.items():
            scene_name = f"scene_{cluster_id}"

            for image_id in image_ids:
                # Get pose if available
                if image_id in poses:
                    R, t = poses[image_id]
                    rotation_matrix = format_rotation_matrix(R)
                    translation_vector = format_translation_vector(t)
                else:
                    rotation_matrix = ';'.join(['nan'] * 9)
                    translation_vector = ';'.join(['nan'] * 3)

                rows.append({
                    'dataset': dataset_id,
                    'scene': scene_name,
                    'image': image_id,
                    'rotation_matrix': rotation_matrix,
                    'translation_vector': translation_vector
                })

        # Process outliers
        for image_id in outliers:
            rows.append({
                'dataset': dataset_id,
                'scene': 'outliers',
                'image': image_id,
                'rotation_matrix': ';'.join(['nan'] * 9),
                'translation_vector': ';'.join(['nan'] * 3)
            })

    # Create DataFrame and save as CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def load_ground_truth(gt_dir):
    """
    Load ground truth data from directory.

    Args:
        gt_dir: Path to ground truth directory

    Returns:
        ground_truth: Dict with ground truth data
    """
    ground_truth = {}

    # Find all scene directories
    dataset_dirs = [d for d in Path(gt_dir).iterdir() if d.is_dir()]

    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name
        dataset_data = {
            'scenes': defaultdict(list),
            'poses': {}
        }

        # Process scene directories
        scene_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name != 'outliers']

        for scene_dir in scene_dirs:
            scene_id = scene_dir.name

            # Load pose files
            pose_files = list(scene_dir.glob('*.txt'))

            for pose_file in pose_files:
                image_id = pose_file.stem

                # Add image to scene
                dataset_data['scenes'][scene_id].append(image_id)

                # Read pose
                with open(pose_file, 'r') as f:
                    lines = f.readlines()

                # Parse rotation and translation
                R = np.array([
                    [float(x) for x in lines[0].strip().split()],
                    [float(x) for x in lines[1].strip().split()],
                    [float(x) for x in lines[2].strip().split()]
                ])

                t = np.array([float(x) for x in lines[3].strip().split()])

                # Store pose
                dataset_data['poses'][image_id] = {
                    'rotation': R.tolist(),
                    'translation': t.tolist()
                }

        # Process outliers
        outlier_dir = dataset_dir / 'outliers'
        if outlier_dir.exists():
            outliers = []

            for img_file in outlier_dir.glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    outliers.append(img_file.stem)

            dataset_data['outliers'] = outliers

        # Convert defaultdict to dict
        dataset_data['scenes'] = dict(dataset_data['scenes'])

        ground_truth[dataset_id] = dataset_data

    return ground_truth


def create_submission_dict(clusters, poses, outliers=None):
    """
    Create submission dictionary from clustering and pose data.

    Args:
        clusters: Dict mapping cluster ids to lists of image ids
        poses: Dict mapping image ids to camera poses (R, t)
        outliers: List of outlier image ids

    Returns:
        submission: Submission dict in the required format
    """
    submission = {
        'clusters': clusters,
        'poses': {}
    }

    # Format poses
    for image_id, (R, t) in poses.items():
        submission['poses'][image_id] = {
            'rotation': R.tolist() if R is not None else None,
            'translation': t.tolist() if t is not None else None
        }

    # Add outliers
    if outliers:
        submission['outliers'] = outliers

    return submission