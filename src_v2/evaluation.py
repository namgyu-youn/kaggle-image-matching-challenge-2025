import numpy as np
import torch
import json

def compute_pose_error(R_pred, t_pred, R_gt, t_gt):
    """
    Compute pose error between predicted and ground truth camera poses.

    Args:
        R_pred: Predicted rotation matrix (3x3)
        t_pred: Predicted translation vector (3,)
        R_gt: Ground truth rotation matrix (3x3)
        t_gt: Ground truth translation vector (3,)

    Returns:
        rotation_error: Rotation error in degrees
        translation_error: Translation error (Euclidean distance)
    """
    # Rotation error (in degrees)
    if isinstance(R_pred, torch.Tensor):
        R_pred = R_pred.cpu().numpy()
    if isinstance(t_pred, torch.Tensor):
        t_pred = t_pred.cpu().numpy()
    if isinstance(R_gt, torch.Tensor):
        R_gt = R_gt.cpu().numpy()
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.cpu().numpy()

    R_rel = R_gt @ R_pred.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    rotation_error = np.degrees(np.arccos((trace - 1) / 2))

    # Translation error (Euclidean distance)
    translation_error = np.linalg.norm(t_gt - t_pred)

    return rotation_error, translation_error


def is_pose_correct(R_pred, t_pred, R_gt, t_gt, rot_threshold=5.0, trans_threshold=0.3):
    """
    Check if a predicted pose is correct based on thresholds.

    Args:
        R_pred: Predicted rotation matrix (3x3)
        t_pred: Predicted translation vector (3,)
        R_gt: Ground truth rotation matrix (3x3)
        t_gt: Ground truth translation vector (3,)
        rot_threshold: Rotation error threshold in degrees
        trans_threshold: Translation error threshold

    Returns:
        is_correct: Boolean indicating if the pose is correct
    """
    rot_error, trans_error = compute_pose_error(R_pred, t_pred, R_gt, t_gt)
    return rot_error < rot_threshold and trans_error < trans_threshold


def compute_mAA(cluster, scene, poses_pred, poses_gt, rot_threshold=5.0, trans_threshold=0.3):
    """
    Compute mean Average Accuracy for a cluster with respect to a scene.

    Args:
        cluster: List of image ids in the cluster
        scene: List of image ids in the ground truth scene
        poses_pred: Dict mapping image ids to predicted poses (R, t)
        poses_gt: Dict mapping image ids to ground truth poses (R, t)
        rot_threshold: Rotation error threshold in degrees
        trans_threshold: Translation error threshold

    Returns:
        mAA: Mean Average Accuracy score
    """
    # Find intersection of cluster and scene
    common_images = set(cluster).intersection(set(scene))

    if not common_images:
        return 0.0

    # Count correctly registered images
    correct_poses = 0
    for img_id in common_images:
        if img_id not in poses_pred or img_id not in poses_gt:
            continue

        R_pred, t_pred = poses_pred[img_id]
        R_gt, t_gt = poses_gt[img_id]

        if is_pose_correct(R_pred, t_pred, R_gt, t_gt, rot_threshold, trans_threshold):
            correct_poses += 1

    # mAA = correctly registered images / total images in the scene
    return correct_poses / len(scene) if scene else 0.0


def compute_clustering_score(cluster, scene):
    """
    Compute clustering score for a cluster with respect to a scene.

    Args:
        cluster: List of image ids in the cluster
        scene: List of image ids in the ground truth scene

    Returns:
        clustering_score: Clustering score (|S∩C|/|C|)
    """
    # Intersection size
    intersection_size = len(set(cluster).intersection(set(scene)))

    # Cluster size
    cluster_size = len(cluster)

    # Clustering score
    return intersection_size / cluster_size if cluster_size > 0 else 0.0


def find_best_matches(scenes, clusters, poses_pred, poses_gt):
    """
    Greedy assignment of scenes to clusters based on mAA scores.

    Args:
        scenes: Dict mapping scene ids to lists of image ids
        clusters: Dict mapping cluster ids to lists of image ids
        poses_pred: Dict mapping image ids to predicted poses (R, t)
        poses_gt: Dict mapping image ids to ground truth poses (R, t)

    Returns:
        scene_to_cluster: Dict mapping scene ids to assigned cluster ids
        cluster_to_scene: Dict mapping cluster ids to assigned scene ids
        scores: Dict with mAA and clustering scores for each assignment
    """
    scene_to_cluster = {}
    cluster_to_scene = {}
    scores = {}

    # Calculate mAA and clustering scores for all scene-cluster pairs
    all_scores = {}
    for scene_id, scene_images in scenes.items():
        for cluster_id, cluster_images in clusters.items():
            # Skip already assigned clusters
            if cluster_id in cluster_to_scene:
                continue

            mAA = compute_mAA(cluster_images, scene_images, poses_pred, poses_gt)
            clust_score = compute_clustering_score(cluster_images, scene_images)
            all_scores[(scene_id, cluster_id)] = (mAA, clust_score)

    # Greedy assignment
    remaining_scenes = list(scenes.keys())

    while remaining_scenes:
        best_score = -1
        best_assignment = None

        for scene_id in remaining_scenes:
            for cluster_id in clusters.keys():
                # Skip already assigned clusters
                if cluster_id in cluster_to_scene:
                    continue

                mAA, clust_score = all_scores[(scene_id, cluster_id)]

                # Primary criterion: mAA
                # Secondary criterion: clustering score
                if mAA > best_score or (mAA == best_score and clust_score > all_scores[best_assignment][1]):
                    best_score = mAA
                    best_assignment = (scene_id, cluster_id)

        if best_assignment and best_score > 0:
            scene_id, cluster_id = best_assignment
            scene_to_cluster[scene_id] = cluster_id
            cluster_to_scene[cluster_id] = scene_id
            scores[scene_id] = all_scores[best_assignment]
            remaining_scenes.remove(scene_id)
        else:
            # No more valid assignments
            break

    return scene_to_cluster, cluster_to_scene, scores


def compute_dataset_scores(scenes, clusters, scene_to_cluster, poses_pred, poses_gt):
    """
    Compute overall mAA and clustering scores for a dataset.

    Args:
        scenes: Dict mapping scene ids to lists of image ids
        clusters: Dict mapping cluster ids to lists of image ids
        scene_to_cluster: Dict mapping scene ids to assigned cluster ids
        poses_pred: Dict mapping image ids to predicted poses (R, t)
        poses_gt: Dict mapping image ids to ground truth poses (R, t)

    Returns:
        mAA_score: Overall mAA score for the dataset
        clustering_score: Overall clustering score for the dataset
        combined_score: Harmonic mean of mAA and clustering scores
    """
    if not scene_to_cluster:
        return 0.0, 0.0, 0.0

    # Initialize counters
    total_scene_images = 0
    total_cluster_images = 0
    total_correct_poses = 0
    total_intersection_size = 0

    # Process each assigned scene-cluster pair
    for scene_id, cluster_id in scene_to_cluster.items():
        scene_images = scenes[scene_id]
        cluster_images = clusters[cluster_id]

        # Count images for clustering score
        intersection = set(scene_images).intersection(set(cluster_images))
        total_intersection_size += len(intersection)
        total_cluster_images += len(cluster_images)

        # Count correct poses for mAA
        correct_poses = 0
        for img_id in intersection:
            if img_id not in poses_pred or img_id not in poses_gt:
                continue

            R_pred, t_pred = poses_pred[img_id]
            R_gt, t_gt = poses_gt[img_id]

            if is_pose_correct(R_pred, t_pred, R_gt, t_gt):
                correct_poses += 1

        total_correct_poses += correct_poses
        total_scene_images += len(scene_images)

    # Compute overall scores
    mAA_score = total_correct_poses / total_scene_images if total_scene_images > 0 else 0.0
    clustering_score = total_intersection_size / total_cluster_images if total_cluster_images > 0 else 0.0

    # Compute combined score (harmonic mean)
    if mAA_score > 0 and clustering_score > 0:
        combined_score = 2 * (mAA_score * clustering_score) / (mAA_score + clustering_score)
    else:
        combined_score = 0.0

    return mAA_score, clustering_score, combined_score


def evaluate_submission(submission_path, gt_path):
    """
    Evaluate a submission against ground truth.

    Args:
        submission_path: Path to submission JSON file
        gt_path: Path to ground truth JSON file

    Returns:
        overall_score: Final score averaged over all datasets
        dataset_scores: Dict with scores for each dataset
    """
    # Load submission and ground truth
    with open(submission_path, 'r') as f:
        submission = json.load(f)

    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    dataset_scores = {}
    all_dataset_scores = []

    # Process each dataset
    for dataset_id in ground_truth.keys():
        if dataset_id not in submission:
            print(f"Warning: Dataset {dataset_id} missing in submission.")
            dataset_scores[dataset_id] = {
                'mAA': 0.0,
                'clustering': 0.0,
                'combined': 0.0
            }
            all_dataset_scores.append(0.0)
            continue

        # Extract ground truth data
        gt_scenes = ground_truth[dataset_id]['scenes']
        gt_poses = ground_truth[dataset_id]['poses']

        # Extract submission data
        sub_clusters = submission[dataset_id]['clusters']
        sub_poses = submission[dataset_id]['poses']

        # Convert pose data to numpy arrays
        poses_pred = {}
        poses_gt = {}

        for img_id, pose_data in sub_poses.items():
            if 'rotation' in pose_data and 'translation' in pose_data:
                R = np.array(pose_data['rotation']).reshape(3, 3)
                t = np.array(pose_data['translation'])
                poses_pred[img_id] = (R, t)

        for img_id, pose_data in gt_poses.items():
            if 'rotation' in pose_data and 'translation' in pose_data:
                R = np.array(pose_data['rotation']).reshape(3, 3)
                t = np.array(pose_data['translation'])
                poses_gt[img_id] = (R, t)

        # Find best matches between scenes and clusters
        scene_to_cluster, _, _ = find_best_matches(
            gt_scenes, sub_clusters, poses_pred, poses_gt
        )

        # Compute dataset scores
        mAA, clustering, combined = compute_dataset_scores(
            gt_scenes, sub_clusters, scene_to_cluster, poses_pred, poses_gt
        )

        dataset_scores[dataset_id] = {
            'mAA': mAA,
            'clustering': clustering,
            'combined': combined
        }

        all_dataset_scores.append(combined)

    # Compute overall score
    overall_score = np.mean(all_dataset_scores) if all_dataset_scores else 0.0

    return overall_score, dataset_scores