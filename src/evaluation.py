import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def evaluate_similarity(model, dataloader, device='cuda'):
    """
    Evaluate similarity prediction performance.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with validation data
        device: Device to run evaluation on

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating similarity"):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass
            outputs = model(batch)

            # Get predictions
            if isinstance(outputs, dict):
                similarity = outputs['similarity'].squeeze().cpu().numpy()
            else:
                similarity = outputs[0].squeeze().cpu().numpy()

            # Store predictions and labels
            all_scores.extend(similarity.tolist())
            all_preds.extend((similarity > 0.5).astype(int).tolist())
            all_labels.extend(batch['label'].cpu().numpy().tolist())

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    return metrics


def evaluate_pose(model, dataloader, device='cuda', rot_threshold=5.0, trans_threshold=0.3):
    """
    Evaluate pose estimation performance.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with validation data
        device: Device to run evaluation on
        rot_threshold: Rotation error threshold in degrees
        trans_threshold: Translation error threshold

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    rotation_errors = []
    translation_errors = []
    correct_poses = 0
    total_poses = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating pose"):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass
            outputs = model(batch)

            # Get predictions
            if isinstance(outputs, dict):
                R_pred = outputs['rotation']
                t_pred = outputs['translation']
            else:
                R_pred, t_pred, _ = outputs

            # Get ground truth
            R_gt = batch['rotation']
            t_gt = batch['translation']

            # Compute errors
            for i in range(R_pred.size(0)):
                rot_error, trans_error = compute_pose_error(
                    R_pred[i], t_pred[i], R_gt[i], t_gt[i]
                )
                rotation_errors.append(rot_error)
                translation_errors.append(trans_error)

                if rot_error < rot_threshold and trans_error < trans_threshold:
                    correct_poses += 1
                total_poses += 1

    # Compute metrics
    metrics = {
        'mean_rotation_error': np.mean(rotation_errors),
        'median_rotation_error': np.median(rotation_errors),
        'mean_translation_error': np.mean(translation_errors),
        'median_translation_error': np.median(translation_errors),
        'pose_accuracy': correct_poses / total_poses if total_poses > 0 else 0
    }

    return metrics


def visualize_matches(image1, image2, keypoints1, keypoints2, matches, save_path=None):
    """
    Visualize matches between two images.

    Args:
        image1: First image (H, W, 3)
        image2: Second image (H, W, 3)
        keypoints1: Keypoints in first image (N, 2)
        keypoints2: Keypoints in second image (N, 2)
        matches: Matches between keypoints (M, 2)
        save_path: Path to save visualization
    """
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display images
    ax[0].imshow(image1)
    ax[1].imshow(image2)

    # Plot keypoints
    ax[0].scatter(keypoints1[:, 0], keypoints1[:, 1], c='r', s=5)
    ax[1].scatter(keypoints2[:, 0], keypoints2[:, 1], c='r', s=5)

    # Plot matches
    for match in matches:
        idx1, idx2 = match
        kp1 = keypoints1[idx1]
        kp2 = keypoints2[idx2]

        # Draw lines between matches
        x1, y1 = kp1
        x2, y2 = kp2

        # Plot line with random color
        color = np.random.rand(3)
        ax[0].plot(x1, y1, 'o', c=color, markersize=5)
        ax[1].plot(x2, y2, 'o', c=color, markersize=5)

    # Remove axes
    ax[0].axis('off')
    ax[1].axis('off')

    # Set title
    ax[0].set_title(f'Image 1 ({len(keypoints1)} keypoints)')
    ax[1].set_title(f'Image 2 ({len(keypoints2)} keypoints)')

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_evaluation_results(metrics, output_path):
    """
    Save evaluation results to JSON file.

    Args:
        metrics: Dictionary with evaluation metrics
        output_path: Path to save results
    """
    # Convert numpy values to Python types
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics[k] = v.tolist()
        elif isinstance(v, np.generic):
            metrics[k] = v.item()

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def evaluate_model(model, val_loader, device='cuda', output_dir=None):
    """
    Comprehensive model evaluation.

    Args:
        model: Model to evaluate
        val_loader: DataLoader with validation data
        device: Device to run evaluation on
        output_dir: Directory to save results

    Returns:
        all_metrics: Dictionary with all evaluation metrics
    """
    model.eval()

    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # Evaluate similarity
    similarity_metrics = evaluate_similarity(model, val_loader, device)

    # Evaluate pose if available
    try:
        pose_metrics = evaluate_pose(model, val_loader, device)
        all_metrics = {**similarity_metrics, **pose_metrics}
    except Exception as e:
        print(f"Error in pose evaluation: {e}")
        all_metrics = similarity_metrics

    # Save results
    if output_dir:
        save_evaluation_results(all_metrics, output_dir / 'evaluation_results.json')

        # Create summary file
        with open(output_dir / 'evaluation_summary.txt', 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=================\n\n")

            f.write("Similarity Metrics:\n")
            f.write(f"  Accuracy: {similarity_metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {similarity_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {similarity_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {similarity_metrics['f1']:.4f}\n")
            f.write(f"  AUC: {similarity_metrics['auc']:.4f}\n\n")

            if 'pose_accuracy' in all_metrics:
                f.write("Pose Metrics:\n")
                f.write(f"  Pose Accuracy: {all_metrics['pose_accuracy']:.4f}\n")
                f.write(f"  Mean Rotation Error: {all_metrics['mean_rotation_error']:.2f} degrees\n")
                f.write(f"  Median Rotation Error: {all_metrics['median_rotation_error']:.2f} degrees\n")
                f.write(f"  Mean Translation Error: {all_metrics['mean_translation_error']:.4f}\n")
                f.write(f"  Median Translation Error: {all_metrics['median_translation_error']:.4f}\n")

    return all_metrics