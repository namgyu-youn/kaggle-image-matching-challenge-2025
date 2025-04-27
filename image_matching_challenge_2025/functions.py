import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.metrics import pairwise_distances
from scipy.sparse import csgraph
from pathlib import Path

def extract_sift_features(image):
    """Extract SIFT features from an image"""
    if torch.is_tensor(image):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_features(desc1, desc2, ratio_thresh=0.7):
    """Match SIFT features using Lowe's ratio test"""
    if desc1 is None or desc2 is None:
        return []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


def estimate_fundamental_matrix(kp1, kp2, matches, method=cv2.FM_RANSAC, confidence=0.999):
    """Estimate fundamental matrix between two images"""
    if len(matches) < 8:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, method,
                                     ransacReprojThreshold=3.0,
                                     confidence=confidence)

    return F, mask


def compute_similarity_matrix(features, metric='cosine'):
    """Compute similarity matrix between features"""
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    if metric == 'cosine':
        similarity = np.dot(features, features.T)
    elif metric == 'euclidean':
        distances = pairwise_distances(features, metric='euclidean')
        similarity = 1.0 / (1.0 + distances)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return similarity


def graph_based_clustering(similarity_matrix, threshold=0.5):
    """Perform graph-based clustering"""
    # Create adjacency matrix
    adjacency = similarity_matrix > threshold

    # Find connected components
    n_components, labels = csgraph.connected_components(
        adjacency, directed=False, return_labels=True
    )

    return labels, n_components


def spectral_clustering(similarity_matrix, n_clusters=None, max_clusters=10):
    """Perform spectral clustering"""
    if n_clusters is None:
        # Estimate number of clusters using eigenvalue gap
        laplacian = csgraph.laplacian(similarity_matrix, normed=True)
        eigenvalues, _ = np.linalg.eigh(laplacian)

        # Find largest gap in eigenvalues
        gaps = np.diff(eigenvalues)
        n_clusters = min(np.argmax(gaps[:max_clusters]) + 1, max_clusters)

    clustering = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )

    labels = clustering.fit_predict(similarity_matrix)
    return labels, n_clusters


def detect_outliers(features, similarity_matrix, threshold=0.1):
    """Detect outlier images"""
    # Compute average similarity to other images
    mean_similarity = np.mean(similarity_matrix, axis=1)

    # Images with low average similarity are outliers
    outlier_mask = mean_similarity < threshold

    return outlier_mask


def create_submission_format(predictions, clusters, outliers):
    """Create submission format from predictions"""
    submission_data = []

    for dataset_id, images in predictions.items():
        for img_path, pred in images.items():
            img_name = Path(img_path).name

            if pred['is_outlier']:
                scene = 'outliers'
                rotation = ';'.join(['nan'] * 9)
                translation = ';'.join(['nan'] * 3)
            else:
                scene = f"cluster{pred['cluster_id']}"

                if pred.get('rotation') is not None:
                    rotation = ';'.join([f"{val:.6f}" for val in pred['rotation'].flatten()])
                else:
                    rotation = ';'.join(['nan'] * 9)

                if pred.get('translation') is not None:
                    translation = ';'.join([f"{val:.6f}" for val in pred['translation']])
                else:
                    translation = ';'.join(['nan'] * 3)

            submission_data.append({
                'dataset': dataset_id,
                'scene': scene,
                'image': img_name,
                'rotation_matrix': rotation,
                'translation_vector': translation
            })

    return submission_data


def visualize_clusters(image_paths, labels, output_path=None, max_images_per_cluster=10):
    """Visualize clustering results"""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig, axes = plt.subplots(n_clusters, max_images_per_cluster,
                            figsize=(2*max_images_per_cluster, 2*n_clusters))

    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    for i, label in enumerate(unique_labels):
        cluster_images = [image_paths[j] for j in range(len(labels)) if labels[j] == label]
        cluster_images = cluster_images[:max_images_per_cluster]

        for j, img_path in enumerate(cluster_images):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 150))

            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Cluster {label}')

        # Hide unused subplots
        for j in range(len(cluster_images), max_images_per_cluster):
            axes[i, j].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def bundle_adjustment(cameras, points3d, observations):
    """Perform bundle adjustment (placeholder for now)"""
    # This would typically use libraries like Ceres or GTSAM
    # For now, we'll just return the input as is
    return cameras, points3d


def triangulate_points(P1, P2, pts1, pts2):
    """Triangulate 3D points from two camera views"""
    points4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points3d = points4d[:3] / points4d[3]
    return points3d.T


def decompose_essential_matrix(E):
    """Decompose essential matrix into rotation and translation"""
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation matrix
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Four possible solutions
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    return R1, R2, t


def compute_pose_from_essential(E, pts1, pts2, K):
    """Compute camera pose from essential matrix"""
    R1, R2, t = decompose_essential_matrix(E)

    # Test all four possible solutions
    solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    best_R, best_t = None, None
    max_positive = 0

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    for R, t in solutions:
        P2 = K @ np.hstack((R, t.reshape(3, 1)))

        # Triangulate points
        points3d = triangulate_points(P1, P2, pts1, pts2)

        # Count points in front of both cameras
        z1 = points3d[:, 2]
        points3d_cam2 = (R @ points3d.T).T + t
        z2 = points3d_cam2[:, 2]

        positive = np.sum((z1 > 0) & (z2 > 0))

        if positive > max_positive:
            max_positive = positive
            best_R, best_t = R, t

    return best_R, best_t


def filter_matches_epipolar(kp1, kp2, matches, F, threshold=3.0):
    """Filter matches using epipolar constraint"""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Convert to homogeneous coordinates
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    # Compute epipolar lines
    lines2 = (F @ pts1_h.T).T
    lines1 = (F.T @ pts2_h.T).T

    # Compute distances to epipolar lines
    dist2 = np.abs(np.sum(pts2_h * lines2, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
    dist1 = np.abs(np.sum(pts1_h * lines1, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)

    # Filter matches
    valid_mask = (dist1 < threshold) & (dist2 < threshold)
    valid_matches = [m for i, m in enumerate(matches) if valid_mask[i]]

    return valid_matches


def compute_mAA(pred_cameras, gt_cameras, threshold=0.1):
    """Compute mean Average Accuracy for camera pose estimation"""
    # Placeholder implementation
    # In a real implementation, this would compare predicted and ground truth camera centers
    return 0.75  # Dummy value


def estimate_camera_intrinsics(image_shape):
    """Estimate camera intrinsics from image shape"""
    height, width = image_shape[:2]

    # Assume standard pinhole camera model
    focal_length = max(width, height) * 1.2
    cx = width / 2
    cy = height / 2

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])

    return K