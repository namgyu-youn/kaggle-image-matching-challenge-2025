import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import torch
from collections import defaultdict
import networkx as nx

def hierarchical_clustering(similarity_matrix, threshold=0.7, n_clusters=None):
    """
    Perform hierarchical clustering based on similarity matrix.

    Args:
        similarity_matrix: Similarity matrix between images
        threshold: Similarity threshold for cutting the dendrogram
        n_clusters: Number of clusters (optional)

    Returns:
        labels: Cluster labels for each image
    """
    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix

    # Ensure symmetry
    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)

    # Perform hierarchical clustering
    if n_clusters is not None:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='average'
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            affinity='precomputed',
            linkage='average'
        )

    labels = clustering.fit_predict(distance_matrix)

    return labels


def dbscan_clustering(similarity_matrix, eps=0.3, min_samples=5):
    """
    Perform DBSCAN clustering based on similarity matrix.

    Args:
        similarity_matrix: Similarity matrix between images
        eps: Maximum distance between samples
        min_samples: Minimum number of samples in a cluster

    Returns:
        labels: Cluster labels for each image
    """
    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix

    # Ensure symmetry
    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)

    # Perform DBSCAN clustering
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed'
    )

    labels = clustering.fit_predict(distance_matrix)

    # Handle outliers (label -1) by assigning them to separate clusters
    if -1 in labels:
        max_label = labels.max()
        outlier_indices = np.where(labels == -1)[0]

        for idx, outlier_idx in enumerate(outlier_indices):
            labels[outlier_idx] = max_label + idx + 1

    return labels


def geometric_verification_clustering(matches, camera_poses, threshold=0.1, min_inliers=15):
    """
    Perform clustering based on geometric verification of matches.

    Args:
        matches: Dict of feature matches between image pairs
        camera_poses: Dict of estimated camera poses between image pairs
        threshold: Reprojection error threshold
        min_inliers: Minimum number of inliers for valid match

    Returns:
        clusters: Dict mapping cluster ids to lists of image ids
    """
    # Create graph
    G = nx.Graph()

    # Add edges based on verified matches
    for (img1, img2), pose_data in camera_poses.items():
        if pose_data['inliers'] >= min_inliers and pose_data['error'] < threshold:
            # Add edge with weight based on inliers and error
            weight = pose_data['inliers'] / (1 + pose_data['error'])
            G.add_edge(img1, img2, weight=weight)

    # Extract connected components as clusters
    clusters = {}
    for i, component in enumerate(nx.connected_components(G)):
        clusters[i] = list(component)

    return clusters


def get_clusters_from_labels(labels, image_ids):
    """
    Convert cluster labels to dictionary mapping cluster ids to lists of image ids.

    Args:
        labels: Array of cluster labels
        image_ids: List of image ids corresponding to the labels

    Returns:
        clusters: Dict mapping cluster ids to lists of image ids
    """
    clusters = defaultdict(list)

    for idx, label in enumerate(labels):
        clusters[int(label)].append(image_ids[idx])

    return dict(clusters)


def detect_outliers(features, similarity_matrix=None, threshold=0.6, method='isolation_forest'):
    """
    Detect outlier images using various methods.

    Args:
        features: Feature vectors for images
        similarity_matrix: Similarity matrix between images (optional)
        threshold: Threshold for outlier detection
        method: Method for outlier detection

    Returns:
        is_outlier: Boolean array indicating which images are outliers
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    n_samples = len(features)
    is_outlier = np.zeros(n_samples, dtype=bool)

    if method == 'isolation_forest':
        # Use Isolation Forest for outlier detection
        clf = IsolationForest(contamination=0.1, random_state=42)
        outlier_scores = clf.fit_predict(features)
        is_outlier = outlier_scores == -1

    elif method == 'lof':
        # Use Local Outlier Factor
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outlier_scores = clf.fit_predict(features)
        is_outlier = outlier_scores == -1

    elif method == 'similarity':
        # Use similarity matrix
        if similarity_matrix is None:
            raise ValueError("Similarity matrix is required for 'similarity' method")

        # Average similarity to other images
        avg_similarity = np.mean(similarity_matrix, axis=1)
        is_outlier = avg_similarity < threshold

    return is_outlier


def cluster_images(features, image_ids, method='hierarchical', **kwargs):
    """
    Cluster images based on their features.

    Args:
        features: Feature vectors for images
        image_ids: List of image ids corresponding to the features
        method: Clustering method ('hierarchical', 'dbscan', etc.)
        **kwargs: Additional parameters for the clustering method

    Returns:
        clusters: Dict mapping cluster ids to lists of image ids
        outliers: List of image ids classified as outliers
    """
    # Compute similarity matrix if not provided
    if 'similarity_matrix' not in kwargs:
        # Normalize features
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
        similarity_matrix = np.dot(features_norm, features_norm.T)
    else:
        similarity_matrix = kwargs['similarity_matrix']

    # Detect outliers
    outlier_method = kwargs.get('outlier_method', 'similarity')
    outlier_threshold = kwargs.get('outlier_threshold', 0.6)

    is_outlier = detect_outliers(
        features,
        similarity_matrix=similarity_matrix,
        threshold=outlier_threshold,
        method=outlier_method
    )

    # Get outlier image ids
    outliers = [image_ids[i] for i, flag in enumerate(is_outlier) if flag]

    # Filter out outliers for clustering
    non_outlier_indices = ~is_outlier
    image_ids_filtered = [image_ids[i] for i, flag in enumerate(non_outlier_indices) if flag]
    similarity_matrix_filtered = similarity_matrix[non_outlier_indices][:, non_outlier_indices]

    # Perform clustering
    if method == 'hierarchical':
        threshold = kwargs.get('threshold', 0.7)
        n_clusters = kwargs.get('n_clusters', None)

        labels = hierarchical_clustering(
            similarity_matrix_filtered,
            threshold=threshold,
            n_clusters=n_clusters
        )

    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.3)
        min_samples = kwargs.get('min_samples', 5)

        labels = dbscan_clustering(
            similarity_matrix_filtered,
            eps=eps,
            min_samples=min_samples
        )

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Convert labels to clusters
    clusters = get_clusters_from_labels(labels, image_ids_filtered)

    return clusters, outliers