import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

class OutlierDetector:
    """Outlier detection for image matching"""

    def __init__(self, method='isolation_forest', contamination=0.1, n_neighbors=20):
        """
        Initialize outlier detector.

        Args:
            method: Outlier detection method
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors for LOF method
        """
        self.method = method
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.model = None

    def fit(self, features):
        """
        Fit outlier detection model.

        Args:
            features: Feature vectors for images (n_samples, n_features)
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.model.fit(features)

        elif self.method == 'lof':
            self.model = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                novelty=True
            )
            self.model.fit(features)

        elif self.method == 'ocsvm':
            self.model = OneClassSVM(nu=self.contamination, gamma='auto')
            self.model.fit(features)

        elif self.method == 'similarity':
            # No fitting needed for similarity-based method
            self.features = features

        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

    def predict(self, features, similarity_matrix=None, threshold=0.6):
        """
        Predict outliers.

        Args:
            features: Feature vectors for images
            similarity_matrix: Optional similarity matrix
            threshold: Threshold for similarity-based method

        Returns:
            is_outlier: Boolean array indicating which images are outliers
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if self.method == 'isolation_forest' or self.method == 'ocsvm':
            # -1 for outliers, 1 for inliers
            predictions = self.model.predict(features)
            is_outlier = predictions == -1

        elif self.method == 'lof':
            # -1 for outliers, 1 for inliers
            predictions = self.model.predict(features)
            is_outlier = predictions == -1

        elif self.method == 'similarity':
            if similarity_matrix is None:
                # Compute similarity matrix on-the-fly
                features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
                similarity_matrix = np.dot(features_norm, features_norm.T)

            # Average similarity to other images
            avg_similarity = np.mean(similarity_matrix, axis=1)
            is_outlier = avg_similarity < threshold

        return is_outlier

    def compute_outlier_score(self, features):
        """
        Compute outlier scores (higher means more likely to be an outlier).

        Args:
            features: Feature vectors for images

        Returns:
            scores: Outlier scores for each image
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if self.method == 'isolation_forest':
            # Negated decision function (higher = more anomalous)
            return -self.model.decision_function(features)

        elif self.method == 'lof':
            # Negated decision function (higher = more anomalous)
            return -self.model.decision_function(features)

        elif self.method == 'ocsvm':
            # Negated decision function (higher = more anomalous)
            return -self.model.decision_function(features)

        elif self.method == 'similarity':
            # Compute similarity matrix
            features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
            similarity_matrix = np.dot(features_norm, features_norm.T)

            # Average similarity (lower = more anomalous)
            avg_similarity = np.mean(similarity_matrix, axis=1)
            return -avg_similarity

        return None


def detect_structural_outliers(matches, threshold=10):
    """
    Detect outliers based on number of feature matches.

    Args:
        matches: Dict of feature matches between image pairs
        threshold: Minimum number of matches for non-outlier

    Returns:
        outliers: Set of image ids classified as outliers
    """
    # Count matches per image
    match_counts = {}

    for (img1, img2), match_data in matches.items():
        n_matches = len(match_data)

        if img1 not in match_counts:
            match_counts[img1] = 0
        if img2 not in match_counts:
            match_counts[img2] = 0

        match_counts[img1] += n_matches
        match_counts[img2] += n_matches

    # Detect outliers (images with few matches)
    outliers = set()
    for img_id, count in match_counts.items():
        if count < threshold:
            outliers.add(img_id)

    return outliers


def compute_feature_distribution(features, n_bins=10):
    """
    Compute feature distribution for anomaly detection.

    Args:
        features: Feature vectors for images
        n_bins: Number of bins for histogram

    Returns:
        histograms: Histogram of feature values for each image
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    n_samples, n_features = features.shape
    histograms = np.zeros((n_samples, n_bins))

    # Compute histogram for each image
    for i in range(n_samples):
        hist, _ = np.histogram(features[i], bins=n_bins, range=(-1, 1))
        histograms[i] = hist / np.sum(hist)  # normalize

    return histograms


def detect_outliers_ensemble(features, similarity_matrix=None, n_methods=3, threshold=0.6):
    """
    Ensemble approach for outlier detection.

    Args:
        features: Feature vectors for images
        similarity_matrix: Optional similarity matrix
        n_methods: Number of methods to use in ensemble
        threshold: Voting threshold

    Returns:
        is_outlier: Boolean array indicating which images are outliers
    """
    n_samples = len(features)
    votes = np.zeros(n_samples)

    # Use multiple methods
    methods = ['isolation_forest', 'lof', 'similarity'][:n_methods]

    for method in methods:
        detector = OutlierDetector(method=method)
        detector.fit(features)
        is_outlier = detector.predict(features, similarity_matrix)
        votes += is_outlier

    # Majority vote
    is_outlier_ensemble = votes / len(methods) > threshold

    return is_outlier_ensemble