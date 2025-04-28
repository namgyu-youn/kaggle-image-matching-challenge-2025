import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import DBSCAN


def ensemble_predictions(predictions_list, weights=None):
    """Ensemble predictions from multiple models

    Args:
        predictions_list: List of prediction dictionaries
        weights: Optional weights for each model

    Returns:
        Ensemble predictions
    """
    if weights is None:
        weights = [1.0] * len(predictions_list)

    weights = np.array(weights) / sum(weights)

    ensemble_results = {}

    # Process by dataset
    all_datasets = set()
    for pred in predictions_list:
        all_datasets.update(pred.keys())

    for dataset_id in all_datasets:
        ensemble_results[dataset_id] = {}

        # Gather all images in this dataset
        all_images = set()
        for pred in predictions_list:
            if dataset_id in pred:
                all_images.update(pred[dataset_id].keys())

        for img_path in all_images:
            # Average similarity scores
            sim_scores = []
            for i, pred in enumerate(predictions_list):
                if dataset_id in pred and img_path in pred[dataset_id]:
                    sim_score = pred[dataset_id][img_path].get('similarity', 0.0)
                    sim_scores.append(sim_score * weights[i])

            avg_sim_score = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0

            # Voting for cluster ID
            cluster_votes = defaultdict(float)
            for i, pred in enumerate(predictions_list):
                if dataset_id in pred and img_path in pred[dataset_id]:
                    cluster_id = pred[dataset_id][img_path].get('cluster_id')
                    if cluster_id is not None:
                        cluster_votes[cluster_id] += weights[i]

            top_cluster = max(cluster_votes.items(), key=lambda x: x[1])[0] if cluster_votes else None

            # Rotation and translation (weighted average of valid values)
            rot_sum = 0
            trans_sum = 0
            rot_weight_sum = 0
            trans_weight_sum = 0

            for i, pred in enumerate(predictions_list):
                if dataset_id in pred and img_path in pred[dataset_id]:
                    rotation = pred[dataset_id][img_path].get('rotation')
                    translation = pred[dataset_id][img_path].get('translation')

                    if rotation is not None:
                        rot_sum += rotation * weights[i]
                        rot_weight_sum += weights[i]

                    if translation is not None:
                        trans_sum += translation * weights[i]
                        trans_weight_sum += weights[i]

            avg_rotation = rot_sum / rot_weight_sum if rot_weight_sum > 0 else None
            avg_translation = trans_sum / trans_weight_sum if trans_weight_sum > 0 else None

            # Determine outlier status (majority vote)
            outlier_votes = 0
            for i, pred in enumerate(predictions_list):
                if dataset_id in pred and img_path in pred[dataset_id]:
                    is_outlier = pred[dataset_id][img_path].get('is_outlier', False)
                    if is_outlier:
                        outlier_votes += weights[i]

            is_outlier = outlier_votes > 0.5

            # Store ensemble result
            ensemble_results[dataset_id][img_path] = {
                'similarity': avg_sim_score,
                'cluster_id': top_cluster,
                'rotation': avg_rotation,
                'translation': avg_translation,
                'is_outlier': is_outlier
            }

    return ensemble_results


def weighted_clustering(similarity_matrix, weights=None, eps=0.3, min_samples=3):
    """Weighted clustering based on similarity matrix

    Args:
        similarity_matrix: Similarity matrix (n_models, n_images, n_images)
        weights: Optional weights for each model

    Returns:
        Cluster assignments
    """
    if weights is None:
        weights = np.ones(similarity_matrix.shape[0]) / similarity_matrix.shape[0]

    # Compute weighted similarity matrix
    weighted_sim = np.sum(similarity_matrix * weights[:, np.newaxis, np.newaxis], axis=0)

    # Convert to distance matrix
    distance_matrix = 1.0 - weighted_sim

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    # Map -1 (noise) to separate singleton clusters
    unique_labels = set(labels)
    if -1 in unique_labels:
        noise_indices = np.where(labels == -1)[0]
        next_label = max(unique_labels) + 1

        for idx in noise_indices:
            labels[idx] = next_label
            next_label += 1

    return labels


def ensemble_submissions(submission_files, output_path, weights=None):
    """Ensemble multiple submission files

    Args:
        submission_files: List of submission file paths
        output_path: Path to output ensemble submission
        weights: Optional weights for each submission
    """
    if weights is None:
        weights = [1.0] * len(submission_files)

    weights = np.array(weights) / sum(weights)

    # Load all submissions
    submissions = []
    for file_path in submission_files:
        df = pd.read_csv(file_path)
        submissions.append(df)

    # Group by dataset and image
    grouped_data = defaultdict(lambda: defaultdict(list))

    for i, df in enumerate(submissions):
        for _, row in df.iterrows():
            dataset = row['dataset']
            image = row['image']
            scene = row['scene']
            rotation_matrix = row['rotation_matrix']
            translation_vector = row['translation_vector']

            grouped_data[dataset][image].append({
                'scene': scene,
                'rotation_matrix': rotation_matrix,
                'translation_vector': translation_vector,
                'weight': weights[i]
            })

    # Create ensemble submission
    ensemble_rows = []

    for dataset, images in grouped_data.items():
        for image, predictions in images.items():
            # Determine scene by weighted voting
            scene_votes = defaultdict(float)
            for pred in predictions:
                scene_votes[pred['scene']] += pred['weight']

            ensemble_scene = max(scene_votes.items(), key=lambda x: x[1])[0]

            # If outlier, use NaN for rotation and translation
            if ensemble_scene == 'outliers':
                ensemble_rotation = ';'.join(['nan'] * 9)
                ensemble_translation = ';'.join(['nan'] * 3)
            else:
                # Compute weighted average for valid rotation matrices and translation vectors
                valid_rotations = []
                valid_weights_rot = []
                valid_translations = []
                valid_weights_trans = []

                for pred in predictions:
                    if pred['scene'] != 'outliers':
                        # Check rotation matrix
                        rot_vals = pred['rotation_matrix'].split(';')
                        if not any(val == 'nan' for val in rot_vals):
                            rot_matrix = np.array([float(val) for val in rot_vals]).reshape(3, 3)
                            valid_rotations.append(rot_matrix)
                            valid_weights_rot.append(pred['weight'])

                        # Check translation vector
                        trans_vals = pred['translation_vector'].split(';')
                        if not any(val == 'nan' for val in trans_vals):
                            trans_vector = np.array([float(val) for val in trans_vals])
                            valid_translations.append(trans_vector)
                            valid_weights_trans.append(pred['weight'])

                # Compute weighted average for rotation matrix
                if valid_rotations:
                    weighted_sum = sum(w * r for w, r in zip(valid_weights_rot, valid_rotations))
                    avg_rotation = weighted_sum / sum(valid_weights_rot)

                    # Ensure proper rotation matrix
                    u, _, vh = np.linalg.svd(avg_rotation, full_matrices=False)
                    avg_rotation = u @ vh

                    ensemble_rotation = ';'.join([f"{val:.6f}" for val in avg_rotation.flatten()])
                else:
                    ensemble_rotation = ';'.join(['nan'] * 9)

                # Compute weighted average for translation vector
                if valid_translations:
                    weighted_sum = sum(w * t for w, t in zip(valid_weights_trans, valid_translations))
                    avg_translation = weighted_sum / sum(valid_weights_trans)

                    ensemble_translation = ';'.join([f"{val:.6f}" for val in avg_translation])
                else:
                    ensemble_translation = ';'.join(['nan'] * 3)

            # Add to ensemble rows
            ensemble_rows.append({
                'dataset': dataset,
                'scene': ensemble_scene,
                'image': image,
                'rotation_matrix': ensemble_rotation,
                'translation_vector': ensemble_translation
            })

    # Create and save ensemble submission
    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_df.to_csv(output_path, index=False)

    return ensemble_df


def merge_clusters(clusters, similarity_matrix, threshold=0.3):
    """Merge similar clusters based on inter-cluster similarity

    Args:
        clusters: Dict of cluster assignments
        similarity_matrix: Similarity matrix between images
        threshold: Similarity threshold for merging

    Returns:
        Updated cluster assignments
    """
    # Group images by cluster
    cluster_indices = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        cluster_indices[cluster_id].append(idx)

    # Compute inter-cluster similarities
    cluster_ids = sorted(cluster_indices.keys())
    n_clusters = len(cluster_ids)
    cluster_sim = np.zeros((n_clusters, n_clusters))

    for i, c1 in enumerate(cluster_ids):
        for j, c2 in enumerate(cluster_ids):
            if i != j:
                # Compute average similarity between images in different clusters
                sim_sum = 0.0
                count = 0

                for idx1 in cluster_indices[c1]:
                    for idx2 in cluster_indices[c2]:
                        sim_sum += similarity_matrix[idx1, idx2]
                        count += 1

                if count > 0:
                    cluster_sim[i, j] = sim_sum / count

    # Merge clusters
    merged_clusters = np.array(clusters, copy=True)
    merge_map = {c: c for c in cluster_ids}  # Map original cluster ID to new ID

    for i, c1 in enumerate(cluster_ids):
        for j, c2 in enumerate(cluster_ids):
            if i < j and cluster_sim[i, j] > threshold:
                # Merge c2 into c1
                c1_new = merge_map[c1]
                c2_new = merge_map[c2]

                if c1_new != c2_new:
                    # Update merge map
                    for c, mapped_c in merge_map.items():
                        if mapped_c == c2_new:
                            merge_map[c] = c1_new

    # Apply merges
    for idx, cluster_id in enumerate(clusters):
        merged_clusters[idx] = merge_map[cluster_id]

    return merged_clusters


def consensus_clustering(clustering_results, similarity_matrix=None):
    """Generate consensus clustering from multiple clustering results

    Args:
        clustering_results: List of clustering results
        similarity_matrix: Optional similarity matrix for merging

    Returns:
        Consensus clustering
    """
    n_images = len(clustering_results[0])
    n_clusterings = len(clustering_results)

    # Create co-association matrix
    co_assoc = np.zeros((n_images, n_images))

    for clustering in clustering_results:
        for i in range(n_images):
            for j in range(i + 1, n_images):
                if clustering[i] == clustering[j]:
                    co_assoc[i, j] += 1
                    co_assoc[j, i] += 1

    # Normalize
    co_assoc /= n_clusterings

    # Final clustering using DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
    consensus = clustering.fit_predict(1.0 - co_assoc)

    # Handle outliers
    if -1 in consensus:
        outlier_indices = np.where(consensus == -1)[0]
        max_label = consensus.max()

        # Assign each outlier to its own cluster
        for i, idx in enumerate(outlier_indices):
            consensus[idx] = max_label + i + 1

    # Merge similar clusters if similarity matrix is provided
    if similarity_matrix is not None:
        consensus = merge_clusters(consensus, similarity_matrix)

    return consensus