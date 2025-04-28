import numpy as np
import cv2

def estimate_pose_pnp(keypoints1, keypoints2, matches, K1, K2):
    """
    Estimate pose using PnP given keypoint matches.

    Args:
        keypoints1: Keypoints in the first image (N, 2)
        keypoints2: Keypoints in the second image (M, 2)
        matches: Indices of matching keypoints (K, 2)
        K1: Camera intrinsic matrix for first image
        K2: Camera intrinsic matrix for second image

    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        inliers: Number of inliers
    """
    # Get matched keypoints
    src_pts = keypoints1[matches[:, 0]].reshape(-1, 1, 2)
    dst_pts = keypoints2[matches[:, 1]].reshape(-1, 1, 2)

    # Find essential matrix
    E, mask = cv2.findEssentialMat(
        src_pts, dst_pts, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    if E is None:
        return None, None, 0

    # Recover pose from essential matrix
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K1, mask=mask)

    # Count inliers
    inliers = np.sum(mask > 0)

    return R, t, inliers

def triangulate_points(keypoints1, keypoints2, matches, K1, K2, R, t):
    """
    Triangulate 3D points from matches.

    Args:
        keypoints1: Keypoints in the first image (N, 2)
        keypoints2: Keypoints in the second image (M, 2)
        matches: Indices of matching keypoints (K, 2)
        K1: Camera intrinsic matrix for first image
        K2: Camera intrinsic matrix for second image
        R: Rotation matrix between cameras (3, 3)
        t: Translation vector between cameras (3,)

    Returns:
        points_3d: Triangulated 3D points (K, 3)
    """
    # Get matched keypoints
    pts1 = keypoints1[matches[:, 0]]
    pts2 = keypoints2[matches[:, 1]]

    # Create projection matrices
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t.reshape(3, 1)])

    # Convert points to homogeneous coordinates
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None)

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)

    # Convert to 3D
    points_3d = points_4d[:3, :] / points_4d[3, :]

    return points_3d.T


def estimate_pose_from_3d_2d(points_3d, points_2d, K):
    """
    Estimate camera pose from 3D-2D correspondences using PnP.

    Args:
        points_3d: 3D points in world coordinates (N, 3)
        points_2d: 2D points in image coordinates (N, 2)
        K: Camera intrinsic matrix

    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        inliers: Number of inliers
    """
    # Convert to expected format
    points_3d = points_3d.astype(np.float32)
    points_2d = points_2d.astype(np.float32)

    # Estimate pose with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, K, None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=8.0,
        iterationsCount=100
    )

    if not success:
        return None, None, 0

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec.reshape(3), len(inliers) if inliers is not None else 0


def refine_pose(R, t, points_3d, points_2d, K):
    """
    Refine camera pose using bundle adjustment.

    Args:
        R: Initial rotation matrix (3, 3)
        t: Initial translation vector (3,)
        points_3d: 3D points in world coordinates (N, 3)
        points_2d: 2D points in image coordinates (N, 2)
        K: Camera intrinsic matrix

    Returns:
        R_refined: Refined rotation matrix (3, 3)
        t_refined: Refined translation vector (3,)
    """
    # Convert rotation matrix to vector
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Refine pose
    _, rvec_refined, tvec_refined = cv2.solvePnP(
        points_3d, points_2d, K, None,
        rvec=rvec, tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Convert back to matrix
    R_refined, _ = cv2.Rodrigues(rvec_refined)

    return R_refined, tvec_refined.reshape(3)


def get_relative_pose(R1, t1, R2, t2):
    """
    Calculate relative pose between two cameras.

    Args:
        R1: Rotation matrix of first camera
        t1: Translation vector of first camera
        R2: Rotation matrix of second camera
        t2: Translation vector of second camera

    Returns:
        R_rel: Relative rotation matrix
        t_rel: Relative translation vector
    """
    # R_rel is the rotation from camera 1 to camera 2
    R_rel = R2 @ R1.T

    # t_rel is the translation from camera 1 to camera 2 in camera 2's frame
    t_rel = t2 - R_rel @ t1

    return R_rel, t_rel


def calculate_reprojection_error(points_3d, points_2d, R, t, K):
    """
    Calculate reprojection error.

    Args:
        points_3d: 3D points in world coordinates (N, 3)
        points_2d: 2D points in image coordinates (N, 2)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        K: Camera intrinsic matrix

    Returns:
        mean_error: Mean reprojection error
        errors: Reprojection error for each point
    """
    # Project 3D points to 2D
    points_3d = points_3d.reshape(-1, 3)
    points_2d = points_2d.reshape(-1, 2)

    # Project points
    rvec, _ = cv2.Rodrigues(R)
    projected_points, _ = cv2.projectPoints(
        points_3d, rvec, t, K, None
    )
    projected_points = projected_points.reshape(-1, 2)

    # Calculate errors
    errors = np.linalg.norm(points_2d - projected_points, axis=1)
    mean_error = np.mean(errors)

    return mean_error, errors


def estimate_scale(t_pred, t_gt, R_pred=None, R_gt=None):
    """
    Estimate scale factor between predicted and ground truth translations.

    Args:
        t_pred: Predicted translation vectors (N, 3) or (3,)
        t_gt: Ground truth translation vectors (N, 3) or (3,)
        R_pred: Predicted rotation matrices (optional)
        R_gt: Ground truth rotation matrices (optional)

    Returns:
        scale: Estimated scale factor
    """
    # Ensure arrays
    t_pred = np.asarray(t_pred)
    t_gt = np.asarray(t_gt)

    # Single translation case
    if t_pred.ndim == 1:
        return np.linalg.norm(t_gt) / np.linalg.norm(t_pred)

    # Multiple translations case
    scales = []
    for i in range(len(t_pred)):
        # If rotations are provided, align translations
        if R_pred is not None and R_gt is not None:
            t_pred_aligned = R_gt[i] @ R_pred[i].T @ t_pred[i]
            scales.append(np.linalg.norm(t_gt[i]) / np.linalg.norm(t_pred_aligned))
        else:
            scales.append(np.linalg.norm(t_gt[i]) / np.linalg.norm(t_pred[i]))

    return np.median(scales)


def align_poses(poses_pred, poses_gt, method='umeyama'):
    """
    Align predicted poses to ground truth poses.

    Args:
        poses_pred: Dict of predicted poses {img_id: (R, t)}
        poses_gt: Dict of ground truth poses {img_id: (R, t)}
        method: Alignment method ('umeyama' or 'scale_only')

    Returns:
        poses_aligned: Dict of aligned poses {img_id: (R, t)}
    """
    # Extract cameras with poses in both sets
    common_imgs = set(poses_pred.keys()) & set(poses_gt.keys())

    if len(common_imgs) < 3:
        print(f"Warning: Not enough common poses for alignment ({len(common_imgs)})")
        return poses_pred

    # Extract camera centers
    centers_pred = []
    centers_gt = []

    for img_id in common_imgs:
        R_pred, t_pred = poses_pred[img_id]
        R_gt, t_gt = poses_gt[img_id]

        # Camera center C = -R^T * t
        center_pred = -R_pred.T @ t_pred
        center_gt = -R_gt.T @ t_gt

        centers_pred.append(center_pred)
        centers_gt.append(center_gt)

    centers_pred = np.array(centers_pred)
    centers_gt = np.array(centers_gt)

    if method == 'umeyama':
        # Use Umeyama algorithm for similarity transformation
        try:
            # Compute centroids
            centroid_pred = centers_pred.mean(axis=0)
            centroid_gt = centers_gt.mean(axis=0)

            # Center the points
            centered_pred = centers_pred - centroid_pred
            centered_gt = centers_gt - centroid_gt

            # Compute optimal rotation
            M = centered_gt.T @ centered_pred
            U, _, Vt = np.linalg.svd(M)
            R_align = U @ Vt

            # Fix for reflection case
            if np.linalg.det(R_align) < 0:
                Vt[-1, :] *= -1
                R_align = U @ Vt

            # Compute scale
            scale = np.sum(centered_gt * (R_align @ centered_pred.T).T) / np.sum(centered_pred**2)

            # Compute translation
            t_align = centroid_gt - scale * (R_align @ centroid_pred)

        except ImportError:
            print("Warning: scipy.spatial.transform not available, falling back to scale_only")
            method = 'scale_only'

    if method == 'scale_only':
        # Only estimate scale
        scale = estimate_scale(centers_pred.reshape(-1), centers_gt.reshape(-1))
        R_align = np.eye(3)
        t_align = np.zeros(3)

    # Apply alignment to all poses
    poses_aligned = {}

    for img_id, (R_pred, t_pred) in poses_pred.items():
        # Apply similarity transformation
        R_aligned = R_pred  # Keep orientation unchanged
        t_aligned = scale * t_pred + R_pred @ t_align

        poses_aligned[img_id] = (R_aligned, t_aligned)

    return poses_aligned


def reconstruct_scene(images, features, matches, K, min_inliers=15):
    """
    Reconstruct a 3D scene from images and feature matches.

    Args:
        images: List of image ids
        features: Dict of features for each image
        matches: Dict of matches between image pairs
        K: Camera intrinsic matrix
        min_inliers: Minimum number of inliers for a valid pose

    Returns:
        poses: Dict of camera poses {img_id: (R, t)}
        points_3d: Dict of 3D points {point_id: coordinates}
        inliers: Dict of inliers per image pair
    """
    # Initialize
    poses = {}
    points_3d = {}
    inliers = {}

    # Find strongest image pair to start
    max_inliers = 0
    best_pair = None

    for (img1, img2), match_data in matches.items():
        if img1 not in images or img2 not in images:
            continue

        # Get keypoints
        kp1 = features[img1]['keypoints']
        kp2 = features[img2]['keypoints']

        # Get matches
        match_indices = np.array(match_data)

        # Estimate relative pose
        R, t, n_inliers = estimate_pose_pnp(kp1, kp2, match_indices, K, K)

        inliers[(img1, img2)] = n_inliers

        if n_inliers > max_inliers and n_inliers >= min_inliers:
            max_inliers = n_inliers
            best_pair = (img1, img2, R, t)

    if best_pair is None:
        print("No valid initial pair found")
        return poses, points_3d, inliers

    # Initialize with best pair
    img1, img2, R_rel, t_rel = best_pair

    # Set first camera as origin
    poses[img1] = (np.eye(3), np.zeros(3))
    poses[img2] = (R_rel, t_rel.flatten())

    # Triangulate initial points
    kp1 = features[img1]['keypoints']
    kp2 = features[img2]['keypoints']
    match_indices = np.array(matches[(img1, img2)])

    initial_points_3d = triangulate_points(
        kp1, kp2, match_indices, K, K, R_rel, t_rel
    )

    # Store 3D points
    for i, (idx1, idx2) in enumerate(match_indices):
        point_id = f"{img1}_{idx1}"
        points_3d[point_id] = initial_points_3d[i]

    # Register remaining cameras
    registered = set([img1, img2])

    while len(registered) < len(images):
        best_inliers = min_inliers - 1
        best_img = None
        best_pose = None

        for img in images:
            if img in registered:
                continue

            # Find 2D-3D correspondences through matches with registered images
            points_3d_list = []
            points_2d_list = []

            for reg_img in registered:
                if (reg_img, img) in matches:
                    match_data = matches[(reg_img, img)]
                elif (img, reg_img) in matches:
                    match_data = matches[(img, reg_img)]
                    # Swap indices for consistency
                    match_data = [(idx2, idx1) for idx1, idx2 in match_data]
                else:
                    continue

                for idx_reg, idx_img in match_data:
                    point_id = f"{reg_img}_{idx_reg}"
                    if point_id in points_3d:
                        points_3d_list.append(points_3d[point_id])
                        points_2d_list.append(features[img]['keypoints'][idx_img])

            if len(points_3d_list) < min_inliers:
                continue

            # Estimate pose using PnP
            points_3d_array = np.array(points_3d_list).astype(np.float32)
            points_2d_array = np.array(points_2d_list).astype(np.float32)

            R, t, n_inliers = estimate_pose_from_3d_2d(
                points_3d_array, points_2d_array, K
            )

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_img = img
                best_pose = (R, t)

        if best_img is None:
            print(f"No more cameras can be registered, {len(registered)}/{len(images)} done")
            break

        # Add best camera
        poses[best_img] = best_pose
        registered.add(best_img)

        # Triangulate more points
        for reg_img in registered:
            if reg_img == best_img:
                continue

            if (reg_img, best_img) in matches:
                img1, img2 = reg_img, best_img
                match_data = matches[(reg_img, best_img)]
            elif (best_img, reg_img) in matches:
                img1, img2 = best_img, reg_img
                match_data = matches[(best_img, reg_img)]
            else:
                continue

            R1, t1 = poses[img1]
            R2, t2 = poses[img2]

            # Get relative pose
            R_rel, t_rel = get_relative_pose(R1, t1, R2, t2)

            # Triangulate
            kp1 = features[img1]['keypoints']
            kp2 = features[img2]['keypoints']
            match_indices = np.array(match_data)

            new_points_3d = triangulate_points(
                kp1, kp2, match_indices, K, K, R_rel, t_rel
            )

            # Store new points
            for i, (idx1, idx2) in enumerate(match_indices):
                point_id = f"{img1}_{idx1}"
                if point_id not in points_3d:  # Avoid overwriting existing points
                    points_3d[point_id] = new_points_3d[i]

    return poses, points_3d, inliers