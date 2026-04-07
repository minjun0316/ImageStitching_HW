import numpy as np


def compute_homography_least_squares(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    if len(src_pts) < 4 or len(dst_pts) < 4:
        raise ValueError("At least 4 point correspondences are required.")

    a_rows = []
    b_rows = []

    for (x, y), (u, v) in zip(src_pts, dst_pts):
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b_rows.append(u)
        b_rows.append(v)

    a = np.asarray(a_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)

    h, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    h = np.append(h, 1.0)
    h_matrix = h.reshape(3, 3)
    return h_matrix


def apply_homography(points: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    projected = (h_matrix @ homogeneous.T).T
    projected /= projected[:, 2:3]
    return projected[:, :2].astype(np.float32)


def compute_reprojection_errors(src_pts: np.ndarray, dst_pts: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    projected = apply_homography(src_pts, h_matrix)
    return np.linalg.norm(projected - dst_pts, axis=1)


def compute_homography_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    num_iters: int = 2000,
    inlier_threshold: float = 3.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if len(src_pts) < 4 or len(dst_pts) < 4:
        raise ValueError("At least 4 point correspondences are required for RANSAC.")

    rng = np.random.default_rng(random_state)
    best_inlier_mask = np.zeros(len(src_pts), dtype=bool)
    best_h = None

    for _ in range(num_iters):
        sample_idx = rng.choice(len(src_pts), size=4, replace=False)

        try:
            h_candidate = compute_homography_least_squares(src_pts[sample_idx], dst_pts[sample_idx])
            errors = compute_reprojection_errors(src_pts, dst_pts, h_candidate)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            continue

        if not np.all(np.isfinite(errors)):
            continue

        inlier_mask = errors < inlier_threshold

        if inlier_mask.sum() > best_inlier_mask.sum():
            best_inlier_mask = inlier_mask
            best_h = h_candidate

    if best_h is None or best_inlier_mask.sum() < 4:
        raise RuntimeError("RANSAC failed to find a valid homography.")

    refined_h = compute_homography_least_squares(src_pts[best_inlier_mask], dst_pts[best_inlier_mask])
    return refined_h, best_inlier_mask
