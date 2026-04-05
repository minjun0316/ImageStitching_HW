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
