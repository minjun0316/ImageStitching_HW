import numpy as np


def pairwise_distances(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    desc1_sq = np.sum(desc1 ** 2, axis=1, keepdims=True)
    desc2_sq = np.sum(desc2 ** 2, axis=1)[None, :]
    dists = desc1_sq + desc2_sq - 2.0 * desc1 @ desc2.T
    return np.maximum(dists, 0.0)


def knn_match(
    kp1: np.ndarray,
    desc1: np.ndarray,
    kp2: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    if len(desc1) < 2 or len(desc2) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

    dists = pairwise_distances(desc1, desc2)
    matches = []

    for i in range(dists.shape[0]):
        nn_idx = np.argsort(dists[i])[:2]
        best, second = dists[i, nn_idx[0]], dists[i, nn_idx[1]]
        ratio = best / (second + 1e-12)

        if ratio < ratio_thresh:
            matches.append((i, int(nn_idx[0]), float(ratio)))

    if not matches:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

    src = np.asarray([kp1[i] for i, _, _ in matches], dtype=np.float32)
    dst = np.asarray([kp2[j] for _, j, _ in matches], dtype=np.float32)
    return src, dst, matches
