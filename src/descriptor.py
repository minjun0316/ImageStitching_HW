import cv2
import numpy as np


def extract_patch_descriptors(
    image: np.ndarray,
    keypoints: np.ndarray,
    patch_size: int = 11,
    num_cells: int = 4,
    num_bins: int = 8,
    smoothing_sigma: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), smoothing_sigma)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    sift = cv2.SIFT_create(
        nfeatures=max(len(keypoints), 1),
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=smoothing_sigma,
    )

    descriptor_size = max(float(patch_size), float(num_cells * num_bins) / 2.0)
    cv_keypoints = [
        cv2.KeyPoint(float(x), float(y), descriptor_size)
        for x, y in keypoints.astype(np.float32)
    ]

    valid_cv_keypoints, descriptors = sift.compute(gray_u8, cv_keypoints)
    if descriptors is None or not valid_cv_keypoints:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    valid_keypoints = np.asarray([[kp.pt[0], kp.pt[1]] for kp in valid_cv_keypoints], dtype=np.float32)
    descriptors = descriptors.astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-6)
    return valid_keypoints, descriptors
