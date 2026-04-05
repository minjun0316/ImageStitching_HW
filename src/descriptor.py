import cv2
import numpy as np


def extract_patch_descriptors(
    image: np.ndarray,
    keypoints: np.ndarray,
    patch_size: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    half = patch_size // 2
    h, w = gray.shape
    valid_keypoints = []
    descriptors = []

    for x, y in keypoints.astype(int):
        if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
            continue

        patch = gray[y - half:y + half + 1, x - half:x + half + 1].copy()
        patch = patch.reshape(-1)

        mean = patch.mean()
        std = patch.std()
        if std < 1e-6:
            continue

        patch = (patch - mean) / std
        valid_keypoints.append([x, y])
        descriptors.append(patch)

    if not descriptors:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, patch_size * patch_size), dtype=np.float32)

    return np.asarray(valid_keypoints, dtype=np.float32), np.asarray(descriptors, dtype=np.float32)
