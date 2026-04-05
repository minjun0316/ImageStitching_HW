from pathlib import Path

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image


def save_image(path: str, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    canvas = image.copy()
    for x, y in keypoints.astype(int):
        cv2.circle(canvas, (x, y), 3, color, 1)
    return canvas


def draw_matches(
    img1: np.ndarray,
    kp1: np.ndarray,
    img2: np.ndarray,
    kp2: np.ndarray,
    max_draw: int = 60,
) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    count = min(len(kp1), max_draw)
    for idx in range(count):
        x1, y1 = kp1[idx].astype(int)
        x2, y2 = kp2[idx].astype(int)
        color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
        cv2.circle(canvas, (x1, y1), 4, color, 1)
        cv2.circle(canvas, (x2 + w1, y2), 4, color, 1)
        cv2.line(canvas, (x1, y1), (x2 + w1, y2), color, 1)
    return canvas
