import cv2
import numpy as np


def harris_response(gray: np.ndarray, k: float = 0.04, window_size: int = 3) -> np.ndarray:
    gray = gray.astype(np.float32)
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    ixx = cv2.GaussianBlur(ix * ix, (window_size, window_size), 1.0)
    iyy = cv2.GaussianBlur(iy * iy, (window_size, window_size), 1.0)
    ixy = cv2.GaussianBlur(ix * iy, (window_size, window_size), 1.0)

    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    response = det - k * (trace ** 2)
    return response


def non_maximum_suppression(response: np.ndarray, nms_window: int = 7) -> np.ndarray:
    dilated = cv2.dilate(response, np.ones((nms_window, nms_window), dtype=np.uint8))
    return response == dilated


def detect_harris_corners(
    image: np.ndarray,
    max_corners: int = 800,
    response_thresh_ratio: float = 0.01,
    k: float = 0.04,
    window_size: int = 3,
    nms_window: int = 7,
    border_margin: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = harris_response(gray, k=k, window_size=window_size)

    threshold = response_thresh_ratio * response.max()
    mask = (response > threshold) & non_maximum_suppression(response, nms_window=nms_window)

    h, w = gray.shape
    mask[:border_margin, :] = False
    mask[-border_margin:, :] = False
    mask[:, :border_margin] = False
    mask[:, -border_margin:] = False

    ys, xs = np.where(mask)
    scores = response[ys, xs]

    if len(scores) == 0:
        return np.empty((0, 2), dtype=np.float32), response

    order = np.argsort(scores)[::-1][:max_corners]
    corners = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
    return corners, response
