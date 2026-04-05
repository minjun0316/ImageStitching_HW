import numpy as np


def bilinear_interpolate(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]

    valid = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
    sampled = np.zeros((len(xs), 3), dtype=np.float32)

    if not np.any(valid):
        return sampled, valid

    xv = xs[valid]
    yv = ys[valid]

    x0 = np.floor(xv).astype(int)
    x1 = x0 + 1
    y0 = np.floor(yv).astype(int)
    y1 = y0 + 1

    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)

    ia = image[y0, x0].astype(np.float32)
    ib = image[y0, x1].astype(np.float32)
    ic = image[y1, x0].astype(np.float32)
    id_ = image[y1, x1].astype(np.float32)

    sampled_valid = (
        wa[:, None] * ia +
        wb[:, None] * ib +
        wc[:, None] * ic +
        wd[:, None] * id_
    )
    sampled[valid] = sampled_valid
    return sampled, valid


def backward_warp(
    source: np.ndarray,
    canvas_shape: tuple[int, int],
    h_source_to_canvas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    canvas_h, canvas_w = canvas_shape
    ys, xs = np.indices((canvas_h, canvas_w), dtype=np.float32)
    target_points = np.stack([xs.ravel(), ys.ravel()], axis=1)

    h_inv = np.linalg.inv(h_source_to_canvas)
    homogeneous = np.hstack([target_points, np.ones((len(target_points), 1), dtype=np.float32)])
    source_coords = (h_inv @ homogeneous.T).T
    source_coords /= source_coords[:, 2:3]

    src_x = source_coords[:, 0]
    src_y = source_coords[:, 1]

    sampled, valid = bilinear_interpolate(source, src_x, src_y)

    warped = sampled.reshape(canvas_h, canvas_w, 3)
    mask = valid.reshape(canvas_h, canvas_w)
    return warped, mask


def blend_average(images: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros_like(images[0], dtype=np.float32)
    weight = np.zeros(images[0].shape[:2], dtype=np.float32)

    for image, mask in zip(images, masks):
        accumulator += image * mask[..., None].astype(np.float32)
        weight += mask.astype(np.float32)

    weight = np.maximum(weight, 1.0)
    blended = accumulator / weight[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)
