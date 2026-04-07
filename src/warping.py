import numpy as np
import cv2


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


def match_overlap_brightness(
    image: np.ndarray,
    image_mask: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray,
    min_overlap_pixels: int = 256,
    gain_limits: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    overlap = image_mask & reference_mask
    if np.count_nonzero(overlap) < min_overlap_pixels:
        return image

    image_overlap = image[overlap].astype(np.float32)
    reference_overlap = reference[overlap].astype(np.float32)

    image_luma = 0.299 * image_overlap[:, 0] + 0.587 * image_overlap[:, 1] + 0.114 * image_overlap[:, 2]
    reference_luma = 0.299 * reference_overlap[:, 0] + 0.587 * reference_overlap[:, 1] + 0.114 * reference_overlap[:, 2]

    image_mean = float(np.mean(image_luma))
    reference_mean = float(np.mean(reference_luma))
    if image_mean < 1e-6:
        return image

    gain = np.clip(reference_mean / image_mean, gain_limits[0], gain_limits[1])
    adjusted = image.astype(np.float32) * gain
    return np.clip(adjusted, 0, 255)


def blend_average(images: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros_like(images[0], dtype=np.float32)
    weight = np.zeros(images[0].shape[:2], dtype=np.float32)

    for image, mask in zip(images, masks):
        accumulator += image * mask[..., None].astype(np.float32)
        weight += mask.astype(np.float32)

    weight = np.maximum(weight, 1.0)
    blended = accumulator / weight[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def blend_feather(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    blur_size: int = 31,
) -> np.ndarray:
    accumulator = np.zeros_like(images[0], dtype=np.float32)
    weight_sum = np.zeros(images[0].shape[:2], dtype=np.float32)

    kernel_size = max(3, blur_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    for image, mask in zip(images, masks):
        mask_u8 = mask.astype(np.uint8)
        if not np.any(mask_u8):
            continue

        distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        smooth_weight = cv2.GaussianBlur(distance, (kernel_size, kernel_size), 0)
        smooth_weight *= mask.astype(np.float32)

        max_weight = float(np.max(smooth_weight))
        if max_weight > 1e-6:
            smooth_weight /= max_weight
        else:
            smooth_weight = mask.astype(np.float32)

        accumulator += image * smooth_weight[..., None]
        weight_sum += smooth_weight

    weight_sum = np.maximum(weight_sum, 1e-6)
    blended = accumulator / weight_sum[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)
