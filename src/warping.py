import numpy as np
import cv2


def bilinear_interpolate(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 워핑 후 좌표는 정수가 아니라 소수인 경우가 많다.
    # bilinear interpolation은 가장 가까운 4개 픽셀을 거리 비율로 섞어서 값을 만든다.
    h, w = image.shape[:2]

    # 이미지 범위를 벗어난 좌표는 샘플링할 수 없으므로 valid 마스크로 따로 관리한다.
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

    # wa, wb, wc, wd는 4개 이웃 픽셀에 곱할 bilinear weight다.
    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)

    # NumPy 고급 인덱싱으로 해당 픽셀들을 한 번에 가져온다.
    ia = image[y0, x0].astype(np.float32)
    ib = image[y0, x1].astype(np.float32)
    ic = image[y1, x0].astype(np.float32)
    id_ = image[y1, x1].astype(np.float32)

    # wa[:, None]처럼 차원을 하나 늘리면 (N,) -> (N,1)이 되어 RGB 3채널과 broadcast된다.
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
    # backward warping은 "캔버스의 각 픽셀이 원본 이미지의 어디에서 왔는가"를 역으로 묻는 방식이다.
    # forward warping보다 빈 픽셀이 생기기 덜해서 구현이 안정적이다.
    canvas_h, canvas_w = canvas_shape
    # np.indices((H, W))는 각 픽셀의 y/x 인덱스 그리드를 만든다.
    ys, xs = np.indices((canvas_h, canvas_w), dtype=np.float32)
    # ravel()은 2D 배열을 1D로 펴고, stack으로 [x, y] 좌표 목록을 만든다.
    target_points = np.stack([xs.ravel(), ys.ravel()], axis=1)

    # forward warping은 빈 구멍이 생기기 쉬워서 inverse homography로 source를 샘플링한다.
    h_inv = np.linalg.inv(h_source_to_canvas)
    homogeneous = np.hstack([target_points, np.ones((len(target_points), 1), dtype=np.float32)])
    source_coords = (h_inv @ homogeneous.T).T
    source_coords /= source_coords[:, 2:3]

    # 변환된 source 좌표의 x, y를 분리해 interpolation에 넘긴다.
    src_x = source_coords[:, 0]
    src_y = source_coords[:, 1]

    sampled, valid = bilinear_interpolate(source, src_x, src_y)

    # 1차원으로 계산한 결과를 다시 (height, width, channel) 형태로 되돌린다.
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
    # 두 warped 이미지가 겹치는 부분만 골라 평균 밝기를 비교한다.
    # 여기서는 복잡한 색 보정 대신 gain 하나만 곱하는 단순한 exposure matching을 사용한다.
    overlap = image_mask & reference_mask
    if np.count_nonzero(overlap) < min_overlap_pixels:
        return image

    # 불리언 마스크로 겹치는 픽셀만 1차원 목록처럼 뽑아낸다.
    image_overlap = image[overlap].astype(np.float32)
    reference_overlap = reference[overlap].astype(np.float32)

    # RGB/BGR 세 채널을 하나의 밝기값으로 바꾸기 위해 luma 근사식을 사용한다.
    image_luma = 0.299 * image_overlap[:, 0] + 0.587 * image_overlap[:, 1] + 0.114 * image_overlap[:, 2]
    reference_luma = 0.299 * reference_overlap[:, 0] + 0.587 * reference_overlap[:, 1] + 0.114 * reference_overlap[:, 2]

    image_mean = float(np.mean(image_luma))
    reference_mean = float(np.mean(reference_luma))
    if image_mean < 1e-6:
        return image

    # np.clip으로 gain 범위를 제한해 과도한 밝기 변화가 생기지 않게 한다.
    gain = np.clip(reference_mean / image_mean, gain_limits[0], gain_limits[1])
    adjusted = image.astype(np.float32) * gain
    return np.clip(adjusted, 0, 255)


def blend_average(images: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    # accumulator에는 픽셀값 총합, weight에는 각 픽셀에 몇 장이 겹쳤는지를 누적한다.
    # 마지막에 accumulator / weight를 하면 평균 블렌딩이 된다.
    accumulator = np.zeros_like(images[0], dtype=np.float32)
    weight = np.zeros(images[0].shape[:2], dtype=np.float32)

    for image, mask in zip(images, masks):
        # mask[..., None]은 (H, W) 마스크를 (H, W, 1)로 바꿔 RGB 채널과 곱할 수 있게 만든다.
        accumulator += image * mask[..., None].astype(np.float32)
        weight += mask.astype(np.float32)

    # 겹치는 이미지가 전혀 없는 영역에서 0으로 나누는 문제를 막는다.
    weight = np.maximum(weight, 1.0)
    blended = accumulator / weight[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def blend_feather(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    blur_size: int = 31,
) -> np.ndarray:
    # feather blending은 이미지 경계에서는 가중치를 낮추고 내부에서는 높이는 방식이다.
    # 그래서 겹침 구간에서 한 장이 갑자기 끊기는 느낌이 줄어든다.
    accumulator = np.zeros_like(images[0], dtype=np.float32)
    weight_sum = np.zeros(images[0].shape[:2], dtype=np.float32)

    kernel_size = max(3, blur_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    for image, mask in zip(images, masks):
        mask_u8 = mask.astype(np.uint8)
        if not np.any(mask_u8):
            continue

        # cv2.distanceTransform은 각 픽셀이 mask 경계에서 얼마나 떨어져 있는지 계산한다.
        # 경계에서 멀수록 큰 값을 가지므로, 자연스럽게 내부 픽셀에 더 큰 weight가 간다.
        distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        # GaussianBlur로 weight 변화를 부드럽게 만들어 seam을 더 완만하게 만든다.
        smooth_weight = cv2.GaussianBlur(distance, (kernel_size, kernel_size), 0)
        smooth_weight *= mask.astype(np.float32)

        max_weight = float(np.max(smooth_weight))
        if max_weight > 1e-6:
            smooth_weight /= max_weight
        else:
            smooth_weight = mask.astype(np.float32)

        accumulator += image * smooth_weight[..., None]
        weight_sum += smooth_weight

    # weight_sum이 매우 작은 영역에서도 분모가 0이 되지 않도록 보호한다.
    weight_sum = np.maximum(weight_sum, 1e-6)
    blended = accumulator / weight_sum[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)
