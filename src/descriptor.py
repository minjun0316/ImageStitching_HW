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
    # patch_size는 keypoint 주변을 어느 정도 크기로 볼지에 관여하는 파라미터다.
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    # SIFT descriptor는 grayscale 기준으로 계산하므로 먼저 단일 채널로 바꾼다.
    # GaussianBlur를 먼저 적용해 미세한 노이즈와 aliasing 영향을 줄인다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), smoothing_sigma)
    # OpenCV SIFT.compute는 보통 uint8 이미지 입력을 기대하므로 다시 uint8로 맞춘다.
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    # cv2.SIFT_create는 SIFT 객체를 만든다.
    # 여기서는 detector로 SIFT를 쓰지 않고, descriptor 계산기 역할만 사용한다.
    sift = cv2.SIFT_create(
        nfeatures=max(len(keypoints), 1),
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=smoothing_sigma,
    )

    # descriptor_size는 OpenCV KeyPoint의 size 필드에 넣을 값이다.
    # 이 값은 keypoint 주변에서 descriptor를 계산할 스케일에 영향을 준다.
    descriptor_size = max(float(patch_size), float(num_cells * num_bins) / 2.0)
    # Harris가 뽑아 둔 [x, y] 좌표를 OpenCV KeyPoint 객체로 감싼다.
    # 즉 "점은 Harris가 찾고, 그 점에서 descriptor만 SIFT로 계산"하는 구조다.
    cv_keypoints = [
        cv2.KeyPoint(float(x), float(y), descriptor_size)
        for x, y in keypoints.astype(np.float32)
    ]

    # sift.compute는 주어진 위치에서 descriptor를 계산한다.
    # 반환 keypoint는 유효한 점만 다시 돌려줄 수 있고, descriptors는 보통 (N, 128) 배열이다.
    valid_cv_keypoints, descriptors = sift.compute(gray_u8, cv_keypoints)
    if descriptors is None or not valid_cv_keypoints:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    # OpenCV KeyPoint 객체 목록을 다시 [x, y] NumPy 배열로 변환한다.
    valid_keypoints = np.asarray([[kp.pt[0], kp.pt[1]] for kp in valid_cv_keypoints], dtype=np.float32)
    descriptors = descriptors.astype(np.float32)
    # L2 정규화로 descriptor 크기를 맞춰 두면 조명 변화나 overall intensity scale 변화에 더 안정적이다.
    # keepdims=True를 쓰면 (N,)이 아니라 (N,1)로 유지되어 각 행마다 나눗셈이 쉽게 broadcast된다.
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-6)
    return valid_keypoints, descriptors
