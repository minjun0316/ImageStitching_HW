import numpy as np

from src.harris import detect_harris_corners
from src.descriptor import extract_patch_descriptors
from src.matching import knn_match_mutual
from src.homography import apply_homography, compute_homography_ransac
from src.warping import backward_warp, blend_average, blend_feather, match_overlap_brightness


def image_corners(image: np.ndarray) -> np.ndarray:
    # shape[:2]에서 높이/너비를 꺼내 네 꼭짓점을 [x, y]로 반환한다.
    h, w = image.shape[:2]
    return np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def compute_panorama_bounds(images: list[np.ndarray], homographies: list[np.ndarray]) -> tuple[int, int, np.ndarray]:
    # 각 이미지의 warped corner를 모아 최종 파노라마 bounding box를 계산한다.
    all_corners = []
    for image, h_matrix in zip(images, homographies):
        corners = image_corners(image)
        warped = apply_homography(corners, h_matrix)
        all_corners.append(warped)

    # 여러 corner 배열을 하나로 합친 뒤 최소/최대 좌표를 구한다.
    stacked = np.vstack(all_corners)
    min_xy = np.floor(stacked.min(axis=0)).astype(int)
    max_xy = np.ceil(stacked.max(axis=0)).astype(int)

    # 음수 좌표가 생길 수 있으므로 캔버스 안으로 옮기는 translation을 만든다.
    tx = -min_xy[0]
    ty = -min_xy[1]
    translation = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)

    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)
    return height, width, translation


def stitch_three_images(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    harris_params: dict,
    descriptor_patch_size: int,
    ratio_thresh: float,
    descriptor_params: dict | None = None,
    ransac_params: dict | None = None,
) -> dict:
    # img2를 기준 좌표계로 삼아 파노라마를 만들고, None 옵션은 빈 dict로 바꿔 안전하게 넘긴다.
    descriptor_params = descriptor_params or {}
    ransac_params = ransac_params or {}

    # 파이프라인: 특징점 검출 -> descriptor 생성 -> 매칭 -> homography 추정 -> warping -> blending.

    # 1) 각 이미지에서 Harris corner와 response map을 계산한다.
    corners1, response1 = detect_harris_corners(img1, **harris_params)
    corners2, response2 = detect_harris_corners(img2, **harris_params)
    corners3, response3 = detect_harris_corners(img3, **harris_params)

    # 2) Harris 위치에서 descriptor를 만들며, 유효하지 않은 점은 제외될 수 있다.
    kp1, desc1 = extract_patch_descriptors(img1, corners1, patch_size=descriptor_patch_size, **descriptor_params)
    kp2, desc2 = extract_patch_descriptors(img2, corners2, patch_size=descriptor_patch_size, **descriptor_params)
    kp3, desc3 = extract_patch_descriptors(img3, corners3, patch_size=descriptor_patch_size, **descriptor_params)

    # 3) img2 기준으로 img1-img2, img3-img2 사이 mutual match를 구한다.
    # 이 단계는 descriptor가 서로 비슷한 점들을 고르는 과정이다.
    # 다만 descriptor similarity만으로는 반복 패턴이나 비슷한 texture 때문에 틀린 짝이 일부 남을 수 있다.
    raw_src12, raw_dst12, matches12 = knn_match_mutual(kp1, desc1, kp2, desc2, ratio_thresh=ratio_thresh)
    raw_src32, raw_dst32, matches32 = knn_match_mutual(kp3, desc3, kp2, desc2, ratio_thresh=ratio_thresh)

    # Homography는 최소 4개 대응점이 필요하다.
    if len(raw_src12) < 4:
        raise RuntimeError("Not enough matches between img1 and img2.")
    if len(raw_src32) < 4:
        raise RuntimeError("Not enough matches between img3 and img2.")

    # 4) RANSAC으로 outlier를 제거하며 homography와 inlier mask를 추정한다.
    # 즉 KNN matching이 만든 후보 correspondence 중에서 "하나의 homography로 함께 설명되지 않는 점"을 버리는 단계다.
    # RANSAC은 descriptor를 다시 비교하지 않고, reprojection error 기반의 기하학적 일관성으로 inlier/outlier를 나눈다.
    h12, inliers12 = compute_homography_ransac(raw_src12, raw_dst12, **ransac_params)
    h32, inliers32 = compute_homography_ransac(raw_src32, raw_dst32, **ransac_params)

    # 불리언 마스크로 inlier correspondence만 남긴다.
    src12 = raw_src12[inliers12]
    dst12 = raw_dst12[inliers12]
    src32 = raw_src32[inliers32]
    dst32 = raw_dst32[inliers32]

    # img2가 기준 좌표계이므로 자신의 변환은 identity다.
    h1_ref = h12
    h2_ref = np.eye(3, dtype=np.float64)
    h3_ref = h32

    # 세 이미지를 모두 담을 수 있는 panorama canvas 크기와 translation을 계산한다.
    pano_h, pano_w, translation = compute_panorama_bounds([img1, img2, img3], [h1_ref, h2_ref, h3_ref])

    # 최종 변환은 원본 이미지 좌표를 파노라마 캔버스 좌표로 보낸다.
    h1_canvas = translation @ h1_ref
    h2_canvas = translation @ h2_ref
    h3_canvas = translation @ h3_ref

    # 5) 세 이미지를 panorama canvas 위로 backward warping 한다.
    warp1, mask1 = backward_warp(img1, (pano_h, pano_w), h1_canvas)
    warp2, mask2 = backward_warp(img2, (pano_h, pano_w), h2_canvas)
    warp3, mask3 = backward_warp(img3, (pano_h, pano_w), h3_canvas)

    # 가장 단순한 baseline 결과는 겹치는 영역 평균이다.
    panorama_raw = blend_average([warp1, warp2, warp3], [mask1, mask2, mask3])

    # seam을 줄이기 위해 img2 기준으로 img1/img3의 겹침 영역 밝기를 먼저 맞춘다.
    warp1_brightness = match_overlap_brightness(warp1, mask1, warp2, mask2)
    warp3_brightness = match_overlap_brightness(warp3, mask3, warp2, mask2)

    # feather blending은 경계 가중치를 완만하게 바꿔 seam을 더 부드럽게 만든다.
    panorama_brightness = blend_average([warp1_brightness, warp2, warp3_brightness], [mask1, mask2, mask3])
    panorama_feather = blend_feather([warp1_brightness, warp2, warp3_brightness], [mask1, mask2, mask3])

    # 후처리와 시각화에 쓸 중간 결과를 함께 반환한다.
    return {
        # corners*: Harris detector가 찾은 원래 코너 좌표들
        "corners1": corners1,
        "corners2": corners2,
        "corners3": corners3,
        # response*: Harris response 전체 맵
        "response1": response1,
        "response2": response2,
        "response3": response3,
        # raw_matches*: mutual match는 통과했지만 RANSAC 전 단계인 좌표쌍
        "raw_matches12_src": raw_src12,
        "raw_matches12_dst": raw_dst12,
        "raw_matches32_src": raw_src32,
        "raw_matches32_dst": raw_dst32,
        # matches*: RANSAC inlier만 남긴 좌표쌍
        "matches12_src": src12,
        "matches12_dst": dst12,
        "matches32_src": src32,
        "matches32_dst": dst32,
        # matches*_meta: 매칭 인덱스와 ratio score
        "matches12_meta": matches12,
        "matches32_meta": matches32,
        # matches*_inliers: 각 raw match가 inlier인지 나타내는 bool mask
        "matches12_inliers": inliers12,
        "matches32_inliers": inliers32,
        # H12/H32: 각 이미지에서 img2 기준으로 가는 homography
        "H12": h12,
        "H32": h32,
        # warped_img*: blending 전, 캔버스에 정렬된 원본 이미지
        "warped_img1": np.clip(warp1, 0, 255).astype(np.uint8),
        "warped_img3": np.clip(warp3, 0, 255).astype(np.uint8),
        # panorama_*: 단계별 결과 파노라마
        "panorama_raw": panorama_raw,
        "panorama_brightness": panorama_brightness,
        "panorama_feather": panorama_feather,
    }
