import numpy as np

from src.harris import detect_harris_corners
from src.descriptor import extract_patch_descriptors
from src.matching import knn_match_mutual
from src.homography import apply_homography, compute_homography_ransac
from src.warping import backward_warp, blend_average, blend_feather, match_overlap_brightness


def image_corners(image: np.ndarray) -> np.ndarray:
    # NumPy 이미지 배열의 shape은 보통 (height, width, channels) 순서다.
    # 따라서 shape[:2]는 세로, 가로 크기만 꺼내는 표현이다.
    # 네 꼭짓점을 [x, y] 형식으로 반환해 이후 homography 변환에 사용한다.
    h, w = image.shape[:2]
    return np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def compute_panorama_bounds(images: list[np.ndarray], homographies: list[np.ndarray]) -> tuple[int, int, np.ndarray]:
    # 각 이미지의 네 꼭짓점을 homography로 옮겨 보면,
    # 최종 파노라마에서 이미지가 어디에 놓일지 대략적인 외곽을 알 수 있다.
    # 여기서는 세 이미지의 warped corner를 전부 모아서 전체 bounding box를 만든다.
    all_corners = []
    for image, h_matrix in zip(images, homographies):
        corners = image_corners(image)
        warped = apply_homography(corners, h_matrix)
        all_corners.append(warped)

    # np.vstack는 여러 좌표 배열을 세로 방향으로 이어 붙여 하나의 큰 배열로 만든다.
    stacked = np.vstack(all_corners)
    # axis=0 기준 min/max는 "모든 점들 중 x 최소, y 최소 / x 최대, y 최대"를 구하는 뜻이다.
    # floor/ceil을 써서 소수 좌표를 포함하더라도 캔버스가 충분히 크게 잡히게 한다.
    min_xy = np.floor(stacked.min(axis=0)).astype(int)
    max_xy = np.ceil(stacked.max(axis=0)).astype(int)

    # warped 결과는 음수 좌표를 가질 수 있는데, 이미지 배열 인덱스는 음수를 쓸 수 없다.
    # 그래서 최소 좌표가 0 이상이 되도록 translation 행렬을 추가한다.
    tx = -min_xy[0]
    ty = -min_xy[1]
    # 3x3 행렬의 마지막 열 [tx, ty]는 2D 평행이동을 나타낸다.
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
    # None이 들어오면 빈 dict로 바꿔 이후 **descriptor_params, **ransac_params를 안전하게 사용한다.
    descriptor_params = descriptor_params or {}
    ransac_params = ransac_params or {}

    # 1) 각 이미지에서 Harris corner를 찾는다.
    # 반환값은 (corner 좌표 배열, response 맵)이다.
    # response 맵은 Harris score가 픽셀마다 저장된 2D 배열이다.
    corners1, response1 = detect_harris_corners(img1, **harris_params)
    corners2, response2 = detect_harris_corners(img2, **harris_params)
    corners3, response3 = detect_harris_corners(img3, **harris_params)

    # 2) Harris keypoint 위치에서 SIFT descriptor를 계산한다.
    # detector와 descriptor를 분리해서 쓰는 구조라, 점은 Harris가 찾고
    # 각 점 주변을 설명하는 128차원 특징 벡터는 SIFT가 계산한다.
    kp1, desc1 = extract_patch_descriptors(img1, corners1, patch_size=descriptor_patch_size, **descriptor_params)
    kp2, desc2 = extract_patch_descriptors(img2, corners2, patch_size=descriptor_patch_size, **descriptor_params)
    kp3, desc3 = extract_patch_descriptors(img3, corners3, patch_size=descriptor_patch_size, **descriptor_params)

    # 3) img2를 기준으로 img1-img2, img3-img2 사이의 mutual match를 구한다.
    # 반환되는 src/dst는 실제 좌표 배열이고, matches는 인덱스와 ratio 정보를 담은 메타데이터다.
    raw_src12, raw_dst12, matches12 = knn_match_mutual(kp1, desc1, kp2, desc2, ratio_thresh=ratio_thresh)
    raw_src32, raw_dst32, matches32 = knn_match_mutual(kp3, desc3, kp2, desc2, ratio_thresh=ratio_thresh)

    # Homography는 최소 4개 대응점이 필요하다. 부족하면 이후 계산을 진행할 수 없다.
    if len(raw_src12) < 4:
        raise RuntimeError("Not enough matches between img1 and img2.")
    if len(raw_src32) < 4:
        raise RuntimeError("Not enough matches between img3 and img2.")

    # 4) RANSAC으로 outlier를 제거하면서 homography를 추정한다.
    # h12는 img1 좌표를 img2 좌표계로 보내는 3x3 행렬,
    # h32는 img3 좌표를 img2 좌표계로 보내는 3x3 행렬이다.
    h12, inliers12 = compute_homography_ransac(raw_src12, raw_dst12, **ransac_params)
    h32, inliers32 = compute_homography_ransac(raw_src32, raw_dst32, **ransac_params)

    # 불리언 마스크 배열을 이용해 inlier correspondence만 골라낸다.
    # NumPy에서 arr[mask]는 mask가 True인 행만 선택하는 대표적인 필터링 방식이다.
    src12 = raw_src12[inliers12]
    dst12 = raw_dst12[inliers12]
    src32 = raw_src32[inliers32]
    dst32 = raw_dst32[inliers32]

    # img2를 기준 좌표계로 사용하므로 img2는 스스로에게 변환할 필요가 없다.
    # np.eye(3)는 3x3 identity matrix를 만든다.
    h1_ref = h12
    h2_ref = np.eye(3, dtype=np.float64)
    h3_ref = h32

    # 세 이미지를 모두 담을 수 있는 panorama canvas를 계산한다.
    pano_h, pano_w, translation = compute_panorama_bounds([img1, img2, img3], [h1_ref, h2_ref, h3_ref])

    # 행렬곱 순서가 중요하다.
    # translation @ h1_ref는 "먼저 h1_ref로 img2 기준으로 옮기고, 그 다음 translation으로 캔버스 안에 넣는다"는 뜻이다.
    h1_canvas = translation @ h1_ref
    h2_canvas = translation @ h2_ref
    h3_canvas = translation @ h3_ref

    # 5) 세 이미지를 panorama canvas 위로 backward warping 한다.
    # 각 warp 함수는 (warped image, valid mask)를 반환한다.
    # mask는 캔버스의 어떤 픽셀이 실제로 원본 이미지에서 온 것인지 알려 준다.
    warp1, mask1 = backward_warp(img1, (pano_h, pano_w), h1_canvas)
    warp2, mask2 = backward_warp(img2, (pano_h, pano_w), h2_canvas)
    warp3, mask3 = backward_warp(img3, (pano_h, pano_w), h3_canvas)

    # 가장 단순한 baseline 결과: 겹치는 영역을 평균낸 raw panorama.
    panorama_raw = blend_average([warp1, warp2, warp3], [mask1, mask2, mask3])

    # 노출 차이로 seam이 도드라질 수 있어서, 겹치는 영역 평균 밝기를 img2 기준으로 먼저 맞춘다.
    warp1_brightness = match_overlap_brightness(warp1, mask1, warp2, mask2)
    warp3_brightness = match_overlap_brightness(warp3, mask3, warp2, mask2)

    # average blending은 단순하지만 seam이 남기 쉽고,
    # feather blending은 경계 쪽 가중치를 완만하게 바꿔 seam을 더 부드럽게 만든다.
    panorama_brightness = blend_average([warp1_brightness, warp2, warp3_brightness], [mask1, mask2, mask3])
    panorama_feather = blend_feather([warp1_brightness, warp2, warp3_brightness], [mask1, mask2, mask3])

    # 후처리와 시각화에 쓸 수 있도록 중간 결과를 dict로 모두 모아 반환한다.
    return {
        "corners1": corners1,
        "corners2": corners2,
        "corners3": corners3,
        "response1": response1,
        "response2": response2,
        "response3": response3,
        "raw_matches12_src": raw_src12,
        "raw_matches12_dst": raw_dst12,
        "raw_matches32_src": raw_src32,
        "raw_matches32_dst": raw_dst32,
        "matches12_src": src12,
        "matches12_dst": dst12,
        "matches32_src": src32,
        "matches32_dst": dst32,
        "matches12_meta": matches12,
        "matches32_meta": matches32,
        "matches12_inliers": inliers12,
        "matches32_inliers": inliers32,
        "H12": h12,
        "H32": h32,
        "warped_img1": np.clip(warp1, 0, 255).astype(np.uint8),
        "warped_img3": np.clip(warp3, 0, 255).astype(np.uint8),
        "panorama_raw": panorama_raw,
        "panorama_brightness": panorama_brightness,
        "panorama_feather": panorama_feather,
    }
