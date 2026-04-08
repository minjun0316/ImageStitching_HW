import numpy as np

from src.harris import detect_harris_corners
from src.descriptor import extract_patch_descriptors
from src.matching import knn_match_mutual
from src.homography import apply_homography, compute_homography_ransac
from src.warping import backward_warp, blend_average, blend_feather, match_overlap_brightness


def image_corners(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def compute_panorama_bounds(images: list[np.ndarray], homographies: list[np.ndarray]) -> tuple[int, int, np.ndarray]:
    all_corners = []
    for image, h_matrix in zip(images, homographies):
        corners = image_corners(image)
        warped = apply_homography(corners, h_matrix)
        all_corners.append(warped)

    stacked = np.vstack(all_corners)
    min_xy = np.floor(stacked.min(axis=0)).astype(int)
    max_xy = np.ceil(stacked.max(axis=0)).astype(int)

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
    descriptor_params = descriptor_params or {}
    ransac_params = ransac_params or {}

    corners1, response1 = detect_harris_corners(img1, **harris_params)
    corners2, response2 = detect_harris_corners(img2, **harris_params)
    corners3, response3 = detect_harris_corners(img3, **harris_params)

    kp1, desc1 = extract_patch_descriptors(img1, corners1, patch_size=descriptor_patch_size, **descriptor_params)
    kp2, desc2 = extract_patch_descriptors(img2, corners2, patch_size=descriptor_patch_size, **descriptor_params)
    kp3, desc3 = extract_patch_descriptors(img3, corners3, patch_size=descriptor_patch_size, **descriptor_params)

    raw_src12, raw_dst12, matches12 = knn_match_mutual(kp1, desc1, kp2, desc2, ratio_thresh=ratio_thresh)
    raw_src32, raw_dst32, matches32 = knn_match_mutual(kp3, desc3, kp2, desc2, ratio_thresh=ratio_thresh)

    if len(raw_src12) < 4:
        raise RuntimeError("Not enough matches between img1 and img2.")
    if len(raw_src32) < 4:
        raise RuntimeError("Not enough matches between img3 and img2.")

    h12, inliers12 = compute_homography_ransac(raw_src12, raw_dst12, **ransac_params)
    h32, inliers32 = compute_homography_ransac(raw_src32, raw_dst32, **ransac_params)

    src12 = raw_src12[inliers12]
    dst12 = raw_dst12[inliers12]
    src32 = raw_src32[inliers32]
    dst32 = raw_dst32[inliers32]

    h1_ref = h12
    h2_ref = np.eye(3, dtype=np.float64)
    h3_ref = h32

    pano_h, pano_w, translation = compute_panorama_bounds([img1, img2, img3], [h1_ref, h2_ref, h3_ref])

    h1_canvas = translation @ h1_ref
    h2_canvas = translation @ h2_ref
    h3_canvas = translation @ h3_ref

    warp1, mask1 = backward_warp(img1, (pano_h, pano_w), h1_canvas)
    warp2, mask2 = backward_warp(img2, (pano_h, pano_w), h2_canvas)
    warp3, mask3 = backward_warp(img3, (pano_h, pano_w), h3_canvas)

    panorama_raw = blend_average([warp1, warp2, warp3], [mask1, mask2, mask3])

    warp1_brightness = match_overlap_brightness(warp1, mask1, warp2, mask2)
    warp3_brightness = match_overlap_brightness(warp3, mask3, warp2, mask2)
    panorama_brightness = blend_average([warp1_brightness, warp2, warp3_brightness], [mask1, mask2, mask3])
    panorama_feather = blend_feather([warp1_brightness, warp2, warp3_brightness], [mask1, mask2, mask3])

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
