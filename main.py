from pathlib import Path

from src.stitch import stitch_three_images
from src.utils import draw_keypoints, draw_matches, load_image, save_image


def resolve_input_images(data_dir: Path) -> tuple[Path, Path, Path]:
    # 기본 파일명이 있으면 그대로 쓰고, 없으면 data 폴더의 이미지 3장을 이름순으로 사용한다.
    default_paths = [data_dir / "img1.jpg", data_dir / "img2.jpg", data_dir / "img3.jpg"]
    if all(path.exists() for path in default_paths):
        return tuple(default_paths)

    candidates = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        candidates.extend(sorted(data_dir.glob(pattern)))

    unique_candidates = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)

    if len(unique_candidates) < 3:
        raise FileNotFoundError(
            f"Expected 3 input images in {data_dir}, but found {len(unique_candidates)}."
        )

    return tuple(unique_candidates[:3])


def main() -> None:
    # pathlib.Path는 파일 경로를 "문자열"이 아니라 "경로 객체"로 다루게 해 준다.
    # __file__은 현재 실행 중인 main.py의 경로이고, resolve()는 이를 절대경로로 정리한다.
    # parent를 붙이면 main.py가 들어 있는 프로젝트 루트 폴더를 얻을 수 있다.
    root = Path(__file__).resolve().parent
    # Path 객체끼리는 "/" 연산자로 경로를 이어 붙일 수 있다.
    data_dir = root / "data"
    results_dir = root / "results"

    # load_image 내부에서는 cv2.imread를 사용한다.
    # OpenCV는 이미지를 NumPy 배열 형태로 읽고, 채널 순서는 RGB가 아니라 BGR이다.
    # 이 프로젝트는 img2를 기준 이미지로 두고, img1과 img3를 img2 쪽으로 정렬한다.
    img1_path, img2_path, img3_path = resolve_input_images(data_dir)
    img1 = load_image(str(img1_path))
    img2 = load_image(str(img2_path))
    img3 = load_image(str(img3_path))

    # Harris detector 설정을 dict로 묶어 두면 함수 호출 시 **dict 형태로 한 번에 펼쳐 넘길 수 있다.
    # 예를 들어 detect_harris_corners(img1, **harris_params)는
    # detect_harris_corners(img1, max_corners=800, ...)와 같은 뜻이다.
    harris_params = {
        "max_corners": 800,
        "response_thresh_ratio": 0.01,
        "k": 0.04,
        "window_size": 3,
        "nms_window": 7,
        "border_margin": 12,
    }

    # stitch_three_images는 이 프로젝트의 핵심 파이프라인 함수다.
    # 내부에서 코너 검출 -> descriptor 계산 -> 매칭 -> homography -> warping -> blending을 수행한다.
    # 반환값은 dict이며, 중간 결과(코너, 매칭점, H 행렬)와 최종 결과 이미지가 함께 들어 있다.
    results = stitch_three_images(
        img1=img1,
        img2=img2,
        img3=img3,
        harris_params=harris_params,
        descriptor_patch_size=17,
        ratio_thresh=0.85,
        descriptor_params={
            "num_cells": 4,
            "num_bins": 8,
            "smoothing_sigma": 1.2,
        },
        ransac_params={
            "num_iters": 2000,
            "inlier_threshold": 3.0,
            "random_state": 42,
        },
    )

    # draw_keypoints와 draw_matches는 시각화용 보조 함수다.
    # save_image는 cv2.imwrite를 감싸서 결과를 파일로 저장한다.
    # results["..."]처럼 dict에서 키로 값을 꺼내 원하는 결과 이미지를 저장한다.
    save_image(str(results_dir / "corners_img1.jpg"), draw_keypoints(img1, results["corners1"]))
    save_image(str(results_dir / "corners_img2.jpg"), draw_keypoints(img2, results["corners2"]))
    save_image(str(results_dir / "corners_img3.jpg"), draw_keypoints(img3, results["corners3"]))
    save_image(str(results_dir / "before_matches_warped_img1.jpg"), results["warped_img1"])
    save_image(str(results_dir / "before_matches_warped_img3.jpg"), results["warped_img3"])
    save_image(
        str(results_dir / "matches_12.jpg"),
        draw_matches(img1, results["matches12_src"], img2, results["matches12_dst"]),
    )
    save_image(
        str(results_dir / "matches_23.jpg"),
        draw_matches(img2, results["matches32_dst"], img3, results["matches32_src"]),
    )
    save_image(str(results_dir / "panorama_raw.jpg"), results["panorama_raw"])
    save_image(str(results_dir / "panorama_brightness.jpg"), results["panorama_brightness"])
    save_image(str(results_dir / "panorama_feather.jpg"), results["panorama_feather"])

    # pathlib의 exists()는 파일 존재 여부를 확인하고, unlink()는 파일을 삭제한다.
    # 예전 출력 파일명이 남아 있으면 현재 결과와 혼동될 수 있어서 정리한다.
    legacy_panorama = results_dir / "panorama.jpg"
    if legacy_panorama.exists():
        legacy_panorama.unlink()
    legacy_panorama_average = results_dir / "panorama_average.jpg"
    if legacy_panorama_average.exists():
        legacy_panorama_average.unlink()
    legacy_warp1 = results_dir / "warped_img1.jpg"
    if legacy_warp1.exists():
        legacy_warp1.unlink()
    legacy_warp3 = results_dir / "warped_img3.jpg"
    if legacy_warp3.exists():
        legacy_warp3.unlink()

    # 터미널 로그는 결과 품질을 빠르게 보는 용도다.
    # raw_matches는 mutual matching 통과 수, matches는 RANSAC inlier만 남긴 수다.
    # H12, H32는 각각 img1->img2, img3->img2 호모그래피 행렬이다.
    print("Image stitching completed.")
    print(f"Input images: {img1_path.name}, {img2_path.name}, {img3_path.name}")
    print(f"Mutual matches img1-img2: {len(results['raw_matches12_src'])}, RANSAC inliers: {len(results['matches12_src'])}")
    print(f"Mutual matches img3-img2: {len(results['raw_matches32_src'])}, RANSAC inliers: {len(results['matches32_src'])}")
    print(f"H12 (img1 -> img2):\n{results['H12']}")
    print(f"H32 (img3 -> img2):\n{results['H32']}")
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    main()
