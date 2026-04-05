from pathlib import Path

from src.stitch import stitch_three_images
from src.utils import draw_keypoints, draw_matches, load_image, save_image


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    results_dir = root / "results"

    img1 = load_image(str(data_dir / "img1.jpg"))
    img2 = load_image(str(data_dir / "img2.jpg"))
    img3 = load_image(str(data_dir / "img3.jpg"))

    harris_params = {
        "max_corners": 800,
        "response_thresh_ratio": 0.01,
        "k": 0.04,
        "window_size": 3,
        "nms_window": 7,
        "border_margin": 12,
    }

    results = stitch_three_images(
        img1=img1,
        img2=img2,
        img3=img3,
        harris_params=harris_params,
        descriptor_patch_size=11,
        ratio_thresh=0.75,
    )

    save_image(str(results_dir / "corners_img1.jpg"), draw_keypoints(img1, results["corners1"]))
    save_image(str(results_dir / "corners_img2.jpg"), draw_keypoints(img2, results["corners2"]))
    save_image(str(results_dir / "corners_img3.jpg"), draw_keypoints(img3, results["corners3"]))
    save_image(
        str(results_dir / "matches_12.jpg"),
        draw_matches(img1, results["matches12_src"], img2, results["matches12_dst"]),
    )
    save_image(
        str(results_dir / "matches_23.jpg"),
        draw_matches(img2, results["matches32_dst"], img3, results["matches32_src"]),
    )
    save_image(str(results_dir / "panorama.jpg"), results["panorama"])

    print("Image stitching completed.")
    print(f"H12 (img1 -> img2):\n{results['H12']}")
    print(f"H32 (img3 -> img2):\n{results['H32']}")
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    main()
