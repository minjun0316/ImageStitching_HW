from pathlib import Path

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    # cv2.imread는 파일을 NumPy 배열로 읽는다.
    # 반환 shape은 보통 (height, width, 3)이고 채널 순서는 BGR이다.
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image


def save_image(path: str, image: np.ndarray) -> None:
    # Path(path).parent는 저장 대상 파일의 상위 폴더다.
    # mkdir(..., exist_ok=True)를 써서 폴더가 이미 있어도 오류 없이 넘어가게 한다.
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    # 원본을 보존하기 위해 copy() 위에 그린다.
    # keypoints는 [x, y] 실수 좌표 배열이라 OpenCV drawing 함수에 넣기 전에 int로 바꾼다.
    canvas = image.copy()
    for x, y in keypoints.astype(int):
        cv2.circle(canvas, (x, y), 6, color, 2)
    return canvas


def draw_matches(
    img1: np.ndarray,
    kp1: np.ndarray,
    img2: np.ndarray,
    kp2: np.ndarray,
    max_draw: int = 60,
) -> np.ndarray:
    # 두 이미지를 좌우로 붙인 큰 canvas를 만든 뒤 대응점을 선으로 이어 매칭 품질을 본다.
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # np.zeros는 검은 배경 canvas를 만든다.
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # 매칭점이 너무 많으면 보기 어려우므로 앞에서부터 최대 max_draw개만 그린다.
    count = min(len(kp1), max_draw)
    for idx in range(count):
        x1, y1 = kp1[idx].astype(int)
        x2, y2 = kp2[idx].astype(int)
        # np.random.randint로 매칭마다 다른 색을 만들어 시각적으로 구분한다.
        color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
        # 오른쪽 이미지 점은 canvas에서 w1만큼 오른쪽으로 밀려 있으므로 x 좌표에 w1을 더한다.
        cv2.circle(canvas, (x1, y1), 6, color, 2)
        cv2.circle(canvas, (x2 + w1, y2), 6, color, 2)
        cv2.line(canvas, (x1, y1), (x2 + w1, y2), color, 2)
    return canvas
