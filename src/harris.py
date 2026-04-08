import cv2
import numpy as np


def harris_response(gray: np.ndarray, k: float = 0.04, window_size: int = 3) -> np.ndarray:
    # Harris corner detector는 밝기 변화의 2차 구조를 이용해 "코너다운 지점"을 찾는다.
    # 입력은 grayscale 이미지이며, 계산 안정성을 위해 float32로 바꿔 둔다.
    gray = gray.astype(np.float32)
    # cv2.Sobel은 x/y 방향 미분(gradient)을 구하는 함수다.
    # ix는 x방향 변화량, iy는 y방향 변화량으로 edge/corner 계산의 기초가 된다.
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # structure tensor M의 성분을 만들기 위해 Ix^2, Iy^2, IxIy를 계산한다.
    # GaussianBlur는 주변 정보를 평균내는 smoothing 역할을 해서 response를 더 안정적으로 만든다.
    ixx = cv2.GaussianBlur(ix * ix, (window_size, window_size), 1.0)
    iyy = cv2.GaussianBlur(iy * iy, (window_size, window_size), 1.0)
    ixy = cv2.GaussianBlur(ix * iy, (window_size, window_size), 1.0)

    # Harris 식 R = det(M) - k * trace(M)^2
    # det가 크고 trace에 비해 충분히 크면 두 방향으로 변화가 큰 코너로 해석된다.
    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    response = det - k * (trace ** 2)
    return response


def non_maximum_suppression(response: np.ndarray, nms_window: int = 7) -> np.ndarray:
    # cv2.dilate는 주변 window 안의 최댓값을 퍼뜨린다.
    # 원래 response와 dilated 결과가 같은 위치만 남기면 local maximum만 고를 수 있다.
    dilated = cv2.dilate(response, np.ones((nms_window, nms_window), dtype=np.uint8))
    return response == dilated


def detect_harris_corners(
    image: np.ndarray,
    max_corners: int = 800,
    response_thresh_ratio: float = 0.01,
    k: float = 0.04,
    window_size: int = 3,
    nms_window: int = 7,
    border_margin: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    # Harris response는 grayscale에서 계산하므로 먼저 BGR 이미지를 단일 채널 grayscale로 바꾼다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = harris_response(gray, k=k, window_size=window_size)

    # response.max()의 일정 비율보다 큰 점만 후보로 두어 약한 코너를 제거한다.
    threshold = response_thresh_ratio * response.max()
    # 불리언 마스크끼리 &를 쓰면 "threshold 통과"이면서 "local maximum"인 점만 남는다.
    mask = (response > threshold) & non_maximum_suppression(response, nms_window=nms_window)

    # 가장자리 근처는 descriptor 계산 시 patch가 잘리기 쉬워서 미리 제외한다.
    h, w = gray.shape
    mask[:border_margin, :] = False
    mask[-border_margin:, :] = False
    mask[:, :border_margin] = False
    mask[:, -border_margin:] = False

    # np.where(mask)는 True인 위치의 (y좌표 배열, x좌표 배열)을 반환한다.
    ys, xs = np.where(mask)
    # 같은 위치의 response 값을 꺼내 각 코너의 점수로 사용한다.
    scores = response[ys, xs]

    if len(scores) == 0:
        return np.empty((0, 2), dtype=np.float32), response

    # np.argsort(scores)[::-1]는 점수를 큰 순서대로 정렬하기 위한 인덱스 배열이다.
    order = np.argsort(scores)[::-1][:max_corners]
    # [x, y] 형식으로 쌓아서 최종 corner 좌표 배열을 만든다.
    corners = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
    return corners, response
