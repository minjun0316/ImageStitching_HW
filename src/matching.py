import numpy as np


def pairwise_distances(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    # 두 descriptor 집합 사이의 모든 쌍에 대한 squared Euclidean distance를 계산한다.
    # 결과 shape은 (len(desc1), len(desc2)) 이다.
    desc1_sq = np.sum(desc1 ** 2, axis=1, keepdims=True)
    # [None, :]를 붙이면 shape이 (M,) -> (1, M)이 되어 브로드캐스팅에 맞는다.
    desc2_sq = np.sum(desc2 ** 2, axis=1)[None, :]
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b 공식을 이용하면 거리행렬을 한 번에 계산할 수 있다.
    dists = desc1_sq + desc2_sq - 2.0 * desc1 @ desc2.T
    # 수치오차로 아주 작은 음수가 생길 수 있어 0 아래는 잘라낸다.
    return np.maximum(dists, 0.0)


def _ratio_candidates(desc_query: np.ndarray, desc_train: np.ndarray, ratio_thresh: float) -> dict[int, tuple[int, float]]:
    # Lowe ratio test를 적용해 "가장 가까운 점이 두 번째보다 충분히 더 가까운 경우"만 후보로 남긴다.
    if len(desc_query) < 2 or len(desc_train) < 2:
        return {}

    dists = pairwise_distances(desc_query, desc_train)
    candidates: dict[int, tuple[int, float]] = {}

    for i in range(dists.shape[0]):
        # np.argsort(dists[i])[:2]는 i번째 query descriptor에 대한 최근접/차근접 이웃 인덱스다.
        nn_idx = np.argsort(dists[i])[:2]
        best, second = dists[i, nn_idx[0]], dists[i, nn_idx[1]]
        ratio = best / (second + 1e-12)

        # ratio가 작을수록 1등 이웃이 2등보다 뚜렷하게 가깝다는 뜻이라 더 신뢰할 만하다.
        if ratio < ratio_thresh:
            candidates[i] = (int(nn_idx[0]), float(ratio))

    return candidates


def knn_match_mutual(
    kp1: np.ndarray,
    desc1: np.ndarray,
    kp2: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    # ratio test를 하려면 최소한 최근접/차근접 두 개가 있어야 하므로 descriptor가 2개 미만이면 종료한다.
    if len(desc1) < 2 or len(desc2) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

    # forward는 1->2, backward는 2->1 방향 ratio 후보들이다.
    forward = _ratio_candidates(desc1, desc2, ratio_thresh)
    backward = _ratio_candidates(desc2, desc1, ratio_thresh)
    matches = []

    for i, (j, ratio_ij) in forward.items():
        # mutual matching: 1의 i가 2의 j를 가리키고, 동시에 2의 j도 다시 1의 i를 가리켜야 채택한다.
        reverse = backward.get(j)
        if reverse is None:
            continue
        reverse_idx, ratio_ji = reverse
        if reverse_idx != i:
            continue
        # 양방향 ratio 중 더 나쁜 쪽(max)을 기록해 보수적으로 점수를 저장한다.
        matches.append((i, j, float(max(ratio_ij, ratio_ji))))

    # ratio가 작은 순으로 정렬하면 앞쪽 매칭일수록 더 신뢰도가 높은 편이다.
    matches.sort(key=lambda item: item[2])

    if not matches:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

    # 매칭 인덱스를 실제 좌표 배열로 변환해 이후 homography 계산에 바로 쓸 수 있게 만든다.
    src = np.asarray([kp1[i] for i, _, _ in matches], dtype=np.float32)
    dst = np.asarray([kp2[j] for _, j, _ in matches], dtype=np.float32)
    return src, dst, matches
