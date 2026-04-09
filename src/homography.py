import numpy as np


def compute_homography_least_squares(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    # Homography H는 [x, y, 1]을 [u, v, 1] 쪽으로 보내는 3x3 변환 행렬이다.
    # 점 대응쌍으로부터 선형 방정식 Ah=b를 만들고 np.linalg.lstsq로 근사해를 구한다.
    if len(src_pts) < 4 or len(dst_pts) < 4:
        raise ValueError("At least 4 point correspondences are required.")

    a_rows = []
    b_rows = []

    for (x, y), (u, v) in zip(src_pts, dst_pts):
        # h33을 1로 두면 미지수는 8개가 되고, 한 점 대응쌍은 u식/v식 두 개를 제공한다.
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b_rows.append(u)
        b_rows.append(v)

    a = np.asarray(a_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)

    # h33을 1로 고정한 8개 미지수 문제로 풀고 마지막 원소를 다시 붙인다.
    h, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    h = np.append(h, 1.0)
    h_matrix = h.reshape(3, 3)
    return h_matrix


def apply_homography(points: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    # np.hstack은 좌표 배열 옆에 1 열을 붙여 [x, y, 1] homogeneous coordinate를 만든다.
    # 2D projective transform은 3x3 행렬과 homogeneous 좌표를 곱하는 형태로 표현한다.
    homogeneous = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    # @ 연산자는 NumPy의 행렬곱이다. .T는 transpose다.
    projected = (h_matrix @ homogeneous.T).T
    # 마지막 좌표 w로 나누는 과정을 dehomogenization이라고 한다.
    projected /= projected[:, 2:3]
    return projected[:, :2].astype(np.float32)


def compute_reprojection_errors(src_pts: np.ndarray, dst_pts: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    # np.linalg.norm(..., axis=1)은 각 점쌍마다 유클리드 거리 하나씩 계산한다.
    projected = apply_homography(src_pts, h_matrix)
    return np.linalg.norm(projected - dst_pts, axis=1)


def compute_homography_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    num_iters: int = 2000,
    inlier_threshold: float = 3.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    # RANSAC은 "랜덤 샘플로 모델 추정 -> 모든 점에서 오차 계산 -> inlier 수 비교"를 반복한다.
    # 잘못된 매칭(outlier)이 섞여 있어도 가장 일관된 변환을 찾기 쉽다.
    # 여기서 제거 대상은 "descriptor 상으로는 매칭됐지만, 같은 평면 변환을 따른다고 보기 어려운 correspondence"다.
    # 즉 KNN matching은 비슷한 patch를 찾는 단계이고, RANSAC은 그 match들이 기하학적으로 말이 되는지 검증하는 단계다.
    if len(src_pts) < 4 or len(dst_pts) < 4:
        raise ValueError("At least 4 point correspondences are required for RANSAC.")

    # default_rng는 NumPy의 난수 생성기다. random_state를 고정하면 반복 실행해도 결과가 재현된다.
    rng = np.random.default_rng(random_state)
    best_inlier_mask = np.zeros(len(src_pts), dtype=bool)
    best_h = None

    for _ in range(num_iters):
        # Homography는 최소 4쌍 점으로 추정 가능하므로 매 반복마다 4개를 뽑는다.
        sample_idx = rng.choice(len(src_pts), size=4, replace=False)

        try:
            h_candidate = compute_homography_least_squares(src_pts[sample_idx], dst_pts[sample_idx])
            errors = compute_reprojection_errors(src_pts, dst_pts, h_candidate)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            # 샘플이 나쁘면 역행렬 성질이 불안정하거나 수치적으로 실패할 수 있으니 그냥 다음 반복으로 넘어간다.
            continue

        if not np.all(np.isfinite(errors)):
            continue

        # threshold보다 오차가 작은 점만 inlier로 본다.
        # reprojection error가 크면 현재 homography로 설명되지 않는 match이므로 outlier 취급한다.
        inlier_mask = errors < inlier_threshold

        # 현재까지 가장 많은 inlier를 설명하는 모델을 유지한다.
        if inlier_mask.sum() > best_inlier_mask.sum():
            best_inlier_mask = inlier_mask
            best_h = h_candidate

    if best_h is None or best_inlier_mask.sum() < 4:
        raise RuntimeError("RANSAC failed to find a valid homography.")

    # 가장 좋은 inlier 집합으로 homography를 다시 추정해 최종 결과로 사용한다.
    # 따라서 최종 homography는 "RANSAC이 남긴 inlier들만"으로 계산된다.
    refined_h = compute_homography_least_squares(src_pts[best_inlier_mask], dst_pts[best_inlier_mask])
    return refined_h, best_inlier_mask
