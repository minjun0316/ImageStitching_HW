# ImageStitching_HW

컴퓨터비전 과제용 image stitching 프로젝트다.

## Pipeline

### 기존 파이프라인

1. `data/` 폴더의 `img1`, `img2`, `img3` 로드
2. Harris corner detector로 특징점 검출
3. Patch descriptor 생성
4. KNN matching + ratio test로 좋은 매칭점 선택
5. Least squares method로 homography 계산
6. Backward warping + bilinear interpolation으로 panorama 생성

### 수정된 파이프라인

1. `data/` 폴더의 `img1`, `img2`, `img3` 로드
2. Harris corner detector로 특징점 검출
3. Gaussian smoothing 후 Harris keypoint 위치에서 SIFT descriptor 생성
4. SIFT의 gradient histogram과 dominant orientation을 이용해 orientation normalization 수행
5. 양방향 KNN ratio test 후 mutual matching만 유지
6. RANSAC으로 outlier를 제거한 뒤 inlier만 사용해 homography 재추정
7. Backward warping + bilinear interpolation으로 panorama 생성
8. overlap 영역 평균 밝기에 맞춰 `img1`, `img3`를 `img2` 기준으로 gain 보정
9. feather blending으로 seam 경계를 완화해 최종 panorama 생성

## Structure

- `src/harris.py`: Harris corner detector
- `src/descriptor.py`: Harris keypoint 위에서 계산하는 SIFT descriptor
- `src/matching.py`: bidirectional KNN + mutual matching
- `src/homography.py`: least squares homography + RANSAC
- `src/warping.py`: backward warping, interpolation
- `src/stitch.py`: 3장 stitching 파이프라인
- `src/utils.py`: 입출력 및 시각화
- `main.py`: 전체 실행

## Usage

1. `data/` 폴더에 아래 이름으로 이미지를 넣는다.

```text
data/img1.jpg
data/img2.jpg
data/img3.jpg
```

2. 실행한다.

```bash
python3 main.py
```

기본 설정은 아래처럼 바뀌었다.

- descriptor patch size: `17`
- ratio threshold: `0.7`
- RANSAC iterations: `2000`
- RANSAC inlier threshold: `3.0`

## Output

실행 후 `results/` 폴더에 아래 결과가 저장된다.

- Harris corner visualization
- `img1-img2` RANSAC inlier matching visualization
- `img2-img3` RANSAC inlier matching visualization
- `panorama_raw.jpg`: warping 직후 average blending 결과
- `panorama_brightness.jpg`: overlap 기반 밝기 보정 후 average blending 결과
- `panorama_feather.jpg`: feather blending까지 적용한 최종 결과
- `panorama.jpg`: 최종 결과와 동일한 feather panorama

터미널에는 아래 정보도 함께 출력된다.

- mutual matches 개수
- RANSAC inlier 개수
- 최종 homography

## 변경 내용과 이유

### 1. RANSAC 추가

- 기존: 모든 매칭점을 그대로 least squares homography 계산에 사용
- 변경: 랜덤하게 4점을 뽑아 homography를 반복 추정하고, reprojection error가 작은 inlier만 남긴 뒤 다시 homography 계산
- 이유: 반복 패턴이나 잘못된 correspondence가 섞이면 homography 전체가 쉽게 무너졌기 때문

### 2. Mutual matching 추가

- 기존: `A -> B` 한 방향 ratio test만 사용
- 변경: `A -> B`, `B -> A` 모두 통과하고 서로를 다시 가리키는 매칭만 채택
- 이유: 비슷한 코너가 많은 장면에서 one-way nearest neighbor가 잘못된 짝을 만들 가능성이 컸기 때문

### 3. Descriptor 개선

- 기존: raw intensity patch를 평균 / 표준편차 정규화해서 사용
- 변경: Harris corner 위치에서 SIFT descriptor를 계산하도록 바꾸고, Gaussian smoothing과 dominant orientation normalization을 사용
- 이유: raw patch보다 밝기 변화, 소규모 회전, 국소 texture 변화에 더 안정적으로 대응하기 위해서

### 4. Matching 조건 강화

- Lowe ratio test는 `best / second < threshold` 구조이므로 threshold를 올리면 조건이 느슨해진다.
- 그래서 실제로 매칭을 더 강하게 거르기 위해 기본값을 `0.75 -> 0.7`로 낮췄다.
- 이유: 요청한 "matching 조건 강화"를 구현할 때 ratio test의 수학적 의미상 더 작은 threshold가 더 엄격하기 때문이다.

### 5. 밝기 보정 추가

- 기존: warping 후 겹치는 영역을 그대로 average blending
- 변경: overlap 영역 평균 밝기를 비교해서 `img1`, `img3`를 `img2` 기준 밝기에 맞추는 gain compensation 적용
- 이유: 촬영 시 노출 차이 때문에 겹치는 부분의 밝기 경계가 눈에 띄었기 때문

### 6. 경계 완화 blending 추가

- 기존: 겹치는 영역에서 단순 average blending만 사용
- 변경: 각 warped image mask의 내부 가중치를 더 크게 두는 feather blending 적용
- 이유: seam 경계가 선처럼 보이는 현상을 줄이고 겹침 구간 전환을 더 부드럽게 만들기 위해서

## 단계별 Panorama 결과 해석

- `panorama_raw.jpg`
  - 적용: backward warping + average blending
  - 목적: 세 장을 하나의 panorama 좌표계로 먼저 합치는 단계
  - 남는 문제: 노출 차이, seam 경계

- `panorama_brightness.jpg`
  - 적용: overlap-based brightness gain compensation + average blending
  - 목적: 이미지 간 밝기 차이로 생기는 경계 완화
  - 줄인 문제: 노출 차이, 명암 불일치

- `panorama_feather.jpg`
  - 적용: brightness compensation 후 feather blending
  - 목적: 겹치는 영역의 전환을 부드럽게 만들어 seam 완화
  - 줄인 문제: 경계선이 도드라져 보이는 seam artifact

## 한계점

- Harris detector 기반이라 큰 scale 변화나 극단적인 viewpoint 변화에는 여전히 약하다.
- descriptor는 SIFT로 강화했지만 detector 자체가 scale-invariant하지 않아서 모든 장면에서 안정적이지는 않다.
- RANSAC threshold와 iteration 수는 데이터셋에 따라 다시 튜닝이 필요하다.
- overlap 기반 밝기 보정과 feather blending으로 경계는 완화했지만, 복잡한 장면에서는 ghosting이나 잔여 seam artifact가 남을 수 있다.

## 커밋 히스토리용 요약

GitHub 커밋 메시지나 README 변경 이력에 아래 요약을 그대로 써도 된다.

```text
Baseline pipeline:
Harris corners -> raw patch descriptor -> one-way KNN ratio test -> least-squares homography -> backward warping

Updated pipeline:
Harris corners -> SIFT descriptor on Harris keypoints with Gaussian smoothing/orientation normalization
-> bidirectional mutual ratio matching -> RANSAC inlier filtering -> refined homography -> backward warping
-> overlap-based brightness gain compensation
-> feather blending for seam reduction

Why changed:
- reduce repeated-pattern mismatches
- reject outliers before homography estimation
- make descriptor more robust to illumination and small rotation changes
- tighten matching quality with a stricter Lowe ratio threshold

Remaining limitations:
- still limited by Harris detection under large scale/viewpoint change
- RANSAC and descriptor hyperparameters still need dataset-specific tuning
- brightness compensation and feather blending reduce seams, but ghosting can still remain
```
