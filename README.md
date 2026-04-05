# ImageStitching_HW

컴퓨터비전 과제용 image stitching 프로젝트다.

## Pipeline

1. `data/` 폴더의 `img1`, `img2`, `img3` 로드
2. Harris corner detector로 특징점 검출
3. Patch descriptor 생성
4. KNN matching + ratio test로 좋은 매칭점 선택
5. Least squares method로 homography 계산
6. Backward warping + bilinear interpolation으로 panorama 생성

## Structure

- `src/harris.py`: Harris corner detector
- `src/descriptor.py`: patch descriptor
- `src/matching.py`: KNN matching
- `src/homography.py`: least squares homography
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

## Output

실행 후 `results/` 폴더에 아래 결과가 저장된다.

- Harris corner visualization
- `img1-img2` matching visualization
- `img2-img3` matching visualization
- 최종 panorama

## Notes

- 현재 구현은 과제 파이프라인에 맞춰 Harris + patch descriptor를 사용한다.
- Homography는 least squares로 계산하며, 별도의 RANSAC은 넣지 않았다.
- 실제 촬영 이미지에서는 corner 수, patch 크기, ratio threshold를 조정해야 할 수 있다.
