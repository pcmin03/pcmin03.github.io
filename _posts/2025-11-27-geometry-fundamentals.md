---
title: 3D Geometry Fundamentals for Vision
categories: [3D Geometry]
tags: [3D Geometry, Camera Models]
article_header:
  type: overlay
  theme: dark
  background_color: '#0d1b2a'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(13, 27, 42, .85), rgba(176, 125, 103, .65))'
    src: /assets/images/study/3d-geometry.jpg
---

3D 비전 파이프라인의 핵심은 **카메라 모델, 좌표계, 변환 행렬**을 정확하게 다루는 것이다. 이 글은 프로젝트를 시작할 때 꼭 필요한 기초 공식만 모았다.

## 1. Camera Model
- **Pinhole model**: $s\begin{bmatrix}u\\v\\1\end{bmatrix}=K[R|t]\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}$
- **Intrinsic matrix** $K$는 focal length, principal point, skew를 포함한다.
- **Extrinsic** $[R|t]$는 월드 좌표를 카메라 좌표로 옮기는 회전·이동 행렬이다.

## 2. Coordinate Frames
- World, camera, image, pixel 좌표를 명확히 구분하고 단위(미터 vs 픽셀)를 기록한다.
- Homogeneous 좌표를 쓰면 translation을 행렬 곱으로 처리할 수 있다.

## 3. Transform Chain
1. World → Camera: $X_c = RX_w + t$
2. Camera → Image: $x = (X_c / Z_c, Y_c / Z_c)$
3. Image → Pixel: $u = f_x x + c_x$

## 4. Practical Tips
- `cv::Rodrigues` / `scipy.spatial.transform.Rotation`으로 회전 표현을 상호 변환한다.
- Double precision으로 optimization을 진행하고, 결과만 float32로 저장하면 수치적인 안정성이 올라간다.
- Pose graph를 다룰 때는 `SE(3)` 지수맵을 사용해 선형화 오류를 줄인다.
