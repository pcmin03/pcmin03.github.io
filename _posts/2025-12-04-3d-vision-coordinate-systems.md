---
title: "3D Vision Tutorial: Understanding 3D Coordinate Systems"
categories: [3D Vision]
tags: [3D Vision, Tutorial, YouTube, Coordinate Systems]
article_header:
  type: overlay
  theme: dark
  background_color: '#0d1b2a'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(13, 27, 42, .85), rgba(176, 125, 103, .65))'
    src: /assets/images/study/3d-geometry.jpg
---

이 포스트에서는 3D Vision의 기초인 3D Coordinate Systems에 대해 다룹니다. 아래 YouTube 비디오를 통해 자세히 알아보세요.

<!--more-->

## 3D Coordinate Systems 이해하기

<div style="position: relative; width: 100%; height: 0; padding-bottom: 56.25%; margin-bottom: 1.5rem;">
  <iframe 
    src="https://www.youtube.com/embed/hgBlCaCIV10?list=PLubUquiqNQdN83-fPBzzViEEqohpdlwk2&index=5" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen 
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
  ></iframe>
</div>

## 주요 내용 요약

### 1. 좌표계의 종류

**Cartesian Coordinates (직교 좌표계)**
- 가장 일반적인 3D 좌표계
- X, Y, Z 세 개의 수직 축 사용
- Right-handed와 Left-handed 시스템 구분

**Spherical Coordinates (구면 좌표계)**
- 반경(r), 방위각(θ), 고도각(φ) 사용
- 3D 공간의 점을 구면으로 표현

**Cylindrical Coordinates (원통 좌표계)**
- 반경(r), 각도(θ), 높이(z) 사용
- 원통 형태의 좌표 표현

### 2. 좌표 변환 (Transformations)

**Translation (평행 이동)**
- 3D 공간에서 객체를 이동시키는 변환
- Translation vector를 사용

**Rotation (회전)**
- Euler Angles: 세 개의 회전 각도
- Quaternions: 짐벌 락을 피하는 4원수 표현
- Rotation Matrices: 3x3 회전 행렬

**Scaling (크기 조정)**
- 균일 또는 비균일 스케일링
- 각 축별로 독립적인 스케일링 가능

### 3. Homogeneous Coordinates (동차 좌표)

- 4D 표현 (x, y, z, w) 사용
- 행렬 곱셈으로 변환 결합 가능
- 원근 투영(perspective projection) 지원
- 효율적인 계산 가능

### 4. 좌표계의 종류

**World Space (월드 좌표계)**
- 전역 좌표계
- 모든 객체의 절대 위치

**Local Space (로컬 좌표계)**
- 객체 상대 좌표계
- 객체의 중심을 기준으로 한 좌표

**View Space (뷰 좌표계)**
- 카메라 상대 좌표계
- 카메라를 기준으로 한 좌표

**Screen Space (스크린 좌표계)**
- 2D 투영 좌표계
- 최종 화면에 표시되는 좌표

## 응용 분야

- **Computer Graphics**: 렌더링 및 애니메이션
- **Robotics**: 운동학 및 경로 계획
- **Medical Imaging**: 3D 재구성 및 분석
- **Game Development**: 3D 월드 표현
- **Computer Vision**: 3D 객체 인식 및 추적

## 관련 링크

- [YouTube 비디오](https://www.youtube.com/watch?v=hgBlCaCIV10&list=PLubUquiqNQdN83-fPBzzViEEqohpdlwk2&index=5)
- [전체 플레이리스트](https://www.youtube.com/playlist?list=PLubUquiqNQdN83-fPBzzViEEqohpdlwk2)

