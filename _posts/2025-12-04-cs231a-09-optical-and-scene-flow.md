---
title: "[CS231A] Lecture 09: Optical and Scene Flow (광학 흐름 및 장면 흐름)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Optical Flow, Scene Flow, Motion Estimation]
article_header:
  type: overlay
  theme: dark
  background_color: '#0d1b2a'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(13, 27, 42, .85), rgba(176, 125, 103, .65))'
    src: /assets/images/study/3d-geometry.jpg
mathjax: true
mathjax_autoNumber: true
---

**Stanford CS231A: Computer Vision, From 3D Reconstruction to Recognition**

이 포스트는 Stanford CS231A 강의의 아홉 번째 강의 노트인 "Optical and Scene Flow"를 정리한 것입니다.

**원본 강의 노트**: [09-optical-flow.pdf](https://web.stanford.edu/class/cs231a/course_notes/09-optical-flow.pdf)

<!--more-->

## 1. 서론 (Introduction)

광학 흐름(Optical Flow)과 장면 흐름(Scene Flow)은 **이미지 시퀀스에서 픽셀의 움직임을 추정하는 기술**입니다. 비디오 분석, 추적, 자율 주행 등 다양한 응용 분야에서 활용됩니다.

## 2. 모션 필드 (Motion Field)

### 2.1. 모션 필드의 개념

모션 필드는 3D 장면에서 점들의 실제 움직임을 나타냅니다. 이는 카메라의 움직임과 장면 내 객체의 움직임의 조합입니다.

![Optical Flow Overview](/assets/images/posts/cs231a-09/figures/page_1_img_1.png)

### 2.2. 광학 흐름 vs 모션 필드

- **모션 필드**: 3D 공간에서의 실제 움직임
- **광학 흐름**: 2D 이미지 평면에서 관찰되는 픽셀의 움직임

광학 흐름은 모션 필드의 투영이지만, 항상 일치하지는 않습니다 (예: 텍스처 없는 영역).

## 3. 광학 흐름 계산 (Computing Optical Flow)

### 3.1. 광학 흐름 제약 (Optical Flow Constraint)

밝기 일정성 가정(Brightness Constancy Assumption)을 사용합니다:

$$I(x, y, t) = I(x + u, y + v, t + \Delta t)$$

여기서 $(u, v)$는 광학 흐름 벡터입니다.

### 3.2. Lucas-Kanade 방법

작은 윈도우 내에서 광학 흐름을 추정하는 지역적 방법입니다:

$$\begin{pmatrix} u \\ v \end{pmatrix} = (A^T A)^{-1} A^T b$$

여기서 $A$는 공간 그래디언트, $b$는 시간 그래디언트입니다.

### 3.3. Horn-Schunck 방법

전역적 최적화를 통한 광학 흐름 추정:

$$E = \int \int [(I_x u + I_y v + I_t)^2 + \lambda (u_x^2 + u_y^2 + v_x^2 + v_y^2)] dx dy$$

여기서 첫 번째 항은 데이터 항, 두 번째 항은 부드러움 항입니다.

## 4. 다중 스케일 광학 흐름 (Multi-scale Optical Flow)

### 4.1. 피라미드 기반 방법

이미지 피라미드를 구축하여 큰 움직임도 처리할 수 있도록 합니다:
1. 저해상도에서 광학 흐름 추정
2. 고해상도로 업샘플링
3. 고해상도에서 정제

### 4.2. Coarse-to-Fine 전략

거친 스케일에서 시작하여 점진적으로 세밀한 스케일로 정제합니다.

## 5. 장면 흐름 (Scene Flow)

### 5.1. 장면 흐름의 개념

장면 흐름은 **3D 공간에서 점들의 3D 움직임 벡터**입니다. 광학 흐름의 3D 확장입니다.

### 5.2. 스테레오 + 광학 흐름

스테레오 비전과 광학 흐름을 결합하여 장면 흐름을 추정할 수 있습니다:
- 스테레오로 3D 위치 추정
- 광학 흐름으로 2D 움직임 추정
- 두 정보를 결합하여 3D 움직임 계산

## 6. 딥러닝 기반 광학 흐름

### 6.1. FlowNet

딥러닝을 이용한 광학 흐름 추정 네트워크입니다:
- **FlowNet**: 지도 학습 기반
- **FlowNet 2.0**: 개선된 버전
- **PWC-Net**: 피라미드 워프 코스트 볼륨 사용

![Optical Flow Computation](/assets/images/posts/cs231a-09/figures/page_2_img_1.png)
![Optical Flow Results](/assets/images/posts/cs231a-09/figures/page_3_img_1.png)

### 6.2. 자기 지도 학습

비디오 데이터만으로 학습하는 방법:
- **Unsupervised Optical Flow**: 재구성 손실 사용
- **Self-supervised Learning**: 일관성 손실 사용

## 7. 응용 분야 (Applications)

광학 흐름과 장면 흐름은 다음과 같은 분야에서 활용됩니다:

- **비디오 압축**: 움직임 보상
- **객체 추적**: 움직임 기반 추적
- **자율 주행**: 장애물 감지 및 회피
- **제스처 인식**: 동작 인식
- **3D 재구성**: 동적 장면 재구성

## 요약

이 강의에서는 다음과 같은 내용을 다뤘습니다:

1. **광학 흐름**: 2D 이미지에서의 픽셀 움직임
2. **Lucas-Kanade & Horn-Schunck**: 전통적 광학 흐름 방법
3. **다중 스케일 방법**: 피라미드 기반 처리
4. **장면 흐름**: 3D 공간에서의 움직임
5. **딥러닝 방법**: FlowNet 등

광학 흐름과 장면 흐름은 동적 장면 분석의 핵심 기술입니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [09-optical-flow.pdf](https://web.stanford.edu/class/cs231a/course_notes/09-optical-flow.pdf)
