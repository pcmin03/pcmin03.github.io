---
title: "[CS231A] Lecture 08: Monocular Depth Estimation (단안 깊이 추정)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Depth Estimation, Monocular Vision, Deep Learning]
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

이 포스트는 Stanford CS231A 강의의 여덟 번째 강의 노트인 "Monocular Depth Estimation"를 정리한 것입니다.

**원본 강의 노트**: [08-monocular_depth_estimation.pdf](https://web.stanford.edu/class/cs231a/course_notes/08-monocular_depth_estimation.pdf)

<!--more-->

## 1. 서론 (Introduction)

단안 깊이 추정(Monocular Depth Estimation)은 **단일 이미지로부터 깊이 정보를 추정하는 기술**입니다. 스테레오 비전과 달리 하나의 카메라만으로 깊이를 추정하므로 더 실용적이지만, 더 어려운 문제입니다.

## 2. 배경 (Background)

### 2.1. 단안 깊이 추정의 도전 과제

단일 이미지에서 깊이를 추정하는 것은 본질적으로 **ill-posed 문제**입니다. 같은 이미지가 여러 다른 3D 장면에서 나올 수 있기 때문입니다.

![Monocular Depth Estimation Overview](/assets/images/posts/cs231a-08/figures/page_2_img_1.png)

### 2.2. 깊이 단서 (Depth Cues)

단안 깊이 추정은 다음과 같은 단서를 활용합니다:
- **기하학적 단서**: 선형 원근법, 상대적 크기
- **광학적 단서**: 대기 원근법, 그림자
- **텍스처 단서**: 텍스처 그라데이션
- **맥락 단서**: 객체 크기, 장면 이해

## 3. 지도 학습 기반 추정 (Supervised Estimation)

### 3.1. 딥러닝 기반 방법

깊이 맵을 직접 예측하는 딥러닝 모델을 학습합니다:
- **Encoder-Decoder 구조**: 이미지를 인코딩하고 깊이 맵으로 디코딩
- **Multi-scale Features**: 다양한 스케일의 특징 활용
- **Loss Functions**: L1, L2, Scale-invariant loss 등

![Supervised Depth Estimation](/assets/images/posts/cs231a-08/figures/page_3_img_1.png)
![Depth Estimation Results](/assets/images/posts/cs231a-08/figures/page_3_img_2.png)
![Network Architecture](/assets/images/posts/cs231a-08/figures/page_4_img_1.png)

### 3.2. 데이터셋

- **NYU Depth Dataset**: 실내 장면 깊이 데이터
- **KITTI**: 자율 주행 깊이 데이터
- **Make3D**: 야외 장면 깊이 데이터

## 4. 비지도 학습 기반 추정 (Unsupervised Estimation)

### 4.1. 자기 지도 학습 (Self-Supervised Learning)

깊이 레이블 없이 학습하는 방법입니다:
- **Stereo Pairs**: 스테레오 이미지 쌍을 이용한 재구성 손실
- **Monocular Video**: 연속 프레임 간의 일관성 이용
- **Photometric Loss**: 이미지 재구성 오차 최소화

### 4.2. 구조화된 손실 (Structured Loss)

- **Photometric Consistency**: 같은 3D 점이 다른 뷰에서 같은 색상을 가져야 함
- **Smoothness Loss**: 깊이 맵의 부드러움 보장
- **Edge-aware Loss**: 객체 경계에서의 깊이 불연속성 허용

## 5. 깊이 추정 네트워크 구조

### 5.1. U-Net 구조

인코더-디코더 구조에 skip connection을 추가하여 세부 정보를 보존합니다.

### 5.2. Multi-scale Prediction

여러 스케일에서 깊이를 예측하고 융합합니다.

### 5.3. Attention Mechanisms

중요한 영역에 집중하여 더 정확한 깊이 추정을 수행합니다.

## 6. 평가 지표 (Evaluation Metrics)

- **RMSE (Root Mean Squared Error)**: 평균 제곱근 오차
- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **Threshold Accuracy**: 특정 임계값 내 정확도
- **Scale-invariant Metrics**: 스케일 불변 평가

## 7. 응용 분야 (Applications)

단안 깊이 추정은 다음과 같은 분야에서 활용됩니다:

- **자율 주행**: 거리 추정 및 장애물 회피
- **증강 현실**: 3D 객체 배치 및 오클루전 처리
- **로봇 공학**: 경로 계획 및 조작
- **3D 재구성**: 단일 이미지로부터 3D 모델 생성

## 요약

이 강의에서는 다음과 같은 내용을 다뤘습니다:

1. **단안 깊이 추정의 도전 과제**: ill-posed 문제와 깊이 단서
2. **지도 학습 방법**: 깊이 레이블을 이용한 학습
3. **비지도 학습 방법**: 자기 지도 학습을 통한 깊이 추정
4. **네트워크 구조**: U-Net, Multi-scale 등

단안 깊이 추정은 실용적인 3D 비전 기술로, 스테레오 비전의 대안을 제공합니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [08-monocular_depth_estimation.pdf](https://web.stanford.edu/class/cs231a/course_notes/08-monocular_depth_estimation.pdf)
