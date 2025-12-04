---
title: "[CS231A] Lecture 10: Optimal Estimation (최적 추정)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Kalman Filter, State Estimation, Bayesian Filtering]
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

이 포스트는 Stanford CS231A 강의의 열 번째 강의 노트인 "Optimal Estimation"를 정리한 것입니다.

**원본 강의 노트**: [10-optimal-estimation.pdf](https://web.stanford.edu/class/cs231a/course_notes/10-optimal-estimation.pdf)

<!--more-->

## 1. 서론 (Introduction)

최적 추정(Optimal Estimation)은 **노이즈가 있는 관측으로부터 시스템의 상태를 추정하는 기술**입니다. 로봇 공학, 자율 주행, 추적 등에서 핵심적인 역할을 합니다.

## 2. 상태 추정 (State Estimation)

### 2.1. 상태 공간 모델

시스템의 상태를 다음과 같이 모델링합니다:

$$x_t = f(x_{t-1}, u_t, w_t)$$

$$z_t = h(x_t, v_t)$$

여기서:
- $x_t$: 시간 $t$에서의 상태
- $u_t$: 제어 입력
- $z_t$: 관측
- $w_t, v_t$: 프로세스 및 관측 노이즈

### 2.2. 추정 문제

과거 관측 $z_{1:t}$로부터 현재 상태 $x_t$를 추정하는 문제입니다.

## 3. 베이지안 필터 (Bayesian Filter)

### 3.1. 조건부 확률 복습

베이지안 필터는 조건부 확률을 기반으로 합니다:
- **사전 확률 (Prior)**: $P(x_t | z_{1:t-1})$
- **사후 확률 (Posterior)**: $P(x_t | z_{1:t})$
- **우도 (Likelihood)**: $P(z_t | x_t)$

### 3.2. 베이지안 필터 유도

베이즈 정리를 사용하여 사후 확률을 계산합니다:

$$P(x_t | z_{1:t}) = \frac{P(z_t | x_t) P(x_t | z_{1:t-1})}{P(z_t | z_{1:t-1})}$$

### 3.3. 예측 단계 (Prediction)

이전 상태로부터 현재 상태를 예측합니다:

$$P(x_t | z_{1:t-1}) = \int P(x_t | x_{t-1}) P(x_{t-1} | z_{1:t-1}) dx_{t-1}$$

### 3.4. 업데이트 단계 (Update)

관측을 통하여 상태를 업데이트합니다:

$$P(x_t | z_{1:t}) \propto P(z_t | x_t) P(x_t | z_{1:t-1})$$

## 4. 칼만 필터 (Kalman Filter)

### 4.1. 선형 칼만 필터

선형 시스템과 가우시안 노이즈를 가정합니다:

$$x_t = A x_{t-1} + B u_t + w_t$$

$$z_t = C x_t + v_t$$

여기서 $w_t \sim \mathcal{N}(0, Q)$, $v_t \sim \mathcal{N}(0, R)$입니다.

### 4.2. 칼만 필터 알고리즘

1. **예측 (Predict)**:
   - 상태 예측: $\hat{x}_{t|t-1} = A \hat{x}_{t-1|t-1} + B u_t$
   - 공분산 예측: $P_{t|t-1} = A P_{t-1|t-1} A^T + Q$

2. **업데이트 (Update)**:
   - 칼만 gain: $K_t = P_{t|t-1} C^T (C P_{t|t-1} C^T + R)^{-1}$
   - 상태 업데이트: $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (z_t - C \hat{x}_{t|t-1})$
   - 공분산 업데이트: $P_{t|t} = (I - K_t C) P_{t|t-1}$

![Kalman Filter Overview](/assets/images/posts/cs231a-10/figures/page_1_img_1.png)
![Kalman Filter Algorithm](/assets/images/posts/cs231a-10/figures/page_3_img_1.png)

### 4.3. 확장 칼만 필터 (Extended Kalman Filter, EKF)

비선형 시스템을 선형화하여 칼만 필터를 적용합니다.

## 5. 파티클 필터 (Particle Filter)

### 5.1. 파티클 필터의 개념

비선형, 비가우시안 시스템을 처리하기 위한 몬테카를로 방법입니다.

### 5.2. 파티클 필터 알고리즘

1. **샘플링**: 사전 분포에서 파티클 샘플링
2. **가중치 계산**: 관측에 기반하여 가중치 계산
3. **리샘플링**: 가중치에 따라 파티클 재샘플링

![Particle Filter](/assets/images/posts/cs231a-10/figures/page_5_img_1.png)
![State Estimation Applications](/assets/images/posts/cs231a-10/figures/page_5_img_2.png)

## 6. 응용 분야 (Applications)

최적 추정은 다음과 같은 분야에서 활용됩니다:

- **로봇 공학**: 위치 추정 및 추적
- **자율 주행**: 차량 위치 및 속도 추정
- **객체 추적**: 비디오에서 객체 추적
- **센서 융합**: 여러 센서 데이터 통합
- **SLAM (Simultaneous Localization and Mapping)**: 동시 위치 추정 및 맵핑

## 요약

이 강의에서는 다음과 같은 내용을 다뤘습니다:

1. **상태 추정**: 노이즈가 있는 관측으로부터 상태 추정
2. **베이지안 필터**: 확률론적 상태 추정 프레임워크
3. **칼만 필터**: 선형 시스템을 위한 최적 필터
4. **파티클 필터**: 비선형 시스템을 위한 몬테카를로 방법

최적 추정은 불확실성이 있는 환경에서 시스템 상태를 추정하는 핵심 기술입니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [10-optimal-estimation.pdf](https://web.stanford.edu/class/cs231a/course_notes/10-optimal-estimation.pdf)
