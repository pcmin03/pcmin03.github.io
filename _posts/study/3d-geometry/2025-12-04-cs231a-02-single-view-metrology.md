---
title: "[CS231A] Lecture 02: Single View Metrology (단일 뷰 측정학)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Single View, Metrology]
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

이 포스트는 Stanford CS231A 강의의 02번째 강의 노트인 "Single View Metrology"를 한글로 정리한 것입니다.

**원본 강의 노트**: [02-single-view-metrology.pdf](https://web.stanford.edu/class/cs231a/course_notes/02-single-view-metrology.pdf)

<!--more-->

## 강의 개요

이 강의에서는 단일 이미지로부터 3D 정보를 추출하는 단일 뷰 측정학(Single View Metrology)에 대해 다룹니다. 한 장의 이미지만으로도 카메라 파라미터와 3D 구조를 복원하는 방법을 학습합니다.

## 강의 노트 페이지 이미지

<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
  <div><img src='/assets/images/posts/cs231a-02/page_1.png' alt='Page 1' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_2.png' alt='Page 2' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_3.png' alt='Page 3' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_4.png' alt='Page 4' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_5.png' alt='Page 5' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_6.png' alt='Page 6' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_7.png' alt='Page 7' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_8.png' alt='Page 8' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_9.png' alt='Page 9' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_10.png' alt='Page 10' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_11.png' alt='Page 11' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_12.png' alt='Page 12' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_13.png' alt='Page 13' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-02/page_14.png' alt='Page 14' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
</div>

## 주요 수식

이 강의에서 다루는 주요 수식들:

**수식 1: Vanishing Point (소실점)**

소실점은 평행한 직선들이 이미지에서 만나는 점입니다. 평행한 3D 직선들이 투영되어 만나는 점으로, 무한대에 있는 점의 투영입니다.

**수식 2: Vanishing Line (소실선)**

평행한 평면들의 교선이 무한대에서 만나는 선을 소실선이라고 합니다.

**수식 3: Cross Ratio (교차비)**

4개의 점이 같은 직선 위에 있을 때, 교차비는 투영 불변량입니다:

$$\text{CR}(A,B,C,D) = \frac{AC \cdot BD}{AD \cdot BC}$$

**수식 4: Homography (호모그래피)**

평면에서 평면으로의 투영 변환:

$$p' = Hp$$

여기서 $H$는 3×3 호모그래피 행렬입니다.

**수식 5: Camera Calibration from Vanishing Points**

소실점을 이용한 카메라 캘리브레이션:

$$v_i^T \omega v_j = 0$$

여기서 $\omega = K^{-T}K^{-1}$는 이미지의 절대 원뿔선(Image of Absolute Conic, IAC)입니다.

## 1. Introduction

단일 뷰 측정학(Single View Metrology)은 한 장의 이미지만으로부터 3D 정보를 추출하는 기법입니다. 이는 카메라 캘리브레이션과 3D 복원을 동시에 수행하는 방법으로, 여러 뷰가 필요한 전통적인 방법과 달리 단일 이미지로도 측정이 가능합니다.

### 1.1 왜 단일 뷰 측정학인가?

- **실용성**: 한 장의 사진만으로도 3D 정보 추출 가능
- **역사적 사진 분석**: 과거 사진에서 3D 구조 복원
- **건축 측량**: 건물의 치수 측정
- **법의학**: 사고 현장 분석

### 1.2 기본 아이디어

이미지에서 기하학적 단서(geometric cues)를 찾아 3D 정보를 복원합니다:
- 평행선의 소실점
- 직각 관계
- 알려진 길이 비율
- 평면 구조

## 2. Vanishing Points (소실점)

### 2.1 소실점의 정의

평행한 3D 직선들이 이미지 평면에 투영될 때 만나는 점을 소실점(vanishing point)이라고 합니다.

### 2.2 소실점 계산

두 개 이상의 평행한 직선 이미지에서 소실점을 계산할 수 있습니다:

1. 이미지에서 평행한 직선들을 찾습니다
2. 각 직선의 방향 벡터를 계산합니다
3. 이들의 교점을 구합니다

### 2.3 소실점의 성질

- 무한대에 있는 점의 투영
- 카메라의 방향 정보를 포함
- 세 개의 서로 직교하는 방향의 소실점으로 카메라 캘리브레이션 가능

## 3. Vanishing Lines (소실선)

### 3.1 소실선의 정의

평행한 평면들의 교선이 무한대에서 만나는 선을 소실선(vanishing line)이라고 합니다.

### 3.2 소실선 계산

평면 위의 평행한 직선들의 소실점들을 연결하면 소실선을 얻을 수 있습니다.

### 3.3 소실선의 활용

- 평면의 방향 결정
- 평면 간의 각도 계산
- 평면 위의 측정

## 4. Cross Ratio (교차비)

### 4.1 교차비의 정의

4개의 점 $A, B, C, D$가 같은 직선 위에 있을 때, 교차비는:

$$\text{CR}(A,B,C,D) = \frac{AC \cdot BD}{AD \cdot BC}$$

### 4.2 교차비의 불변성

교차비는 투영 변환에 대해 불변입니다. 즉, 3D 공간에서의 교차비와 이미지에서의 교차비가 같습니다.

### 4.3 교차비의 활용

- 길이 비율 측정
- 알려진 비율을 이용한 3D 복원
- 평면 위의 측정

## 5. Homography (호모그래피)

### 5.1 호모그래피의 정의

평면에서 평면으로의 투영 변환을 호모그래피(homography)라고 합니다:

$$p' = Hp$$

여기서 $H$는 3×3 행렬이고, $p$와 $p'$는 동차 좌표입니다.

### 5.2 호모그래피 계산

4개의 대응점 쌍으로 호모그래피를 계산할 수 있습니다. 각 대응점 쌍은 2개의 방정식을 제공하므로, 4쌍이면 8개의 방정식으로 8개의 자유도를 결정할 수 있습니다.

### 5.3 호모그래피의 활용

- 평면 위의 측정
- 이미지 정렬
- 평면 투영

## 6. Camera Calibration from Vanishing Points

### 6.1 기본 원리

서로 직교하는 세 방향의 소실점을 이용하여 카메라 내부 파라미터를 추정할 수 있습니다.

### 6.2 IAC (Image of Absolute Conic)

절대 원뿔선의 이미지(Image of Absolute Conic, IAC)는:

$$\omega = K^{-T}K^{-1}$$

여기서 $K$는 카메라 내부 파라미터 행렬입니다.

### 6.3 직교 제약 조건

서로 직교하는 두 방향의 소실점 $v_i$와 $v_j$에 대해:

$$v_i^T \omega v_j = 0$$

이 제약 조건을 이용하여 $\omega$를 추정하고, 이를 통해 $K$를 복원할 수 있습니다.

## 7. 3D Reconstruction from Single View

### 7.1 기본 절차

1. **소실점 추출**: 이미지에서 평행선을 찾아 소실점 계산
2. **카메라 캘리브레이션**: 소실점을 이용한 내부 파라미터 추정
3. **평면 복원**: 호모그래피를 이용한 평면 복원
4. **3D 측정**: 교차비나 알려진 길이를 이용한 측정

### 7.2 측정 예제

이미지에서:
- 알려진 길이를 이용한 스케일 결정
- 교차비를 이용한 비율 측정
- 평면 호모그래피를 이용한 평면 위 측정

## 8. Applications

### 8.1 건축 측량

- 건물의 높이 측정
- 벽면의 치수 측정
- 건물 간 거리 측정

### 8.2 법의학

- 사고 현장 분석
- 증거물의 위치 및 크기 측정
- 재현 시뮬레이션

### 8.3 역사적 사진 분석

- 과거 건물의 3D 복원
- 역사적 사건의 공간적 재구성

## 요약

이 강의에서는 단일 이미지로부터 3D 정보를 추출하는 단일 뷰 측정학에 대해 학습했습니다:

1. **소실점과 소실선**: 평행한 직선과 평면의 투영 특성
2. **교차비**: 투영 불변량을 이용한 측정
3. **호모그래피**: 평면 간의 투영 변환
4. **카메라 캘리브레이션**: 소실점을 이용한 내부 파라미터 추정
5. **3D 복원**: 단일 이미지로부터 3D 구조 복원

이러한 기법들은 한 장의 이미지만으로도 3D 측정과 복원이 가능하게 해주는 강력한 도구입니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [02-single-view-metrology.pdf](https://web.stanford.edu/class/cs231a/course_notes/02-single-view-metrology.pdf)
