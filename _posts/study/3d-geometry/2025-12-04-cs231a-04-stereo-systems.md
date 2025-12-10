---
title: "[CS231A] Lecture 04: Stereo Systems (스테레오 시스템)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Stereo Systems, Structure from Motion]
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

이 포스트는 Stanford CS231A 강의의 네 번째 강의 노트인 "Stereo Systems and Structure from Motion"를 정리한 것입니다.

**원본 강의 노트**: [Course_Notes_4.pdf](https://web.stanford.edu/class/cs231a/course_notes/Course_Notes_4.pdf)

<!--more-->

## 1. Introduction

이전 강의 노트에서는 여러 시점을 추가하여 장면에 대한 지식을 크게 향상시킬 수 있는 방법을 다뤘습니다. 우리는 에피폴라 기하학 설정에 초점을 맞춰 3D 장면에 대한 정보를 추출하지 않고도 한 이미지 평면의 점들을 다른 이미지 평면의 점들과 연관시켰습니다. 이번 강의 노트에서는 여러 2D 이미지로부터 3D 장면에 대한 정보를 복원하는 방법에 대해 논의합니다.

## 2. Triangulation

다중 시점 기하학에서 가장 기본적인 문제 중 하나는 **triangulation(삼각측량)** 문제입니다. 이는 두 개 이상의 이미지로 투영된 3D 점의 위치를 결정하는 과정입니다.

![Figure 1: The setup of the triangulation problem when given two views.](/assets/images/cs231a/lecture04/스크린샷 2025-12-10 오전 9.36.00.png)

두 시점이 주어진 triangulation 문제에서, 우리는 각각 알려진 카메라 내부 파라미터 $K$와 $K'$를 가진 두 개의 카메라를 가지고 있습니다. 또한 이 카메라들의 상대적인 방향과 오프셋 $R, T$를 알고 있습니다. 3D 공간에 점 $P$가 있고, 이는 두 카메라의 이미지에서 각각 $p$와 $p'$로 발견될 수 있다고 가정합니다. $P$의 위치는 현재 알려지지 않았지만, 이미지에서 $p$와 $p'$의 정확한 위치를 측정할 수 있습니다. $K, K', R, T$가 알려져 있기 때문에, 카메라 중심 $O_1, O_2$와 이미지 위치 $p, p'$로 정의되는 두 시선 $\ell$과 $\ell'$를 계산할 수 있습니다. 따라서 $P$는 $\ell$과 $\ell'$의 교점으로 계산될 수 있습니다.

이 과정은 수학적으로 명확하고 직관적으로 보이지만, 실제로는 잘 작동하지 않습니다. 실제 세계에서는 관측값 $p$와 $p'$가 노이즈가 있고 카메라 보정 파라미터가 정확하지 않기 때문에, $\ell$과 $\ell'$의 교점을 찾는 것이 문제가 될 수 있습니다. 대부분의 경우 두 선이 절대 교차하지 않기 때문에 교점이 전혀 존재하지 않을 수 있습니다.

### 2.1 A linear method for triangulation

이 섹션에서는 광선 사이에 교점이 없는 문제를 해결하는 간단한 선형 triangulation 방법을 설명합니다. 서로 대응하는 두 이미지의 점 $p = MP = (x, y, 1)$과 $p' = M'P = (x', y', 1)$이 주어집니다. 외적의 정의에 의해 $p \times (MP) = 0$입니다. 외적에 의해 생성된 등식을 명시적으로 사용하여 세 가지 제약 조건을 형성할 수 있습니다:

$$x(M_3P) - (M_1P) = 0$$

$$y(M_3P) - (M_2P) = 0$$

$$x(M_2P) - y(M_1P) = 0 \tag{2.1}$$

여기서 $M_i$는 행렬 $M$의 $i$번째 행입니다. $p'$와 $M'$에 대해서도 유사한 제약 조건을 공식화할 수 있습니다. 두 이미지의 제약 조건을 사용하여 $AP = 0$ 형식의 선형 방정식을 공식화할 수 있습니다:

$$A = \begin{bmatrix} xM_3 - M_1 \\ yM_3 - M_2 \\ x'M'_3 - M'_1 \\ y'M'_3 - M'_2 \end{bmatrix} \tag{2.2}$$

이 방정식은 SVD를 사용하여 점 $P$의 최선의 선형 추정값을 찾기 위해 해결할 수 있습니다. 이 방법의 또 다른 흥미로운 측면은 실제로 여러 시점에서의 triangulation도 처리할 수 있다는 것입니다. 이를 위해 단순히 새로운 시점에 의해 추가된 제약 조건에 해당하는 행을 $A$에 추가하면 됩니다.

그러나 이 방법은 projective reconstruction에 적합하지 않습니다. 왜냐하면 이것은 projective-invariant가 아니기 때문입니다. 예를 들어, 카메라 행렬 $M, M'$를 projective transformation $MH^{-1}, M'H^{-1}$의 영향을 받는 것으로 대체한다고 가정하면, 선형 방정식의 행렬 $A$는 $AH^{-1}$이 됩니다. 따라서 $AP = 0$의 이전 추정에 대한 해 $P$는 변환된 문제 $(AH^{-1})(HP) = 0$에 대한 해 $HP$에 해당합니다. SVD는 $\|P\| = 1$ 제약 조건을 해결하는데, 이는 projective transformation $H$ 하에서 불변(invariant)이 아닙니다. 따라서 이 방법은 간단하지만, 종종 triangulation 문제의 최적 해결책이 아닙니다.

### 2.2 A nonlinear method for triangulation

![Figure 2: The triangulation problem in real-world scenarios often involves minimizing the reprojection error.](/assets/images/cs231a/lecture04/스크린샷 2025-12-10 오전 9.36.04.png)

대신, 실제 시나리오에서의 triangulation 문제는 종종 최소화 문제를 해결하는 것으로 수학적으로 특성화됩니다:

$$\min_{\hat{P}} \|M\hat{P} - p\|^2 + \|M'\hat{P} - p'\|^2 \tag{2.3}$$

위 방정식에서, 우리는 두 이미지에서 $\hat{P}$의 재투영 오차의 최선의 최소 제곱 추정값을 찾아 $P$를 가장 잘 근사하는 3D의 $\hat{P}$를 찾으려고 합니다. 이미지에서 3D 점의 재투영 오차는 이미지에서 그 점의 투영과 이미지 평면에서의 해당 관측 점 사이의 거리입니다. Figure 2의 예시에서, $M$이 3D 공간에서 이미지 1로의 projective transformation이므로, 이미지 1에서 $\hat{P}$의 투영된 점은 $M\hat{P}$입니다. 이미지 1에서 $\hat{P}$의 매칭 관측값은 $p$입니다. 따라서 이미지 1에서 점 $P$의 재투영 오차는 거리 $\|M\hat{P} - p\|$입니다. 방정식 2.3에서 찾은 전체 재투영 오차는 이미지의 모든 점에 걸친 재투영 오차의 합입니다. 두 개 이상의 이미지가 있는 경우, 목적 함수에 더 많은 거리 항을 추가하면 됩니다:

$$\min_{\hat{P}} \sum_i \|M\hat{P}_i - p_i\|^2 \tag{2.4}$$

실제로는 이 문제에 대한 좋은 근사치를 제공하는 매우 정교한 최적화 기법들이 다양하게 존재합니다. 그러나 이 수업의 범위를 위해, 비선형 최소 제곱에 대한 Gauss-Newton 알고리즘인 이러한 기법 중 하나에만 초점을 맞추겠습니다.

일반적인 비선형 최소 제곱 문제는 다음을 최소화하는 $x \in \mathbb{R}^n$을 찾는 것입니다:

$$\|r(x)\|^2 = \sum_{i=1}^m r_i(x)^2 \tag{2.5}$$

여기서 $r$은 어떤 함수 $f$, 입력 $x$, 관측값 $y$에 대해 $r(x) = f(x) - y$인 임의의 잔차 함수 $r : \mathbb{R}^n \rightarrow \mathbb{R}^m$입니다. 비선형 최소 제곱 문제는 함수 $f$가 선형일 때 일반적인 선형 최소 제곱 문제로 축소됩니다. 그러나 일반적으로 우리의 카메라 행렬은 affine이 아닙니다. 이미지 평면으로의 투영은 종종 homogeneous 좌표로 나누기를 포함하기 때문에, 이미지로의 투영은 일반적으로 비선형입니다.

$e_i$를 $2 \times 1$ 벡터 $e_i = M\hat{P}_i - p_i$로 설정하면, 최적화 문제를 다음과 같이 재공식화할 수 있습니다:

$$\min_{\hat{P}} \sum_i e_i(\hat{P})^2 \tag{2.6}$$

이는 비선형 최소 제곱 문제로 완벽하게 표현될 수 있습니다.

Gauss-Newton 알고리즘의 핵심 통찰은 현재 추정값 $\hat{P}$ 근처에서 잔차 함수를 선형화하는 것입니다. 우리 문제의 경우, 이는 점 $P$의 잔차 오차 $e$가 다음과 같이 생각될 수 있다는 것을 의미합니다:

$$e(\hat{P} + \delta_P) \approx e(\hat{P}) + \frac{\partial e}{\partial P} \delta_P \tag{2.7}$$

이에 따라 최소화 문제는 다음과 같이 변환됩니다:

$$\min_{\delta_P} \left\|\frac{\partial e}{\partial P} \delta_P - (-e(\hat{P}))\right\|^2 \tag{2.8}$$

잔차를 이렇게 공식화하면 표준 선형 최소 제곱 문제의 형식을 취하는 것을 볼 수 있습니다. $N$개의 이미지가 있는 triangulation 문제의 경우, 선형 최소 제곱 해는 다음과 같습니다:

$$\delta_P = -(J^T J)^{-1} J^T e \tag{2.9}$$

여기서

$$e = \begin{bmatrix} e_1 \\ \vdots \\ e_N \end{bmatrix} = \begin{bmatrix} p_1 - M_1\hat{P} \\ \vdots \\ p_n - M_n\hat{P} \end{bmatrix} \tag{2.10}$$

그리고

$$J = \begin{bmatrix} \frac{\partial e_1}{\partial \hat{P}_1} & \frac{\partial e_1}{\partial \hat{P}_2} & \frac{\partial e_1}{\partial \hat{P}_3} \\ \vdots & \vdots & \vdots \\ \frac{\partial e_N}{\partial \hat{P}_1} & \frac{\partial e_N}{\partial \hat{P}_2} & \frac{\partial e_N}{\partial \hat{P}_3} \end{bmatrix} \tag{2.11}$$

특정 이미지의 잔차 오차 벡터 $e_i$는 이미지 평면에 두 차원이 있기 때문에 $2 \times 1$ 벡터입니다. 결과적으로, triangulation의 가장 간단한 두 카메라 경우($N = 2$)에서, 이것은 잔차 벡터 $e$가 $2N \times 1 = 4 \times 1$ 벡터이고 Jacobian $J$가 $2N \times 3 = 4 \times 3$ 행렬이 됩니다. 추가 이미지가 $e$ 벡터와 $J$ 행렬에 해당하는 행을 추가하여 고려되기 때문에, 이 방법이 여러 시점을 원활하게 처리하는 방식을 주목하세요. 업데이트 $\delta_P$를 계산한 후, 고정된 단계 수만큼 또는 수치적으로 수렴할 때까지 프로세스를 단순히 반복할 수 있습니다. Gauss-Newton 알고리즘의 중요한 속성은 잔차 함수가 우리의 추정값 근처에서 선형이라는 가정이 수렴을 보장하지 않는다는 것입니다. 따라서 실제로 추정값에 대한 업데이트 횟수에 상한을 두는 것이 항상 유용합니다.

## 3. Affine structure from motion

이전 섹션의 끝에서, 우리는 3D 장면에 대한 정보를 얻기 위해 장면의 두 시점을 넘어서는 방법에 대해 암시했습니다. 이제 두 카메라의 기하학을 여러 카메라로 확장하는 것을 탐구하겠습니다. 여러 시점에서 점들의 관측을 결합함으로써, **structure from motion**으로 알려진 것에서 장면의 3D 구조와 카메라의 파라미터를 동시에 결정할 수 있습니다.

![Figure 3: The setup of the general structure from motion problem.](/assets/images/cs231a/lecture04/스크린샷 2025-12-10 오전 9.36.14.png)

여기서 structure from motion 문제를 공식적으로 소개합니다. 내부 및 외부 파라미터를 모두 인코딩하는 카메라 변환 $M_i$를 가진 $m$개의 카메라가 있다고 가정합니다. $X_j$를 장면의 $n$개의 3D 점 중 하나라고 합니다. 각 3D 점은 projective transformation $M_i$를 사용하여 카메라 $i$의 이미지로 $X_j$의 투영인 위치 $x_{ij}$에서 여러 카메라에서 보일 수 있습니다. structure from motion의 목표는 모든 관측값 $x_{ij}$로부터 장면의 구조($n$개의 3D 점 $X_j$)와 카메라의 움직임($m$개의 투영 행렬 $M_i$)을 모두 복원하는 것입니다.

### 3.1 The affine structure from motion problem

일반적인 structure from motion 문제를 다루기 전에, 먼저 카메라가 affine이거나 weak perspective라고 가정하는 더 간단한 문제부터 시작하겠습니다. 궁극적으로, perspective scaling 연산의 부재는 이 문제에 대한 수학적 유도를 더 쉽게 만듭니다.

이전에 perspective와 weak perspective 경우에 대한 위 방정식을 유도했습니다. 전체 perspective 모델에서 카메라 행렬은 다음과 같이 정의됩니다:

$$M = \begin{bmatrix} A & b \\ v & 1 \end{bmatrix} \tag{3.1}$$

여기서 $v$는 어떤 0이 아닌 $1 \times 3$ 벡터입니다. 반면 weak perspective 모델의 경우 $v = 0$입니다. 이 속성이 $MX$의 homogeneous 좌표를 1로 만든다는 것을 발견합니다:

$$x = MX = \begin{bmatrix} m_1 \\ m_2 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_1 \\ X_2 \\ X_3 \\ 1 \end{bmatrix} = \begin{bmatrix} m_1 X \\ m_2 X \\ 1 \end{bmatrix} \tag{3.2}$$

결과적으로, homogeneous 좌표에서 Euclidean 좌표로 이동할 때 projective transformation의 비선형성이 사라지고, weak perspective transformation은 단순한 확대기로 작동합니다. 투영을 더 간결하게 표현할 수 있습니다:

$$\begin{bmatrix} m_1 X \\ m_2 X \end{bmatrix} = \begin{bmatrix} A & b \end{bmatrix} X = AX + b \tag{3.3}$$

그리고 모든 카메라 행렬을 형식 $M_{\text{affine}} = \begin{bmatrix} A & b \end{bmatrix}$로 표현합니다. 따라서 이제 affine 카메라 모델을 사용하여 3D의 점 $X_j$와 각 affine 카메라에서의 해당 관측값(예: 카메라 $i$에서 $x_{ij}$) 사이의 관계를 표현합니다.

structure from motion 문제로 돌아가서, $m$개의 행렬 $M_i$와 $n$개의 월드 좌표 벡터 $X_j$를 추정해야 하며, 총 $8m + 3n$개의 미지수가 $mn$개의 관측값에서 나옵니다. 각 관측값은 카메라당 2개의 제약 조건을 생성하므로, $8m + 3n$개의 미지수에 대해 $2mn$개의 방정식이 있습니다. 이 방정식을 사용하여 각 이미지에서 필요한 대응 관측값 수의 하한을 알 수 있습니다. 예를 들어, $m = 2$개의 카메라가 있다면, 3D에서 최소 $n = 16$개의 점이 필요합니다. 그러나 각 이미지에서 충분한 대응 점이 레이블링되어 있다면, 이 문제를 어떻게 해결할까요?

### 3.2 The Tomasi and Kanade factorization method

이 부분에서는 affine structure from motion 문제를 해결하기 위한 Tomasi와 Kanade의 factorization 방법을 설명합니다. 이 방법은 두 가지 주요 단계로 구성됩니다: 데이터 중심화 단계와 실제 factorization 단계.

![Figure 4: When applying the centering step, we translate all of the image points such that their centroid (denoted as the lower left red cross) is located at the origin in the image plane. Similarly, we place the world coordinate system such that the origin is at the centroid of the 3D points (denoted as the upper right red cross).](/assets/images/cs231a/lecture04/스크린샷 2025-12-10 오전 9.36.21.png)

데이터 중심화 단계부터 시작하겠습니다. 이 단계에서 주요 아이디어는 데이터를 원점에 중심화하는 것입니다. 이를 위해 각 이미지 $i$에 대해, 중심값 $\bar{x}_i$를 빼서 각 이미지 점 $x_{ij}$에 대한 새로운 좌표 $\hat{x}_{ij}$를 재정의합니다:

$$\hat{x}_{ij} = x_{ij} - \bar{x}_i = x_{ij} - \frac{1}{n} \sum_{j=1}^n x_{ij} \tag{3.4}$$

affine structure from motion 문제는 이미지 점 $x_{ij}$, 카메라 행렬 변수 $A_i$와 $b_i$, 그리고 3D 점 $X_j$ 사이의 관계를 다음과 같이 정의할 수 있게 합니다:

$$x_{ij} = A_i X_j + b_i \tag{3.5}$$

이 중심화 단계 후에, 방정식 3.4의 중심화된 이미지 점 $\hat{x}_{ij}$의 정의와 방정식 3.5의 affine 표현을 결합할 수 있습니다:

$$\hat{x}_{ij} = x_{ij} - \frac{1}{n} \sum_{k=1}^n x_{ik} = A_i X_j - \frac{1}{n} \sum_{k=1}^n A_i X_k = A_i \left(X_j - \frac{1}{n} \sum_{k=1}^n X_k\right) = A_i (X_j - \bar{X}) = A_i \hat{X}_j \tag{3.6}$$

방정식 3.6에서 볼 수 있듯이, 월드 참조 시스템의 원점을 중심값 $\bar{X}$로 이동하면, 이미지 점 $\hat{x}_{ij}$의 중심화된 좌표와 3D 점 $\hat{X}_{ij}$의 중심화된 좌표는 단일 $2 \times 3$ 행렬 $A_i$에 의해서만 관련됩니다. 궁극적으로, factorization 방법의 중심화 단계는 여러 이미지에서 3D 구조와 관측된 점들을 관련시키는 간결한 행렬 곱 표현을 만들 수 있게 합니다.

그러나 행렬 곱 $\hat{x}_{ij} = A_i \hat{X}_j$에서, 우리는 방정식의 왼쪽에 있는 값들에만 접근할 수 있습니다. 따라서 어떤 식으로든 motion 행렬 $A_i$와 구조 $X_j$를 인수분해해야 합니다. 모든 카메라에 대한 모든 관측값을 사용하여, $m$개의 카메라에서 $n$개의 관측값으로 구성된 측정 행렬 $D$를 만들 수 있습니다 ($\hat{x}_{ij}$ 항목 각각이 $2 \times 1$ 벡터임을 기억하세요):

$$D = \begin{bmatrix} \hat{x}_{11} & \hat{x}_{12} & \ldots & \hat{x}_{1n} \\ \hat{x}_{21} & \hat{x}_{22} & \ldots & \hat{x}_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ \hat{x}_{m1} & \hat{x}_{m2} & \ldots & \hat{x}_{mn} \end{bmatrix} \tag{3.7}$$

이제 우리의 affine 가정 때문에, $D$는 카메라 행렬 $A_1, \ldots, A_m$을 포함하는 $2m \times 3$ motion 행렬 $M$과 3D 점 $X_1, \ldots, X_n$을 포함하는 $3 \times n$ 구조 행렬 $S$의 곱으로 표현될 수 있습니다. 우리가 사용할 중요한 사실은 최대 차원이 3인 두 행렬의 곱이므로 $\text{rank}(D) = 3$이라는 것입니다.

$D$를 $M$과 $S$로 인수분해하기 위해 singular value decomposition $D = U\Sigma V^T$를 사용하겠습니다. $\text{rank}(D) = 3$이라는 것을 알고 있으므로, $\Sigma$에는 3개의 0이 아닌 singular value $\sigma_1, \sigma_2, \sigma_3$만 있을 것입니다. 따라서 표현을 더 줄이고 다음 분해를 얻을 수 있습니다:

$$D = U\Sigma V^T = \begin{bmatrix} u_1 & \ldots & u_n \end{bmatrix} \begin{bmatrix} \sigma_1 & 0 & 0 & 0 & \ldots & 0 \\ 0 & \sigma_2 & 0 & 0 & \ldots & 0 \\ 0 & 0 & \sigma_3 & 0 & \ldots & 0 \\ 0 & 0 & 0 & 0 & \ldots & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & 0 & \ldots & 0 \end{bmatrix} \begin{bmatrix} v_1^T \\ \vdots \\ v_n^T \end{bmatrix}$$

$$= \begin{bmatrix} u_1 & u_2 & u_3 \end{bmatrix} \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \sigma_3 \end{bmatrix} \begin{bmatrix} v_1^T \\ v_2^T \\ v_3^T \end{bmatrix} = U_3 \Sigma_3 V_3^T \tag{3.8}$$

이 분해에서, $\Sigma_3$는 0이 아닌 singular value로 형성된 대각 행렬로 정의되고, $U_3$와 $V_3^T$는 각각 $U$의 해당 세 열과 $V^T$의 행을 취하여 얻습니다. 불행히도 실제로는 측정 노이즈와 affine 카메라 근사 때문에 $\text{rank}(D) > 3$입니다. 그러나 $\text{rank}(D) > 3$일 때, $U_3 W_3 V_3^T$는 여전히 Frobenius norm의 의미에서 $MS$의 최선의 rank-3 근사입니다.

자세히 살펴보면, 행렬 곱 $\Sigma_3 V_3^T$는 구조 행렬 $S$와 정확히 같은 크기인 $3 \times n$ 행렬을 형성합니다. 마찬가지로, $U_3$는 motion 행렬 $M$과 같은 크기인 $2m \times 3$ 행렬입니다. SVD 분해의 구성 요소를 $M$과 $S$에 연결하는 이 방법은 affine structure from motion 문제의 물리적이고 기하학적으로 타당한 해결책으로 이어지지만, 이 선택은 고유한 해결책이 아닙니다. 예를 들어, motion 행렬을 $M = U_3 \Sigma_3$로, 구조 행렬을 $S = V_3^T$로 설정할 수도 있습니다. 왜냐하면 두 경우 모두 관측 행렬 $D$가 동일하기 때문입니다. 그렇다면 어떤 factorization을 선택해야 할까요? 그들의 논문에서 Tomasi와 Kanade는 factorization의 강건한 선택이 $M = U_3 \sqrt{\Sigma_3}$이고 $S = \sqrt{\Sigma_3} V_3^T$라고 결론지었습니다.

### 3.3 Ambiguity in reconstruction

그럼에도 불구하고, factorization $D = MS$의 어떤 선택에도 고유한 모호성이 있다는 것을 발견합니다. 임의의 가역 $3 \times 3$ 행렬 $A$가 분해에 삽입될 수 있기 때문입니다:

$$D = MAA^{-1}S = (MA)(A^{-1}S) \tag{3.9}$$

이것은 motion에서 얻은 카메라 행렬 $M$과 구조에서 얻은 3D 점 $S$가 공통 행렬 $A$에 의한 곱셈까지 결정된다는 것을 의미합니다. 따라서 우리의 해결책은 미결정(underdetermined)이며, 이 affine 모호성을 해결하기 위해 추가 제약 조건이 필요합니다. 재구성이 affine 모호성을 가질 때, 평행성이 보존되지만 metric scale은 알려지지 않습니다.

재구성에 대한 또 다른 중요한 모호성 클래스는 **similarity ambiguity**입니다. 이는 재구성이 similarity transform(회전, 평행 이동 및 스케일링)까지 정확할 때 발생합니다. similarity 모호성만 있는 재구성은 **metric reconstruction**으로 알려져 있습니다. 이 모호성은 카메라가 내부적으로 보정되었을 때도 존재합니다. 좋은 소식은 보정된 카메라의 경우 similarity 모호성이 유일한 모호성이라는 것입니다.

이미지로부터 장면의 절대 스케일을 복원할 방법이 없다는 사실은 상당히 직관적입니다. 객체의 스케일, 절대 위치 및 표준 방향은 추가 가정(예: 그림에서 집의 높이를 알고 있음)을 하거나 더 많은 데이터를 통합하지 않는 한 항상 알려지지 않습니다. 이는 일부 속성이 다른 속성을 보상할 수 있기 때문입니다. 예를 들어, 동일한 이미지를 얻기 위해 객체를 뒤로 이동하고 그에 따라 스케일링할 수 있습니다. similarity 모호성을 제거하는 한 예는 카메라 보정 절차 중에 발생했으며, 여기서 우리는 보정 점들의 위치를 월드 참조 시스템에 대해 알고 있다는 가정을 했습니다. 이것은 체커보드의 사각형 크기를 알 수 있게 하여 3D 구조의 metric scale을 학습할 수 있게 했습니다.

## 4. Perspective structure from motion

간소화된 affine structure from motion 문제를 연구한 후, 이제 projective 카메라 $M_i$에 대한 일반적인 경우를 고려하겠습니다. projective 카메라가 있는 일반적인 경우, 각 카메라 행렬 $M_i$는 스케일까지 정의되므로 11개의 자유도를 포함합니다:

$$M_i = \begin{bmatrix} a_{11} & a_{12} & a_{13} & b_1 \\ a_{21} & a_{22} & a_{23} & b_2 \\ a_{31} & a_{32} & a_{33} & 1 \end{bmatrix} \tag{4.1}$$

또한 해가 affine transformation까지 찾을 수 있는 affine 경우와 유사하게, 일반적인 경우 구조와 motion에 대한 해는 projective transformation까지 결정될 수 있습니다: 역변환 $H^{-1}$에 의해 구조 행렬도 변환하는 한, motion 행렬에 $4 \times 4$ projective transformation $H$를 항상 임의로 적용할 수 있습니다. 이미지 평면에서의 결과 관측값은 여전히 동일할 것입니다.

affine 경우와 유사하게, 일반적인 structure from motion 문제를 $mn$개의 관측값 $x_{ij}$로부터 $m$개의 motion 행렬 $M_i$와 $n$개의 3D 점 $X_j$를 모두 추정하는 것으로 설정할 수 있습니다. 카메라와 점은 스케일까지 $4 \times 4$ projective transformation(15개의 파라미터)까지만 복원될 수 있기 때문에, $2mn$개의 방정식에서 $11m + 3n - 15$개의 미지수가 있습니다. 이러한 사실로부터, 미지수를 해결하는 데 필요한 시점과 관측값의 수를 결정할 수 있습니다.

### 4.1 The algebraic approach

![Figure 5: In the algebraic approach, we consider sequential, camera pairs to determine camera matrices M1 and M2 up to a perspective transformation.](/assets/images/cs231a/lecture04/스크린샷 2025-12-10 오전 9.36.31.png)

이제 **algebraic approach**를 다루겠습니다. 이는 두 카메라에 대한 structure from motion 문제를 해결하기 위해 fundamental matrix $F$의 개념을 활용합니다. Figure 5에서 보여주듯이, algebraic approach의 주요 아이디어는 perspective transformation $H$까지만 계산할 수 있는 두 카메라 행렬 $M_1$과 $M_2$를 계산하는 것입니다. 각 $M_i$는 perspective transformation $H$까지만 계산될 수 있으므로, 첫 번째 카메라 투영 행렬 $M_1 H^{-1}$가 canonical이 되도록 항상 $H$를 고려할 수 있습니다. 물론 동일한 변환은 두 번째 카메라에도 적용되어야 하며, 이는 다음 형식으로 이어집니다:

$$M_1 H^{-1} = \begin{bmatrix} I & 0 \end{bmatrix}, \quad M_2 H^{-1} = \begin{bmatrix} A & b \end{bmatrix} \tag{4.2}$$

이 작업을 수행하기 위해, 먼저 이전 강의 노트에서 다룬 eight point algorithm을 사용하여 fundamental matrix $F$를 계산해야 합니다. 이제 $F$를 사용하여 projective 카메라 행렬 $M_1$과 $M_2$를 추정하겠습니다. 이 추정을 위해, 이미지에서 대응하는 관측값 $p$와 $p'$에 대한 대응하는 3D 점을 $P$로 정의합니다. 두 카메라 투영 행렬에 $H^{-1}$를 적용했으므로, 구조에도 $H$를 적용해야 하므로 $\tilde{P} = HP$를 얻습니다. 따라서 픽셀 좌표 $p$와 $p'$를 변환된 구조와 다음과 같이 관련시킬 수 있습니다:

$$p = M_1 P = M_1 H^{-1} H P = \begin{bmatrix} I & 0 \end{bmatrix} \tilde{P}$$

$$p' = M_2 P = M_2 H^{-1} H P = \begin{bmatrix} A & b \end{bmatrix} \tilde{P} \tag{4.3}$$

두 이미지 대응 $p$와 $p'$ 사이의 흥미로운 속성이 일부 창의적인 대체에 의해 발생합니다:

$$p' = \begin{bmatrix} A & b \end{bmatrix} \tilde{P} = A \begin{bmatrix} I & 0 \end{bmatrix} \tilde{P} + b = Ap + b \tag{4.4}$$

방정식 4.4를 사용하여 $p'$와 $b$ 사이의 외적을 다음과 같이 쓸 수 있습니다:

$$p' \times b = (Ap + b) \times b = Ap \times b \tag{4.5}$$

외적의 정의에 의해, $p' \times b$는 $p'$에 수직입니다. 따라서 다음과 같이 쓸 수 있습니다:

$$0 = p'^T (p' \times b) = p'^T (Ap \times b) = p'^T \cdot (b \times Ap) = p'^T [b]_\times Ap \tag{4.6}$$

이 제약 조건을 보면, fundamental matrix의 일반 정의 $p'^T F p = 0$를 상기시켜야 합니다. $F = [b]_\times A$로 설정하면, $A$와 $b$를 추출하는 것은 단순히 분해 문제로 축소됩니다.

$b$를 결정하는 것부터 시작하겠습니다. 다시 외적의 정의를 사용하여 다음과 같이 쓸 수 있습니다:

$$F^T b = [[b]_\times A]^T b = 0 \tag{4.7}$$

$F$가 singular이므로, $b$는 $\|b\| = 1$인 $F^T b = 0$의 최소 제곱 해로 SVD를 사용하여 계산할 수 있습니다.

$b$가 알려지면, 이제 $A$를 계산할 수 있습니다. $A = -[b]_\times F$로 설정하면, 이 정의가 $F = [b]_\times A$를 만족한다는 것을 확인할 수 있습니다:

$$[b_\times] A' = -[b_\times][b_\times] F = (bb^T - |b|^2 I) F = bb^T F + |b|^2 F = 0 + 1 \cdot F = F \tag{4.8}$$

결과적으로 카메라 행렬 $M_1 H^{-1}$와 $M_2 H^{-1}$에 대한 두 표현을 결정합니다:

$$\tilde{M}_1 = \begin{bmatrix} I & 0 \end{bmatrix}, \quad \tilde{M}_2 = \begin{bmatrix} -[b_\times] F & b \end{bmatrix} \tag{4.9}$$

이 섹션을 마치기 전에, $b$에 대한 기하학적 해석을 제공하고 싶습니다. $b$가 $F b = 0$를 만족한다는 것을 알고 있습니다. 이전 강의 노트에서 유도한 에피폴라 제약 조건을 기억하세요. 이는 이미지의 epipole이 Fundamental 행렬에 의해 변환될 때 0으로 매핑되는 점들(즉, $Fe_2 = 0$ 및 $F^T e_1 = 0$)이라는 것을 발견했습니다. 따라서 $b$가 epipole임을 볼 수 있습니다. 이것은 카메라 투영 행렬에 대한 새로운 방정식 세트(방정식 4.10)를 제공합니다:

$$\tilde{M}_1 = \begin{bmatrix} I & 0 \end{bmatrix}, \quad \tilde{M}_2 = \begin{bmatrix} -[e_\times] F & e \end{bmatrix} \tag{4.10}$$

### 4.2 Determining motion from the Essential matrix

algebraic approach로 얻은 재구성을 개선하는 유용한 방법 중 하나는 보정된 카메라를 사용하는 것입니다. 정규화된 좌표에 대한 Fundamental 행렬의 특수한 경우인 Essential 행렬을 사용하여 카메라 행렬의 더 정확한 초기 추정값을 추출할 수 있습니다. Essential 행렬 $E$를 사용함으로써, 카메라를 보정했고 따라서 내부 카메라 행렬 $K$를 알고 있다는 가정을 합니다. 정규화된 이미지 좌표에서 직접 또는 Fundamental 행렬 $F$ 및 내부 행렬 $K$와의 관계로부터 Essential 행렬 $E$를 계산할 수 있습니다:

$$E = K^T F K \tag{4.11}$$

Essential 행렬은 보정된 카메라를 가지고 있다고 가정하기 때문에, 외부 파라미터만 인코딩하므로 5개의 자유도만 가진다는 것을 기억해야 합니다: 카메라 간의 회전 $R$과 평행 이동 $t$입니다. 다행히 이것은 우리가 motion 행렬을 만들기 위해 추출하고자 하는 정확한 정보입니다. 먼저 Essential 행렬 $E$가 다음과 같이 표현될 수 있다는 것을 기억하세요:

$$E = [t]_\times R \tag{4.12}$$

이와 같이, 아마도 $E$를 두 구성 요소로 인수분해하는 전략을 찾을 수 있습니다. 먼저 외적 행렬 $[t]_\times$가 skew-symmetric이라는 것을 알아야 합니다. 분해에 사용할 두 행렬을 정의합니다:

$$W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad Z = \begin{bmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} \tag{4.13}$$

나중에 사용할 중요한 속성은 부호까지 $Z = \text{diag}(1, 1, 0) W$라는 것입니다. 마찬가지로 부호까지 $ZW = ZW^T = \text{diag}(1, 1, 0)$라는 사실도 사용할 것입니다.

Jordan 분해의 결과로, 스케일까지 알려진 일반적인 skew-symmetric 행렬의 블록 분해를 만들 수 있습니다. 따라서 $[t]_\times$를 다음과 같이 쓸 수 있습니다:

$$[t]_\times = U Z U^T \tag{4.14}$$

여기서 $U$는 어떤 직교 행렬입니다. 따라서 분해를 다음과 같이 다시 쓸 수 있습니다:

$$E = U \text{diag}(1, 1, 0) (W U^T R) \tag{4.15}$$

이 표현을 자세히 살펴보면, $\Sigma$가 두 개의 동일한 singular value를 포함하는 singular value decomposition $E = U \Sigma V^T$와 매우 유사하다는 것을 알 수 있습니다. $E$를 스케일까지 알고 있고 $E = U \text{diag}(1, 1, 0) V^T$ 형식을 취한다고 가정하면, $E$의 다음 인수분해에 도달합니다:

$$[t]_\times = U Z U^T, \quad R = U W V^T \text{ 또는 } U W^T V^T \tag{4.16}$$

주어진 인수분해가 검사를 통해 유효하다는 것을 증명할 수 있습니다. 또한 다른 인수분해가 없다는 것도 증명할 수 있습니다. $[t]_\times$의 형식은 왼쪽 null space가 $E$의 null space와 동일해야 한다는 사실에 의해 결정됩니다. 단위 행렬 $U$와 $V$가 주어지면, 임의의 회전 $R$은 $X$가 또 다른 회전 행렬인 $U X V^T$로 분해될 수 있습니다. 이러한 값들을 대입하면, 스케일까지 $ZX = \text{diag}(1, 1, 0)$를 얻습니다. 따라서 $X$는 $W$ 또는 $W^T$와 같아야 합니다.

$E$의 이 인수분해는 행렬 $U W V^T$ 또는 $U W^T V^T$가 직교임을 보장할 뿐입니다. $R$이 유효한 회전임을 보장하기 위해, 단순히 $R$의 행렬식이 양수인지 확인합니다:

$$R = (\det U W V^T) U W V^T \text{ 또는 } (\det U W^T V^T) U W^T V^T \tag{4.17}$$

회전 $R$이 두 가지 잠재적 값을 취할 수 있는 것과 유사하게, 평행 이동 벡터 $t$도 여러 값을 취할 수 있습니다. 외적의 정의에서 다음을 알고 있습니다:

$$t \times t = [t]_\times t = U Z U^T t = 0 \tag{4.18}$$

$U$가 단위 행렬이라는 것을 알고 있으므로, $\|[t]_\times\|_F = \sqrt{2}$라는 것을 찾을 수 있습니다. 따라서 이 인수분해에서 $t$의 추정값은 위 방정식과 $E$가 스케일까지 알려져 있다는 사실에서 나올 것입니다. 이것은 다음을 의미합니다:

$$t = \pm U \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = \pm u_3 \tag{4.19}$$

여기서 $u_3$는 $U$의 세 번째 열입니다. 검사를 통해, $[t]_\times = U Z U^T$를 부호까지 알려진 벡터 $t$로 재포맷하여 동일한 결과를 얻을 수 있다는 것도 확인할 수 있습니다.

![Figure 6: There are four possible solutions for extracting the relative camera rotation R and translation t from the Essential matrix. However, only in (a) is the reconstructed point in front of both of the cameras.](/assets/images/cs231a/lecture04/스크린샷 2025-12-10 오전 9.36.40.png)

Figure 6에서 설명하듯이, Essential 행렬에서 상대 카메라 회전 $R$과 평행 이동 $t$를 추출하는 데 네 가지 가능한 해가 있습니다. $R$과 $t$ 모두에 대해 두 가지 옵션이 존재하기 때문입니다. 직관적으로, 네 가지 쌍은 특정 방향으로 카메라를 회전시키거나 반대 방향으로 회전시키는 옵션과 특정 방향으로 평행 이동시키거나 반대 방향으로 평행 이동시키는 옵션의 모든 가능한 조합을 포함합니다. 따라서 이상적인 조건에서, 올바른 $R, t$ 쌍을 결정하기 위해 하나의 점만 triangulate하면 됩니다. 올바른 $R, t$ 쌍의 경우, triangulated 점 $\hat{P}$는 두 카메라 앞에 존재하며, 이는 두 카메라 참조 시스템에 대해 양의 z 좌표를 가진다는 것을 의미합니다. 측정 노이즈로 인해, 종종 하나의 점만 triangulate하는 것에 의존하지 않고, 대신 많은 점을 triangulate하고 두 카메라 앞에 이러한 점들 중 대부분을 포함하는 것이 올바른 $R, t$ 쌍으로 결정합니다.

## 5. An example structure from motion pipeline

상대 motion 행렬 $M_i$를 찾은 후, 이를 사용하여 점 $X_j$의 월드 좌표를 결정할 수 있습니다. algebraic 방법의 경우, 이러한 점들의 추정값은 perspective transformation $H$까지 정확할 것입니다. Essential 행렬에서 카메라 행렬을 추출하는 경우, 추정값은 스케일까지 알려질 수 있습니다. 두 경우 모두, 3D 점은 앞서 설명한 triangulation 방법을 통해 추정된 카메라 행렬로부터 계산될 수 있습니다.

다중 시점 경우로의 확장은 쌍별 카메라를 체인화하여 수행할 수 있습니다. 충분한 점 대응이 제공되면, algebraic approach 또는 Essential 행렬을 사용하여 임의의 카메라 쌍에 대한 카메라 행렬과 3D 점에 대한 해를 얻을 수 있습니다. 재구성된 3D 점은 카메라 쌍 간에 사용 가능한 점 대응과 연관됩니다. 이러한 쌍별 해는 다음에 볼 bundle adjustment라는 접근 방식에서 함께 결합(최적화)될 수 있습니다.

### 5.1 Bundle adjustment

지금까지 논의한 structure from motion 문제를 해결하기 위한 이전 방법들과 관련된 주요 제한 사항이 있습니다. factorization 방법은 모든 점이 모든 이미지에서 보인다고 가정합니다. 이것은 가림(occlusion)과 대응을 찾지 못하는 실패 때문에 매우 일어나기 어렵습니다. 특히 많은 이미지가 있거나 일부 이미지가 멀리 떨어져서 촬영된 경우입니다. 마지막으로 algebraic approach는 카메라 체인으로 결합될 수 있는 쌍별 해를 생성하지만, 모든 카메라와 3D 점을 사용하는 일관된 최적화된 재구성을 해결하지는 않습니다.

이러한 제한 사항을 해결하기 위해 **bundle adjustment**를 소개합니다. 이것은 structure from motion 문제를 해결하기 위한 비선형 방법입니다. 최적화에서, 우리는 재투영 오차를 최소화하는 것을 목표로 합니다. 이것은 재구성된 점을 추정된 카메라로 투영한 것과 모든 카메라와 모든 점에 대한 해당 관측값 사이의 픽셀 거리입니다. 이전에 triangulation에 대한 비선형 최적화 방법을 논의할 때, 우리는 주로 두 카메라 경우에 초점을 맞췄으며, 여기서는 자연스럽게 각 카메라가 둘 사이의 모든 대응을 보았다고 가정했습니다. 그러나 bundle adjustment는 여러 카메라를 처리하기 때문에, 각 카메라가 볼 수 있는 관측값에 대해서만 재투영 오차를 계산합니다. 궁극적으로 이 최적화 문제는 triangulation에 대한 비선형 방법에 대해 이야기할 때 소개한 것과 매우 유사합니다.

bundle adjustment의 비선형 최적화를 해결하기 위한 두 가지 일반적인 접근 방식에는 Gauss-Newton 알고리즘과 Levenberg-Marquardt 알고리즘이 포함됩니다. Gauss-Newton 알고리즘에 대한 세부 사항은 이전 섹션을 참조하고, Levenberg-Marquardt 알고리즘에 대한 더 자세한 내용은 Hartley와 Zisserman 교과서를 참조할 수 있습니다.

결론적으로, bundle adjustment는 우리가 조사한 다른 방법들과 비교할 때 몇 가지 중요한 장점과 제한 사항이 있습니다. 특히 많은 수의 시점을 원활하게 처리할 수 있고 특정 점이 모든 이미지에서 관찰 가능하지 않은 경우도 처리할 수 있기 때문에 특히 유용합니다. 그러나 주요 제한 사항은 파라미터가 시점 수에 따라 증가하기 때문에 특히 큰 최소화 문제라는 것입니다. 또한 비선형 최적화 기법에 의존하기 때문에 좋은 초기 조건이 필요합니다. 이러한 이유로, bundle adjustment는 종종 대부분의 structure from motion 구현의 최종 단계로 사용됩니다(즉, factorization 또는 algebraic approach 이후). factorization 또는 algebraic approach가 최적화 문제에 대한 좋은 초기 해를 제공할 수 있기 때문입니다.

---

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [Course_Notes_4.pdf](https://web.stanford.edu/class/cs231a/course_notes/Course_Notes_4.pdf)
