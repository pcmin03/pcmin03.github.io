---
title: "[CS231A] Lecture 04: Stereo Systems (스테레오 시스템)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Stereo Vision, Depth Estimation]
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

이 포스트는 Stanford CS231A 강의의 네 번째 강의 노트인 "Stereo Systems"를 정리한 것입니다.

**원본 강의 노트**: [Course_Notes_4.pdf](https://web.stanford.edu/class/cs231a/course_notes/Course_Notes_4.pdf)

<!--more-->
Lecture 4는 **“Stereo Systems and Structure from Motion”** 이야기를 다루는 강의다 

* Lecture 2·3에서 배운 **epipolar geometry**가 “2D–2D 관계”였다면,
* Lecture 4는 그걸 발판으로 **진짜 3D 구조와 카메라 움직임까지 복원하는 과정**을 다루는 강의다.

큰 흐름은 이렇게 흘러간다.

1. 한 점의 3D 위치를 두 개 이상의 이미지에서 복원하는 **Triangulation**
2. 여러 뷰에서 **3D 구조 + 카메라 포즈를 동시에** 추정하는 **Structure from Motion (SfM)**

   * 우선 **Affine(weak perspective)** 가정에서의 Tomasi–Kanade factorization
   * 그 다음 **완전한 Perspective SfM**
3. 마지막으로 모든 걸 한 번 더 묶어서 정리하는 **Bundle Adjustment**

아래에서 스토리랑 수식, 그리고 마지막에 OpenCV / Open3D 코드까지 같이 정리해본다.

---

## 1. Triangulation – 두 눈으로 한 점의 위치를 찾는 문제다

### 1.1 이상적인 세계에서의 직관이다

그림 1을 떠올리면 된다. 왼쪽 카메라 중심을 $O_1$, 오른쪽을 $O_2$라고 두고,
3D 점 $P$가 두 이미지에서 $(p, p')$로 보인다고 하자 

* 각 카메라는 내부 파라미터 $(K, K')$와
* 서로 간의 상대 자세 $(R, T)$를 이미 알고 있다고 가정한다.

그러면 우리는 이렇게 생각할 수 있다.

* 왼쪽에서 $p$가 관측되면, 카메라 중심 $O_1$에서 나가는 **광선 $\ell$** 하나가 정해진다.
* 오른쪽에서 $p'$가 관측되면, $O_2$에서 나가는 **광선 $\ell'$** 하나가 정해진다.

이상적인 수학 세계에서는 두 광선이 딱 **한 점 $P$**에서 교차해야 한다.
바로 그 교차점이 우리가 찾는 3D 위치다.

### 1.2 현실에서 생기는 문제다

현실에서는

* 픽셀 위치 측정에 노이즈가 있고
* 카메라 캘리브레이션 (K, K', R, T)도 완벽하지 않다.

그래서 실제로는 $\ell$과 $\ell'$이 **살짝 어긋나서 아예 교차하지 않는 경우**가 많다(그림 2).
그래서 “교차점”을 찾겠다는 발상만으로는 부족하다.

그래서 강의에서는

1. **선형(Linear) Triangulation**으로 대략적인 초기값을 만들고
2. 그다음 **비선형(Nonlinear) 최소제곱 최적화**로 실제 찾고 싶은
   “reprojection error 최소” 지점을 refine하는 전략을 사용한다.

---

## 2. Linear Triangulation – SVD로 한 방에 근사값을 구하는 방법이다

두 이미지에서 대응점이 $p = (x, y, 1)^T$, $p' = (x', y', 1)^T$이고,
각 카메라 projection matrix가 $(M, M')$라고 하자.

투영식은

$$p = MP, \quad p' = M'P$$

로 쓸 수 있다. 여기서 $P$는 4D 동차좌표 $(X,Y,Z,1)^T$다.

### 2.1 교차곱 조건을 이용하는 아이디어다

벡터 $p$와 $MP$가 같은 방향을 가리키므로,
**교차곱이 0**이어야 한다.

$$p \times (MP) = 0$$

교차곱 성질을 좌표로 풀면 세 개의 스칼라 식을 얻는다:

$$\begin{aligned}
x(M_3 P) - (M_1 P) &= 0 \\
y(M_3 P) - (M_2 P) &= 0 \\
x(M_2 P) - y(M_1 P) &= 0
\end{aligned}$$

여기서 $M_i$는 $M$의 $i$번째 행이다.
마찬가지로 $(p', M')$에 대해 또 세 개를 만들 수 있다.

실제로는 첫 두 식만 써도 충분해서, 두 뷰에 대해 총 **4개의 직선식**을 만든다:

$$A P = 0, \quad A = \begin{bmatrix} x M_3 - M_1 \\ y M_3 - M_2 \\ x' M'_3 - M'_1 \\ y' M'_3 - M'_2 \end{bmatrix}$$

* 미지수는 $P$ (4차원 동차).
* 식은 4개.
* 스케일까지 포함하면 **1차원 null space**를 가지는 선형 시스템이다.

SVD로 $A$를 분해하면,

* $AP = 0$을 만족하는 **비자명한 해**는
* **가장 작은 특이값에 대응하는 오른쪽 singular vector**가 된다.

그 벡터를 $P$로 쓰고
마지막 좌표로 나누어 $(X,Y,Z)$를 얻으면
**선형 triangulation 결과**가 된다.

> 한 줄 요약
> – “카메라 두 개 + 대응점 한 쌍 → SVD 한 번으로 3D 점 하나 대략 복원한다.”

### 2.2 Projective invariance 문제다

이 선형 방법은 간단하지만 **projective invariant가 아니다**라는 문제를 가진다 

* 카메라 행렬들을 어떤 projective transform $H^{-1}$로 바꿔서
  $M \to MH^{-1}$, $M' \to M'H^{-1}$라고 하면,
* 식은 $AP=0$에서 $AH^{-1}(HP)=0$ 꼴이 된다.

즉, 원래 해 $P$가 새 문제에서는 $HP$로 변환된다.
그런데 SVD는 $|P|=1$ 같은 **유클리드 노름 제약**을 쓰고 있고,
이건 projective transform에 대해 invariant하지 않다.

그래서

* 이 방법은 **초기값**으로 쓰기엔 좋지만,
* “projective reconstruction에서 최종 해”로 쓰기에는 부족하다.

그래서 바로 다음에 **비선형 최적화**가 등장한다.

---

## 3. Nonlinear Triangulation – Reprojection Error를 직접 최소화하는 방법이다

현실에서 진짜로 우리가 원하는 건 이거다:

> "3D 점 $P$를 어떤 값으로 잡든,
> 그걸 모든 카메라에 다시 project했을 때
> 실제 관측 값들과 차이가 최소가 되도록 하자."

이를 **reprojection error를 최소화**한다고 부른다.

### 3.1 2-view 경우의 목표 함수다

두 카메라 $(M, M')$, 관측점 $(p, p')$가 있을 때,
우리는 다음을 최소화하고 싶다:

$$\min_{\hat{P}} |M \hat{P} - p|^2 + |M' \hat{P} - p'|^2$$

* $M\hat{P}$: 3D 추정치 $\hat{P}$를 첫 번째 카메라로 투영한 결과
* $p$: 실제 관측 픽셀
* 두 카메라에 대한 오차를 제곱합으로 더한다.

여러 뷰가 있다면 단순히 모든 뷰에 대해 더해주면 된다:

$$\min_{\hat{P}} \sum_i |M_i \hat{P} - p_i|^2$$

### 3.2 Nonlinear Least Squares 형태다

카메라 투영은 **동차 좌표 나누기**가 들어가서 비선형이다.
그래서 전체 문제는 **비선형 최소제곱** 문제가 된다.

일반적인 비선형 최소제곱은

$$\min_x |r(x)|^2 = \sum_i r_i(x)^2$$

형태다. 여기서 $r(x)$는 residual vector다.

Triangulation에서는

* 각 카메라에서의 reprojection error를
* residual 벡터 $e_i(\hat{P})$라고 두면 된다.

$$\min_{\hat{P}} \sum_i |e_i(\hat{P})|^2$$

여기서 $e_i = M_i \hat{P} - p_i$ 같은 형태다(정확히는 정규화된 좌표).

### 3.3 Gauss–Newton으로 푸는 스토리다

Gauss–Newton의 핵심 아이디어는

> “현재 추정치 주변에서 residual을 선형화해서
> 매번 선형 least squares 문제를 푼다.”

1. 우선 선형 triangulation으로 **대략적인 $\hat{P}$**를 하나 구한다.
2. 그 근처에서 **테일러 1차 근사**로 residual을 선형화한다.

$$e(\hat{P} + \delta P) \approx e(\hat{P}) + \frac{\partial e}{\partial P} \delta P$$

3. 이걸 그대로 최소제곱 문제로 바꾸면

$$\min_{\delta P} \left| J \delta P - (-e(\hat{P})) \right|^2$$

여기서 $J = \partial e / \partial P$는 **Jacobian**이다.

4. 선형 least squares의 해는

$$\delta P = -(J^T J)^{-1} J^T e$$

5. 새로운 추정치는

$$\hat{P} \leftarrow \hat{P} + \delta P$$

6. 이 과정을 수 회 반복하다가

   * $\delta P$가 충분히 작아지거나
   * iteration 수가 상한에 도달하면 멈춘다.

이게 **Nonlinear triangulation**이다.
실제 코드에서는 SciPy `least_squares`나 직접 구현한 Gauss–Newton / Levenberg–Marquardt를 많이 쓴다.

---

## 4. Affine Structure from Motion – Weak Perspective 가정에서의 SfM이다

이제 “점 하나”가 아니라 **여러 점 + 여러 카메라**에서
동시에 구조와 모션을 추정하는 이야기로 넘어간다 

### 4.1 SfM 문제 셋업이다

* 카메라가 $m$개 있고, 각 카메라 $i$의 투영 행렬이 $M_i$다.
* 세계에는 3D 점이 $n$개 있고, 각 점을 $X_j$라고 한다.
* 카메라 $i$에서 점 $j$를 본 픽셀 위치를 $x_{ij}$라고 둔다.

목표는

* **모든 카메라 행렬 $M_i$**와
* **모든 3D 점 $X_j$**를
* 관측 $x_{ij}$만 가지고 동시에 찾는 것이다.

이게 바로 **Structure from Motion**이다.

### 4.2 먼저 문제를 쉽게 만드는 Affine / Weak Perspective 가정이다

Weak perspective 카메라에서는

$$M = \begin{bmatrix} m_1 & m_2 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad x = M X = \begin{bmatrix} m_1 X \\ m_2 X \\ 1 \end{bmatrix}$$

그래서 실제 2D 투영은 단순히

$$\begin{bmatrix} u \\ v \end{bmatrix} = A X + b \quad (A: 2\times 3, b:2\times 1)$$

이 된다. 즉, **선형 아핀 변환**만 남는다.

Affine SfM에서는 카메라 $i$에 대해

$$x_{ij} = A_i X_j + b_i$$

라고 쓸 수 있다.

* 카메라 하나당 $A_i$ 2×3, $b_i$ 2×1 → **8 DoF**
* 점 하나당 $X_j$ 3D → 3 DoF

그래서 전체 미지수는 $(8m + 3n)$ 개다.
각 관측 $x_{ij}$는 2D이므로 식은 $2mn$개다.

이를 통해 “최소로 필요한 카메라 수·점 개수” 같은 걸 rough하게 판단할 수 있다.

### 4.3 Tomasi–Kanade Factorization – 클래식 알고리즘이다

Tomasi–Kanade 방법은 크게 두 단계로 이뤄진다.

1. **데이터 중심 맞추기 (centering)**
2. **SVD로 factorization**

#### (1) 데이터 중심 맞추기다

각 이미지 $i$에 대해,
해당 이미지에서 보인 모든 점의 centroid를 $\bar{x}_i$라고 두고

$$\hat{x}_{ij} = x_{ij} - \bar{x}_i$$

로 좌표를 옮긴다.

동시에 세계 좌표도 점들의 평균 $\bar{X}$를 원점으로 옮겨서

$$\hat{X}_j = X_j - \bar{X}$$

라고 두면, 위 식이 다음처럼 단순해진다:

$$\hat{x}_{ij} = A_i \hat{X}_j$$

즉, translation (b_i)가 사라지고 **순수하게 선형 변환만** 남는다.

#### (2) 측정 행렬 D를 만들어 factorization한다

모든 카메라와 모든 점을 한 번에 모아서

$$D = \begin{bmatrix} \hat{x}_{11} & \hat{x}_{12} & \dots & \hat{x}_{1n} \\ \hat{x}_{21} & \hat{x}_{22} & \dots & \hat{x}_{2n} \\ \vdots & \vdots & & \vdots \\ \hat{x}_{m1} & \hat{x}_{m2} & \dots & \hat{x}_{mn} \end{bmatrix}$$

을 만든다(각 $\hat{x}_{ij}$가 2D이므로 실제 크기는 $2m \times n$이다) 

Affine 모델에 따르면

$$D = M S$$

* $M$: $(2m \times 3)$ motion matrix (각 카메라의 $A_i$를 위아래로 붙인 것)
* $S$: $(3 \times n)$ structure matrix (각 점의 3D 좌표를 열로 모은 것)

따라서 **rank(D) = 3**이어야 한다.

이제 SVD를 한다:

$$D = U \Sigma V^T$$

이론적으로는 상위 3개의 singular value만 남기면 된다:

$$D \approx U_3 \Sigma_3 V_3^T$$

여기서

* $U_3$: 처음 3개의 컬럼만
* $\Sigma_3$: 3×3 대각 행렬
* $V_3$: 처음 3개의 row만

이제 $(M, S)$를 다음처럼 잡을 수 있다:

$$M = U_3 \sqrt{\Sigma_3}, \quad S = \sqrt{\Sigma_3} V_3^T$$

이렇게 하면 $D \approx MS$가 된다.
즉, **카메라 자세(모션)와 3D 구조(스트럭처)를 한 번에 분해**한 셈이다.

### 4.4 Affine ambiguity – 평행성만 맞고 metric은 모호한 상태다

하지만 여기에는 **필연적인 모호성**이 있다.

임의의 invertible $3 \times 3$ 행렬 $A$에 대해

$$D = M S = (M A)(A^{-1} S)$$

도 항상 성립한다. 즉,

* $M$과 $S$는 사실 **공통 행렬 $A$**까지 곱해져도
  관측 $D$는 그대로다.

이걸 **affine ambiguity**라고 부른다.

* 평행성은 보존되지만
* 실제 metric scale, 각도 등은 정확히 알 수 없다는 뜻이다.

캘리브레이션된 카메라를 쓰는 경우에는
이 모호성이 **similarity ambiguity(회전·이동·스케일)** 정도로 줄어든다.
어쨌든 **절대 스케일**은 여전히 모른다.

---

## 5. Perspective Structure from Motion – 일반 카메라로 확장한 경우다

Affine 가정을 버리고, 이제 **일반적인 pinhole/perspective 카메라**로 돌아온다 

### 5.1 일반 카메라의 도수와 projective ambiguity다

일반적인 카메라 행렬은

$$M_i = \begin{bmatrix} a_{11} & a_{12} & a_{13} & b_1 \\ a_{21} & a_{22} & a_{23} & b_2 \\ a_{31} & a_{32} & a_{33} & 1 \end{bmatrix}$$

처럼 쓸 수 있고, **스케일까지** 포함하면 11 DoF다.

모든 카메라와 점에 대해

* projective transform $H \in \mathbb{R}^{4\times4}$를 적용해서
* $M_i \to M_i H^{-1}$, $X_j \to H X_j$라고 바꿔도
* 이미지 관측은 변하지 않는다.

그래서 전체 해는 **공통 projective transform $H$**까지 모호하다.

### 5.2 Algebraic Approach – Fundamental matrix에서 카메라 행렬을 뽑아내는 방법이다

2-view를 먼저 생각한다. 목표는:

* 두 이미지에서의 대응점들로 Fundamental matrix $F$를 추정하고
* 거기서 카메라 행렬 $(M_1, M_2)$를 **projective scale까지** 얻는 것이다.

#### (1) 첫 번째 카메라는 canonical로 만들 수 있다

어차피 공통 $H$ 만큼 모호하므로,
우리는 항상 $H$를 골라서

$$M_1 H^{-1} = [I|0], \quad M_2 H^{-1} = [A|b]$$

형태로 바꿀 수 있다.

이제 구조도 $P_e = H P$라고 두면,

$$p = [I|0] P_e, \quad p' = [A|b] P_e$$

여기서 중요한 관계가 하나 나온다:

$$p' = A p + b$$

즉, canonical 좌표에서
두 번째 이미지의 점은 첫 번째 이미지의 점에
선형변환 A와 translation b를 적용한 것과 같다.

#### (2) Fundamental matrix와의 연결이다

이제 $p'$와 $b$의 cross product를 생각한다:

$$p' \times b = (A p + b) \times b = A p \times b = [b]_\times A p$$

여기서 $[b]_\times$는 $b$의 skew-symmetric matrix다.

Epipolar constraint는 항상

$$p'^T F p = 0$$

형태를 가지는데, 위 식과 비교하면

$$F = [b]_\times A$$

이어야 한다는 사실을 알 수 있다.

#### (3) F에서 b, A 분해하기다

우리가 아는 건 $F$뿐이지만,
$F$가 **singular**라는 사실을 이용하면 된다.

$$F^T b = 0$$

이를 SVD로 풀면 $b$는 **$F^T$의 null space**에서 얻는 벡터다.
해석적으로는 **epipole**이다.

그 다음

$$A = -[b]_\times F$$

로 두면, 실제로 $F = [b]_\times A$가 됨을 확인할 수 있다.

결국 카메라 행렬은

$$\tilde{M}_1 = [I|0], \quad \tilde{M}_2 = [-[e]_\times F|e]$$

꼴이 된다. 여기서 $e=b$는 epipole이다.

이렇게 해서 “F → 카메라 행렬 두 개”를 구하는 algebraic approach가 완성된다.

---

## 6. Essential Matrix에서 R, t를 직접 뽑는 방법이다

카메라가 **캘리브레이션되어 있고**
내부 파라미터 $K$를 알고 있다면,
Fundamental matrix $F$를 Essential matrix로 바꿀 수 있다:

$$E = K^T F K$$

Essential matrix는

$$E = [t]_\times R$$

형태를 가지며,
실제로 우리가 알고 싶은 건 바로 이 **회전 $R$**과 **번역 $t$**다.

### 6.1 E의 SVD 구조를 이용하는 아이디어다

$E$는 rank 2이고, 두 개의 singular value가 같으며, 세 번째는 0이다.
그래서 SVD를

$$E = U \operatorname{diag}(1,1,0) V^T$$

꼴로 맞춰줄 수 있다(스케일 무시).

그리고 다음 두 행렬을 정의한다:

$$W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad Z = \begin{bmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

이때 factorization은 다음을 만족한다:

$$[t]_\times = U Z U^T, \quad R = U W V^T \text{ 또는 } U W^T V^T$$

그러나 $R$은 진짜 **회전 행렬**이어야 하므로,
$\det(R) > 0$이 되도록 부호를 조정한다:

$$R = (\det U W V^T) U W V^T$$

같은 식으로 맞춘다.

### 6.2 t는 U의 세 번째 컬럼이다

$[t]_\times$의 구조를 보면 $t$는 $U$의 세 번째 컬럼에서 나온다:

$$t = \pm \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \text{을 U로 변환} = \pm u_3$$

결국 가능한 $(R, t)$ 조합은 4개다:

* $(R_1 = U W V^T, t_1 = +u_3)$
* $(R_1, t_2 = -u_3)$
* $(R_2 = U W^T V^T, t_1)$
* $(R_2, t_2)$

이 네 가지 중 어느 것이 진짜인지 결정하는 방법은 간단하다.

> 여러 대응점에 대해 triangulation을 해 보고,
> **두 카메라에서 모두 앞쪽에 있는 점의 개수**가 가장 많은 조합이
> 진짜 (R, t)다.

그림 6을 보면 네 가지 구성이 그림으로 잘 표현되어 있다 

---

## 7. 전체 SfM 파이프라인과 Bundle Adjustment다

지금까지 나온 것들을 한 파이프라인으로 연결하면 대략 이렇게 된다.

1. 여러 이미지에서 feature를 추출하고 매칭한다.
2. 이미지 쌍마다 F를 추정한다 (8-point + RANSAC).
3. 카메라가 calibrate되어 있으면 F → E → (R, t)를 추출한다.
4. 하나의 reference 카메라를 [I|0]에 두고,
   나머지 카메라들을 기준 좌표계로 chain처럼 붙인다.
5. 각 카메라 쌍 / 여러 카메라에 대해 triangulation을 수행해
   3D 점들을 만든다.
6. 여기까지가 “대충 맞는 3D + 카메라 포즈”.
7. 마지막으로 **Bundle Adjustment**라는 큰 비선형 최적화를 돌려
   모든 카메라와 모든 3D 점을 한꺼번에 refine한다.

Bundle Adjustment의 목적함수는

$$\min_{\{M_i\}, \{X_j\}} \sum_{i,j \in \text{visible}} | \pi(M_i, X_j) - x_{ij} |^2$$

이다. 즉,

* 각 카메라 $M_i$, 점 $X_j$에 대해

* 실제 관측 $x_{ij}$와

* projection $\pi(M_i, X_j)$의 차이를

* **보이는 경우에만** 모두 더한 reprojection error를 최소화하는 문제다.

* Gauss–Newton, Levenberg–Marquardt 등으로 푼다.

* 변수 수가 카메라·점 개수에 비례해서 크기 때문에,
  sparseness를 활용하는 최적화가 중요하다.

실제 SfM 라이브러리(Ceres, g2o, COLMAP 등)가 이 부분을 수행한다.

---

## 8. OpenCV / Open3D로 해볼 수 있는 실습 코드들이다

실제로 Lecture 4 내용을 손으로 만져볼 수 있는 코드를 몇 가지 정리한다.

---

### 8.1 두 뷰에서 Linear + Gauss–Newton Triangulation을 구현한 예시다

#### 8.1.1 Linear triangulation (Lecture 4 식 2.2 그대로 구현)이다

```python
import numpy as np

def linear_triangulation(M1, M2, p1, p2):
    """
    M1, M2 : 3x4 projection matrices
    p1, p2 : (u, v) 픽셀 좌표 (2,)
    return : 3D 점 (X, Y, Z) (3,)
    """
    x, y = p1
    x_, y_ = p2

    A = np.zeros((4, 4))
    A[0] = x * M1[2] - M1[0]
    A[1] = y * M1[2] - M1[1]
    A[2] = x_ * M2[2] - M2[0]
    A[3] = y_ * M2[2] - M2[1]

    # SVD로 null-space 구한다
    _, _, Vt = np.linalg.svd(A)
    P_h = Vt[-1]
    P_h /= P_h[3]
    return P_h[:3]
```

---

#### 8.1.2 Gauss–Newton으로 reprojection error를 줄이는 refinement 코드다

```python
def project_point(M, P):
    """
    M : 3x4 projection matrix
    P : (3,) 3D
    return : (u, v)
    """
    P_h = np.append(P, 1.0)
    x = M @ P_h
    u = x[0] / x[2]
    v = x[1] / x[2]
    return np.array([u, v])

def gn_triangulation(Ms, ps, P0, iters=10):
    """
    Gauss-Newton으로 3D 점 하나를 refine하는 함수다.
    Ms : 리스트 [M1, M2, ...] (각 3x4)
    ps : 리스트 [p1, p2, ...] (각 (2,))
    P0 : 초기값 3D (선형 triangulation 결과 등)
    """
    P = P0.copy().astype(np.float64)

    for _ in range(iters):
        residuals = []
        J_rows = []

        for M, p in zip(Ms, ps):
            # 예측 projection
            P_h = np.append(P, 1.0)
            x = M @ P_h
            Xp, Yp, Zp = x
            u_hat = Xp / Zp
            v_hat = Yp / Zp

            # residual (pred - obs)
            e = np.array([u_hat - p[0],
                          v_hat - p[1]])
            residuals.append(e)

            # Jacobian 계산
            m1, m2, m3 = M[0], M[1], M[2]
            # d(x)/dP = m1[:3], d(z)/dP = m3[:3]
            dx_dP = m1[:3]
            dy_dP = m2[:3]
            dz_dP = m3[:3]

            du_dP = (dx_dP * Zp - Xp * dz_dP) / (Zp ** 2)
            dv_dP = (dy_dP * Zp - Yp * dz_dP) / (Zp ** 2)

            J_rows.append(du_dP)
            J_rows.append(dv_dP)

        e_vec = np.concatenate(residuals)           # (2N,)
        J = np.vstack(J_rows)                       # (2N, 3)

        # normal equation 풀기
        JTJ = J.T @ J
        JTe = J.T @ e_vec
        try:
            delta = -np.linalg.solve(JTJ, JTe)
        except np.linalg.LinAlgError:
            break

        P = P + delta

        if np.linalg.norm(delta) < 1e-6:
            break

    return P
```

사용 예시는:

```python
# 두 카메라의 projection matrix M1, M2
# 두 이미지에서의 픽셀 좌표 p1, p2 (예: (u, v))

P_init = linear_triangulation(M1, M2, p1, p2)
P_refined = gn_triangulation([M1, M2], [p1, p2], P_init)
```

이렇게 하면 Lecture 4에서 설명한

* 선형 triangulation으로 초기 추정
* Gauss–Newton으로 reprojection error 최소화

를 그대로 구현해서 확인할 수 있다.

---

### 8.2 OpenCV + Open3D로 “작은 SfM” 맛보기 코드다

다음 코드는

1. 두 장의 이미지에서 ORB feature 매칭
2. F 및 E 계산
3. `cv2.recoverPose`로 R, t 추출 (Essential decomposition)
4. `cv2.triangulatePoints`로 여러 점을 triangulation
5. Open3D로 점들을 3D로 시각화

까지 한 번에 보여준다.

```python
import cv2
import numpy as np
import open3d as o3d

# 1. 이미지와 내부 파라미터를 준비한다
img1 = cv2.imread("left.jpg")
img2 = cv2.imread("right.jpg")

if img1 is None or img2 is None:
    raise RuntimeError("이미지를 찾을 수 없다.")

h, w = img1.shape[:2]

# 예시용 K (실제 카메라 값으로 바꾸면 더 좋다)
fx = fy = 1200.0
cx = w / 2.0
cy = h / 2.0

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# 2. ORB 특징점 검출 + 매칭
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(2000)
kps1, des1 = orb.detectAndCompute(gray1, None)
kps2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)

N = 500
matches = matches[:N]

pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

# 3. Fundamental matrix와 Essential matrix 계산
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

E = K.T @ F @ K
print("F =\n", F)
print("E =\n", E)

# 4. Essential matrix로부터 R, t 복원 (Lecture 4의 E 분해를 OpenCV가 해 준다)
pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)

_, R, t, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm, K)
print("R =\n", R)
print("t =\n", t)

# 5. Triangulate
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))   # [I | 0]
P2 = K @ np.hstack((R, t))                          # [R | t]

pts1_h = pts1.T
pts2_h = pts2.T

pts4D = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
pts3D = (pts4D[:3] / pts4D[3]).T   # (N, 3)

# 6. Open3D로 포인트 클라우드 시각화
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts3D)

colors = np.zeros_like(pts3D)
colors[:, 0] = 1.0  # 빨간색
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])
```

이 코드를 직접 돌려보면

* Lecture 3에서 배운 **epipolar geometry + Essential matrix** 이론과
* Lecture 4에서 다룬 **triangulation + SfM 초기화** 이론이
* 실제로 **구체적인 3D 점 클라우드**로 눈앞에 나타나는 걸 볼 수 있다.

여기서 추가로

* 여러 뷰를 쌓고
* 카메라 chain을 만들고
* Ceres / g2o / `cv2.bundleAdjust` 같은 걸 써서 전체를 refine하면

실제 SfM 시스템과 거의 같은 구조가 된다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [Course_Notes_4.pdf](https://web.stanford.edu/class/cs231a/course_notes/Course_Notes_4.pdf)
