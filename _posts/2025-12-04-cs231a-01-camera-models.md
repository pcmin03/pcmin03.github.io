---
title: "[CS231A] Lecture 01: Camera Models (카메라 모델)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Camera Models, Computer Vision]
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

이 포스트는 Stanford CS231A 강의의 첫 번째 강의 노트인 "Camera Models"를 정리한 것입니다.

**원본 강의 노트**: [01-camera-models.pdf](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf)

<!--more-->
아래는 **CS231A Course Notes 1: Camera Models** 전체를, 처음 보는 사람도 이해할 수 있게 다시 정리한 마크다운 노트야. 

---

# 1. 카메라 모델 개요

* 컴퓨터 비전에서 카메라는 **3D 세계 → 2D 이미지**로 바꿔주는 장치.
* 우리가 해야 할 일:

  * 3D 점 $P = (x,y,z$) 가
  * 이미지 평면 위의 2D 점 $p = (x', y'$) 로
  * **어떻게 사상$project$** 되는지 수학적으로 모델링하는 것.

이 노트는 그 중에서도:

1. **핀홀 카메라(pinhole  camera) 모델**
2. **렌즈가 있는 실제 카메라(얇은 렌즈 / paraxial  model)**
3. **디지털 이미지 좌표(픽셀)로 가는 과정 및 카메라 행렬 $K$**
4. **내·외부 파라미터 (intrinsic / extrinsic)**
5. **카메라 캘리브레이션**
6. **왜곡$distortion$**
7. **다른 단순화된 카메라 모델(weak  perspective, orthographic)**

까지 한 번에 정리한다.

---

# 2. 핀홀 카메라 모델

## 2.1 개념

* **장치 구성**

  * 3D 물체와 필름(이미지 평면) 사이에 **작은 구멍(조리개, pinhole)** 이 있는 차단막을 둔다.
  * 물체의 각 점에서 여러 광선이 나오지만,
  * **구멍을 통과하는 한 가닥의 광선만** 필름에 도달.
* 이렇게 하면:

  * 3D의 각 점 ↔ 필름 위의 한 점
  * **1:1 대응**이 생겨 이미지가 형성된다.

## 2.2 기하 구조와 좌표계

그림 2 구조를 생각하자:

* 카메라 중심(핀홀): $O$
* 이미지 평면: (\Pi')
* 카메라 좌표계: ((\mathbf{i}, \mathbf{j}, \mathbf{k}))

  * 원점: $O$
  * (\mathbf{k}) 축은 **이미지 평면에 수직**이고, 평면 쪽으로 향함.
* 3D 점:

  * $P = (x, y, z$^T)  (카메라 좌표계)
* 이미지 평면 점:

  * (P' = (x', y')^T)
* 초점 거리:

  * 핀홀과 이미지 평면 사이 거리: $f$

### 2.3 유사 삼각형으로부터의 투영식

삼각형을 보면:

* 한 삼각형: (P, O, (0,0,z))
* 다른 삼각형: (P', C', O)  (C'는 광학 축과 이미지 평면이 만나는 점)

두 삼각형은 닮음 관계라서,

$$\frac{x'}{f} = \frac{x}{z}, \quad \frac{y'}{f} = \frac{y}{z}$$

따라서 **핀홀 투영식**은:

$$P' = \begin{bmatrix} x'  y' \\end{bmatrix} = \begin{bmatrix} f \dfrac{x}{z}  f \dfrac{y}{z} \end{bmatrix}$$

핵심 포인트:

* 투영은 **비선형**이다. (나누기 $z$ 때문)
* $z$가 클수록(멀수록) $(x', y')$가 작아진다 → 원근감.

---

# 3. 조리개 크기와 블러

* 이론에서는 조리개가 **점**이라고 가정했지만, 현실은 **유한한 크기**.
* 조리개가 **커지면**:

  * 더 많은 광선이 통과 → 각 픽셀에 여러 3D 점의 빛이 섞임 → **블러(흐림)** 증가.
* 조리개가 **작아지면**:

  * 광선이 거의 한 가닥만 통과 → **선명**해지지만
  * 들어오는 빛이 적어서 **어두워짐**.

→ **밝기 vs 선명도**의 트레이드오프가 핀홀 모델의 근본적인 한계.

---

# 4. 렌즈가 있는 카메라

## 4.1 렌즈의 역할

* 실제 카메라는 **렌즈**를 써서:

  * 조리개를 크게 열고도
  * 한 점 $P$에서 나온 복수의 광선을 **하나의 점 (P')** 에 모이도록 굴절시킨다.
* 그러면:

  * 더 많은 빛을 쓰면서도
  * 어느 특정 거리의 물체는 또렷하게 보이게 할 수 있다.

하지만:

* 어떤 **특정 거리 $z$** 에 있는 점만 정확히 모인다 → **초점이 맞는다(in  focus)**.
* 더 가깝거나 먼 점은 이미지 평면에서 완전히 모이지 못해 **out-of-focus  blur** 발생.
* 이와 관련된 개념이 **depth  of field(심도)**.

## 4.2 초점거리와 얇은 렌즈 모델 (Paraxial  refraction)

그림 5 상에서:

* 광학 축에 평행한 모든 광선은 렌즈 뒤의 **초점점(focal  point)** 으로 모인다.
* 렌즈 중심에서 초점까지 거리가 **초점거리 $f$**.

또 다른 성질:

* 렌즈 **중심**을 지나는 광선은 방향이 바뀌지 않는다(굴절 X).

이 가정을 포함해서 얇은 렌즈(“thin  lens”) 모델을 쓰면, 핀홀 모델과 비슷한 형태의 수식을 얻는다:

$$P' = \begin{bmatrix} x'y' \\end{bmatrix} \begin{bmatrix} z_0 \dfrac{x}{z} z_0 \dfrac{y}{z} \end{bmatrix} \$$

* 여기서 $z_0$ 는 **이미지 평면과 렌즈 사이의 거리**로, 핀홀 모델의 $f$와 비슷한 역할.
* 단, 렌즈 모델에서는 이 거리가 (f + z_0) 같은 형태로 조금 달라진다고만 이해해두면 된다.

이 모델은 **paraxial refraction model(준축 근사)** 라고 부른다.

---

# 5. 렌즈 때문에 생기는 왜곡 (Radial distortion)

얇은 렌즈 가정은 완벽하지 않아서:

* **Radial distortion(방사 왜곡)** 이 생긴다.
* 중심(광학 축)에서 멀어질수록 확대 비율이 달라짐.

종류:

1. **Barrel distortion (배럴 왜곡)**

   * 격자 선이 바깥쪽으로 볼록하게 휘어짐.
   * 가장자리로 갈수록 **축소**되는 느낌.
2. **Pincushion distortion (핀쿠션 왜곡)**

   * 격자 선이 안쪽으로 오목하게 휘어짐.
   * 가장자리로 갈수록 **확대**되는 느낌.

실제 카메라 보정$calibration$에서는 이 왜곡을 수학적으로 모델링해서 보정한다.

---

# 6. 디지털 이미지 공간으로 가기

지금까지는 **이미지 평면**까지의 투영만 했고, 실제 디지털 이미지(픽셀 좌표)로는 아직 안 갔다.

해야 할 것:

1. **좌표계 차이**

   * 이론상 이미지 평면:

     * 원점: 광학 축과 만나는 점 (C') (이미지 중심)
   * 디지털 이미지:

     * 보통 (0,0)이 왼쪽 아래(또는 왼쪽 위) 코너.
   * 둘 사이에 **평행이동$translation$** 이 필요.

2. **단위 차이**

   * 이미지 평면: cm, mm 같은 실제 길이 단위.
   * 디지털 이미지: **픽셀 단위**.
   * 축마다 픽셀 크기가 다를 수 있음 (비정방 픽셀).

3. **비선형 왜곡**

   * 앞서 말한 radial  distortion 등 (뒤에서 따로 다룸).

## 6.1 평행이동 파라미터 (c_x, c_y)

* 이미지 평면 중심(광학 축과의 교차점)에서,
* 디지털 이미지 좌표계의 원점까지의 오프셋:

$$(c_x, c_y)$$

이를 반영하면:

$$\begin{bmatrix} x'y' \\end{bmatrix} \begin{bmatrix} f \dfrac{x}{z} + c_x  f \dfrac{y}{z} + c_y \end{bmatrix} \$$

## 6.2 픽셀 크기 파라미터 $\k, \l$ (또는 $\alpha, \beta$)

픽셀 단위로 바꾸기 위해:

* $k$: x축 방향의 **pixel / length** 비율
* $l$: y축 방향의 비율

$$\alpha = fk,\quad \beta = fl$$

그래서 식은:

$$\begin{bmatrix} x'  y' \\end{bmatrix} = \begin{bmatrix} fk \dfrac{x}{z} + c_x  fl \dfrac{y}{z} + c_y \end{bmatrix} = \begin{bmatrix} \alpha \dfrac{x}{z} + c_x  \beta \dfrac{y}{z} + c_y \end{bmatrix}$$

* $k = l$ 이면 **정방 픽셀(square  pixels)** 이라고 부른다.

---

# 7. 동차 좌표(Homogeneous  coordinates)

지금까지 투영식은 **나누기 $z$** 때문에 **비선형**이다.
행렬 곱 하나로 정리하기 위해 **동차 좌표(homogeneous  coordinates)** 를 쓴다.

## 7.1 정의

* 2D 점 $(x',y')$ → 동차: $(x',y',1)$
* 3D 점 $(x,y,z)$ → 동차: $(x,y,z,1)$

일반적으로:

* 유클리드 벡터 $(v_1, \dots, v_n)$ → 동차: $(v_1,\dots,v_n,1)$
* 동차 벡터 $(v_1,\dots,v_n, w)$ → 유클리드: $(v_1/w,\dots,v_n/w)$

## 7.2 투영식을 행렬로 표현

식 (4)를 동차 좌표로 다시 쓰면:

$$P'_h = \begin{bmatrix} \alpha  x  +  c_x  z  \beta  y  +  c_y  z  z \\end{bmatrix} \begin{bmatrix} \alpha  &  0  &  c_x  &  00  &  \beta  &  c_y  &  00  &  0  &  1  &  0 \\end{bmatrix} \begin{bmatrix} xyz1 \\end{bmatrix} \$$

동차 좌표를 기본으로 쓰겠다는 약속을 하면, 인덱스 $h$ 는 생략:

$$P' = \begin{bmatrix} x'y'z' \\end{bmatrix} \underbrace{ \begin{bmatrix} \alpha  &  0  &  c_x  &  00  &  \beta  &  c_y  &  00  &  0  &  1  &  0 \\end{bmatrix}}_{M} \begin{bmatrix} xyz1 \\end{bmatrix} = MP \$$

여기서 $M$을 **투영 행렬(projection matrix)** 라고 부른다.

## 7.3 내적 카메라 행렬 $K$

위 식을

$$M = K [I|0]$$

꼴로 분해할 수 있다:

$$P' = MP = K [I|0] P$$

여기서

$$K = \begin{bmatrix} \alpha  &  0  &  c_x  0  &  \beta  &  c_y  0  &  0  &  1 \\end{bmatrix}$$

을 **카메라 행렬(camera matrix)** 또는 **intrinsic matrix** 라고 부른다.

---

# 8. 완전한 카메라 행렬: Skew 포함

현실에는 센서 축이 완벽히 직교하지 않을 수 있다 → **skew** 발생.

* x축과 y축 사이 각도를 (\theta) 라고 하면,
* 완전한 내적 행렬은:

$$K = \begin{bmatrix} \alpha & -\alpha \cot\theta & c_x  0 & \dfrac{\beta}{\sin\theta} & c_y  0 & 0 & 1 \end{bmatrix} \$$

요약:

* $\alpha, \beta$: 두 방향의 **유효 초점거리(focal  length in  pixels)**
* (c_x, c_y): **주점(principal  point)** 또는 이미지 중심 오프셋
* (\theta): **축 사이 각도(스큐)**

이 다섯 개가 **내적 파라미터(intrinsic  parameters)** 라고 불리고,
카메라 하드웨어(센서, 렌즈)에 고유하다.

---

# 9. 외적 파라미터 (Extrinsic  parameters)

지금까지는 **카메라 좌표계**에서의 3D 점 $P$ 를 가정했다.
하지만 보통은 **세계 좌표계(world  frame)** 에서 점 $P_w$ 를 알고 있다.

## 9.1  World → Camera 좌표 변환

* 회전 행렬 (R \in  SO(3))
* 번역 벡터 (T \in \mathbb{R}^3)

을 통해:

$$P = \begin{bmatrix} R  &  T0  &  1 \\end{bmatrix} P_w \$$

여기서 (P, P_w) 는 동차 좌표.

## 9.2 최종 투영식

식 (7)에 대입하면:

$$P\' = K [R|T] P_w = M P_w$$

요약:

* $K$: 내적 파라미터 (5 자유도)
* $\R, \T$: 외적 파라미터 (3 + 3 자유도)
* 전체 투영 행렬 $M$: (3\times4), 총 11 자유도 (스케일 1개는 의미 없음).

---

# 10. 카메라 캘리브레이션(Camera  Calibration)

목표:

* 이미지만 보고 **(K, R, T)** (→ 결국 $M$) 를 추정.

방법:

1. **캘리브레이션 패턴** (체스보드 등) 준비

   * 3D에서 각 코너의 좌표 (P_1,\dots,P_n) 을 알고 있다 (world  frame).
2. 카메라로 사진을 찍고

   * 이미지에서 코너 픽셀 좌표 (p_1,\dots,p_n) 을 추출.

## 10.1 한 점에 대한 수식

* $P_i$: 동차 세계 좌표
* $p_i = (u_i, v_i$): 이미지 좌표
* $M$ 의 각 행을 (m_1^T, m_2^T, m_3^T) 라고 하면:

$$p_i = \begin{bmatrix} u_iv_i \\end{bmatrix} \begin{bmatrix} \dfrac{m_1  P_i}{m_3  P_i} \dfrac{m_2  P_i}{m_3  P_i} \end{bmatrix} \$$

→ 하나의 대응쌍 ((P_i, p_i)) 가 **두 개의 스칼라 방정식**을 준다.

## 10.2 선형 시스템 구성

위 식에서 분모를 없애면:

$$u_i (m_3  P_i) - (m_1  P_i) = 0  v_i (m_3  P_i) - (m_2  P_i) = 0$$

이를 모든 $i = 1,\dots,n$ 에 대해 모으면:

$$\begin{bmatrix} P_1^T  &  0^T  &  -u_1  P_1^T0^T  &  P_1^T  &  -v_1  P_1^T\vdots  &  \vdots  &  \vdotsP_n^T  &  0^T  &  -u_n  P_n^T0^T  &  P_n^T  &  -v_n  P_n^T \\end{bmatrix} \begin{bmatrix} m_1^Tm_2^Tm_3^T \\end{bmatrix} P  m = 0 \$$

* 미지수 $m$ 은 길이 12 벡터지만, 전체 행렬은 스케일까지 포함하므로 **유효 자유도 11개**.
* 최소 6점(12방정식)이 필요하지만, 실제로는 훨씬 많은 점을 써서 **과잉 결정$overdetermined$** 시스템으로 만든다.

## 10.3  SVD로 최소 제곱 해 찾기

우리는 $Pm = 0$ 을 만족하는 **비자명(non-trivial)** 해를 원한다.

문제 정식화:

$$\min_m |Pm|_2^2 \quad \text{s.t.}\quad |m|_2^2 = 1 \$$

* 이 최적화 문제의 해는:

  * $P = UDV^T$ $SVD$ 에서
  * **가장 작은 특이값에 대응하는 $V$ 의 마지막 열**.

이렇게 얻은 $m$ 을 다시 (3\times4) 행렬 $M$ 으로 reshape하면,
투영 행렬을 **스케일까지** 얻는다.

## 10.4 $M$ → (K, R, T) 분해

우리는 실제로는 $M = K [R;T]$ 이기 때문에, 여기서 다시 내·외부 파라미터를 뽑는다.

노트에서는 $M$ 의 스케일을 (\rho) 로 두고 다음과 같이 쓴다 (각 $a_i$ 는 $M$ 의 행):

$$M = \frac{1}{\rho} \begin{bmatrix} \alpha  r_1^T - \alpha \cot\theta, r_2^T + c_x  r_3^T & \alpha  t_x - \alpha \cot\theta  t_y + c_x  t_z \frac{\beta}{\sin\theta} r_2^T + c_y  r_3^T & \frac{\beta}{\sin\theta} t_y + c_y  t_z r_3^T & t_z \end{bmatrix} \begin{bmatrix} a_1^Ta_2^Ta_3^T \\end{bmatrix}$$

이걸 이용해 내적 파라미터를 구하는 식이 주어진다:

$$\begin{aligned} \rho &= \pm \frac{1}{|a_3|} c_x &= \rho^2 (a_1 \cdot  a_3) c_y &= \rho^2 (a_2 \cdot  a_3) \theta &= \cos^{-1}\left( -\dfrac{(a_1 \times  a_3)\cdot (a_2 \times  a_3)}{|a_1\times  a_3|;|a_2\times  a_3|} \right) \alpha &= \rho^2 |a_1 \times  a_3| \sin\theta \beta &= \rho^2 |a_2 \times  a_3| \sin\theta \end{aligned} \$$

외적 파라미터:

$$\begin{aligned} r_1 &= \dfrac{a_2 \times  a_3}{|a_2 \times  a_3|} r_3 &= \rho  a_3 r_2 &= r_3 \times  r_1 T &= \rho  K^{-1} b \end{aligned} \$$

(여기서 $b$ 는 $M$ 의 마지막 열)

핵심 아이디어:

* $M$ 의 각 행/열 관계와 정규직교성($R$ 이 회전 행렬) 조건을 이용해서,
* **대수적으로 (K, R, T) 를 복원**하는 과정이다.

## 10.5 퇴화 구성 (Degenerate  configuration)

모든 점 집합이 잘 되는 건 아니다.

* 예: **모든 $P_i$ 가 한 평면 위에만 있을 때**

  * 투영 행렬을 유일하게 정할 수 없음.
* 이런 경우를 **degenerate  configuration(퇴화 구성)** 이라고 한다.

실제 캘리브레이션에서는 패턴을 여러 위치·각도에서 찍어
3D 점들이 좋은 분포를 갖도록 한다.

---

# 11. 왜곡을 고려한 캘리브레이션

앞에서는 **이상적인 렌즈**를 가정.
실제는 radial  distortion 때문에 추가 파라미터가 필요하다.

## 11.1 간단 모델

동차 좌표에서:

$$Q  P_i = \begin{bmatrix} 1/\lambda  &  0  &  00  &  1/\lambda  &  00  &  0  &  1 \\end{bmatrix} M  P_i = \begin{bmatrix} u_iv_i \\end{bmatrix} = p_i \$$

여기서 (\lambda) 는 왜곡 관련 파라미터(들)의 함수.

이 경우

$$\begin{aligned} u_i  q_3 P_i &= q_1  P_i v_i  q_3 P_i &= q_2  P_i \end{aligned}$$

이 되어 **비선형** 시스템이 된다 → 일반적인 비선형 최적화 필요.

## 11.2 비율을 이용한 단순화

Radial  distortion은 중심으로부터의 거리만 바꾸기 때문에,
**좌표의 비율** (u_i / v_i) 는 유지된다고 가정할 수 있다.

$$\frac{u_i}{v_i} # \frac{m_1  P_i / (m_3  P_i)}{m_2  P_i / (m_3  P_i)} \frac{m_1  P_i}{m_2  P_i} \$$

이걸 $n$개 점에 대해 모으면 선형 방정식:

$$\begin{aligned} v_1 (m_1  P_1) - u_1 (m_2  P_1) &= 0 \vdots v_n (m_1  P_n) - u_n (m_2  P_n) &= 0 \end{aligned}$$

행렬 형태:

$$L_n = \begin{bmatrix} v_1  P_1^T  &  -u_1  P_1^T\vdots  &  \vdotsv_n  P_n^T  &  -u_n  P_n^T \\end{bmatrix} \begin{bmatrix} m_1^Tm_2^T \\end{bmatrix} \$$

여기서 (m_1, m_2) 를 먼저 SVD로 추정하고,
그 다음 $m_3$ 와 (\lambda) 를 더 단순한 비선형 최적화로 찾는 전략을 취한다.

---

# 12. Appendix  A – Rigid  Transformations (3D 변환 기본)

카메라 모델에서 계속 등장하는 변환들을 한 번 정리하자.

## 12.1 회전 행렬 (Rotation  matrices)

3D에서 축별 회전:

* x축 회전 (\alpha):

$$R_x(\alpha) = \begin{bmatrix} 1  &  0  &  00  &  \cos\alpha  &  -\sin\alpha0  &  \sin\alpha  &  \cos\alpha \\end{bmatrix}$$

* y축 회전 (\beta):

$$R_y(\beta) = \begin{bmatrix} \cos\beta  &  0  &  \sin\beta0  &  1  &  0-\sin\beta  &  0  &  \cos\beta \\end{bmatrix}$$

* z축 회전 (\gamma):

$$R_z(\gamma) = \begin{bmatrix} \cos\gamma  &  -\sin\gamma  &  0\sin\gamma  &  \cos\gamma  &  00  &  0  &  1 \\end{bmatrix}$$

연속 회전:

* 먼저 z, 그다음 y, 그다음 x 순서로 돌리면:
  $R = R_x  R_y R_z$ (행렬 곱 순서에 주의).

## 12.2 평행이동 $Translation$

벡터 $t = (t_x, t_y, t_z$^T) 만큼 이동:

$$P' = P + t$$

동차 좌표에서의 번역 행렬:

$$T = \begin{bmatrix} 1  &  0  &  0  &  t_x0  &  1  &  0  &  t_y0  &  0  &  1  &  t_z0  &  0  &  0  &  1 \\end{bmatrix}$$

$$P'_h = T  P_h$$

## 12.3 스케일 $Scaling$

축별 스케일 (S_x, S_y, S_z):

$$S = \begin{bmatrix} S_x  &  0  &  00  &  S_y  &  00  &  0  &  S_z \\end{bmatrix}$$

## 12.4 합성 변환

* 먼저 스케일 $S$, 그 후 회전 $R$, 그 후 평행이동 $t$:

$$T_{\text{total}} = \begin{bmatrix} R  S  &  t0  &  1 \\end{bmatrix}$$

이런 형태의 행렬은 **아핀 변환(affine  transformation)** 이고,
마지막 행이 ([0;0;0;1]) 이 아니면 **사영 변환$projective$** 이다.

---

# 13. Appendix  B – 다른 카메라 모델들

## 13.1  Weak Perspective  Model (약 원근 투영)

아이디어:

1. 카메라에서 거리 $z_0$ 에 있는 **참조 평면 (\Pi)** 를 하나 잡는다.
2. 모든 3D 점을 먼저 이 평면으로 **직교 투영(orthogonal  projection)**.

   * 깊이 차이가 작은 경우 (z \approx  z_0) 로 근사.
3. 그 다음, 이 평면의 점들을 이미지 평면으로 **projective  transform**.

그 결과:

$$x' = \frac{f'}{z_0} x,\quad  y' = \frac{f'}{z_0} y$$

즉, 깊이 $z$ 에 상관없이 **단순한 상수 배$magnification$** 만 남는다.

투영 행렬도 단순해져서:

$$M = \begin{bmatrix} A  &  b0  &  1 \\end{bmatrix}$$

이 경우 3D 점에 대한 투영은:

$$P' = M  P = \begin{bmatrix} m_1m_2m_3 \\end{bmatrix} P = \begin{bmatrix} m_1  Pm_2  P1 \\end{bmatrix} \$$

* 마지막 성분이 항상 1 → **비선형 (1/z)** 가 사라져서 계산이 매우 단순.

## 13.2  Orthographic $Affine$ Projection

* Weak  perspective를 더 단순화:

  * 카메라 중심이 **무한 원점$infinity$** 에 있다고 생각.
  * 모든 광선이 이미지 평면에 **수직**으로 들어온다고 가정.

결과적으로:

$$x' = x,\quad  y' = y$$

즉, 깊이 정보를 완전히 무시하는 **평행 투영**.

* 건축 도면, 산업 설계 등에서 자주 사용.
* 원근감은 없지만, 측정과 기하가 단순해진다.

---

# 14. 전체 흐름 한 번에 정리

3D 점 $P_w$ 가 실제 디지털 이미지의 픽셀 좌표 $p = (u,v$) 로 가는 전체 과정은:

1. **World → Camera**
$$P = \begin{bmatrix} R  &  T0  &  1 \\end{bmatrix} P_w$$
2. **Camera → Ideal  Image Plane (핀홀/렌즈 모델)**
$$(x,y,z) \mapsto \left(f \frac{x}{z}, f\frac{y}{z}\right)$$
3. **Image  Plane → Pixel  coordinates (Intrinsic $K$)**
$$P' = K \begin{bmatrix} R  &  T \\end{bmatrix} P_w$$
4. **Lens  Distortion 보정/적용** (필요하다면)

   * radial / tangential  distortion 모델 적용.

이 4단계를 이해하면,
나중에 **에피폴라 기하, 스테레오, PnP, BA** 같은 CS231A의 나머지 토픽에서
나오는 모든 수식의 “뼈대”가 이미 머릿속에 들어온 거라고 보면 된다.

---

원하면 다음 단계로:

* 예를 들어 **실제 숫자를 넣어서 투영 계산 연습**을 해 보거나,
* OpenCV에서 `cv::calibrateCamera`, `cv::undistort` 가 이 수식과 어떻게 연결되는지,
* 또는 weak  perspective/orthographic 모델이 **multi-view  geometry** 에서 어떻게 쓰이는지

같은 것들을 이어서 같이 볼 수 있어.

---

# 15. 실제 숫자로 계산해보는 카메라 투영 예제

초보자도 단번에 전체 파이프라인을 이해할 수 있도록 완전 단계별로 계산해보겠습니다.

## 📌 설정한 카메라 내부 파라미터 (Intrinsic  matrix)

$$K = \begin{bmatrix} \alpha  &  0  &  c_x  0  &  \beta  &  c_y  0  &  0  &  1 \\end{bmatrix} = \begin{bmatrix} 800  &  0  &  320  0  &  800  &  240  0  &  0  &  1 \\end{bmatrix}$$

* focal length in pixel: $f_x = f_y = 800$
* principal point: $(c_x, c_y) = (320, 240)$

## 📌 카메라 외부 파라미터 (Extrinsic: R, T)

카메라 위치는 world 좌표에서:
* 카메라가 world 원점으로 약간 내려다본다고 가정
* 회전: x축으로 $-20°$
* 번역: $T = (0, 0, 5)^T$

회전행렬:

$$R_x(-20°) = \begin{bmatrix} 1  &  0  &  0  0  &  \cos(-20°)  &  -\sin(-20°)  0  &  \sin(-20°)  &  \cos(-20°) \\end{bmatrix} = \begin{bmatrix} 1  &  0  &  0  0  &  0.94  &  0.34  0  &  -0.34  &  0.94 \\end{bmatrix}$$

## 📌 3D 점 (World coordinates)

$$P_w = (2, 1, 10)$$

## ◆ STEP 1 — World → Camera 변환

$$P_c = R P_w + T$$

계산:

$$R P_w = \begin{bmatrix} 1  &  0  &  0  0  &  0.94  &  0.34  0  &  -0.34  &  0.94 \\end{bmatrix} \begin{bmatrix} 2  1  10 \\end{bmatrix} = \begin{bmatrix} 2  0.94  \cdot  1  +  0.34  \cdot  10  -0.34  \cdot  1  +  0.94  \cdot  10 \\end{bmatrix} = \begin{bmatrix} 2  4.34  9.06 \\end{bmatrix}$$

$$P_c = (2, 4.34, 9.06) + (0, 0, 5) = (2, 4.34, 14.06)$$

## ◆ STEP 2 — Camera → Image plane 투영

핀홀 투영식:

$$x' = f \frac{x}{z}, \qquad  y' = f \frac{y}{z}$$

여기선 $\alpha = \beta = 800$:

$$x' = 800 \cdot \frac{2}{14.06} = 113.8$$

$$y' = 800 \cdot \frac{4.34}{14.06} = 246.7$$

## ◆ STEP  3 — Image  plane → Pixel 좌표로 변환

$$u = x' + c_x, \quad  v = y' + c_y$$

$$u = 113.8 + 320 = 433.8$$

$$v = 246.7 + 240 = 486.7$$

## 👉 최종 이미지 픽셀 좌표:

$$p = (u, v) = (434, 487)$$

---

# 16. OpenCV 함수가 CS231A 수식과 연결되는 방식

CS231A 수식:

$$p = K [R|T] P_w$$

OpenCV는 동일한 모델을 다음 함수로 구현합니다:

## ✔ calibrateCamera()

```cpp  cv::calibrateCamera(objectPoints, imagePoints, imageSize,
                    K, distCoeffs, R, T);
```

**역할**:
* CS231A 식 (12)-(15)에서 $(K, R, T)$를 추정하는 것과 동일
* 실제로 내부적으로 SVD 기반 선형해 + 비선형 최적화를 수행

OpenCV의 `distCoeffs`는 radial/tangential  distortion을 포함합니다:

$$k_1, k_2, p_1, p_2, k_3$$

## ✔ projectPoints()

```cpp  cv::projectPoints(Pw, rvec, tvec, K, distCoeffs, uv);
```

이 함수는 정확히 아래 계산을 수행합니다:

1. $P_c = R  P_w + T$
2. 왜곡(distortion) 적용
3. $x' = X_c/Z_c, \quad  y' = Y_c/Z_c$
4. Pixel 좌표:

$$u = f_x  x' + c_x, \qquad  v = f_y  y' + c_y$$

즉, 우리가 앞에서 손으로 계산한 과정 = `projectPoints`와 동일합니다.

## ✔ undistort(), initUndistortRectifyMap()

왜곡 모델이 있는 경우:
* 입력 픽셀 → 이상적 pinhole 모델의 픽셀로 재변환
* CS231A  PDF의 11장 왜곡 보정 수식 기반

---

# 17. Weak  Perspective / Orthographic 모델이 Multi-View  Geometry에서 쓰이는 곳

## 17.1 인간 포즈 추정 (Human  Pose Estimation)

사람이 카메라에서 **멀리** 있다면:

$$z \approx  z_0 \quad (\text{깊이 변화가 매우 작음})$$

따라서 모든 관절 점이 거의 다음과 같이 보입니다:

$$x' = s  x + t_x, \quad  y' = s  y + t_y$$

→ 단순한 2D 선형 변환 + 스케일
→ 딥러닝에서 3D  pose reconstruction 손실을 단순화할 때 자주 사용됩니다.

## 17.2  Structure-from-Motion (SfM) 초기화

전체 3D 복원을 하기 전:
* 만약 카메라-물체 거리가 멀어서 깊이 변화가 작다면,
* Weak  perspective 모델로 초기값을 구해 두고,
* 이후 bundle  adjustment에서 full  perspective 모델로 refinement

## 17.3  Orthographic 모델의 장점

* 원근감 없음 → 평행 투영
* 계산이 선형
* 얼굴 3D  Morphable Model(3DMM) fitting 초기 단계에서 자주 사용 (Basel  Face Model 등)

## 17.4 정리

| 모델 | 특징 | 사용처 |
|------|------|--------|
| **Perspective** | 원근감 O (1/z) | 일반 카메라 모든 경우 |
| **Weak  perspective** | 깊이 변화 작을 때 approximation | 포즈, SfM 초기화 |
| **Orthographic** | 원근 없음 | 얼굴 모델링, 공업용 설계 |

---

# 18. CS231A  1강 완전 정복 요약

## 📌 1) 핀홀 모델 기본

$$x' = f\frac{x}{z}, \quad  y' = f\frac{y}{z}$$

## 📌 2) Intrinsic  matrix

$$K = \begin{bmatrix} f_x  &  s  &  c_x  0  &  f_y  &  c_y  0  &  0  &  1 \\end{bmatrix}$$

## 📌 3) Extrinsic: World → Camera

$$P_c = R P_w + T$$

## 📌 4) 전체 투영식

$$p = K [R|T] P_w$$

## 📌 5) Distortion 모델

* radial: $k_1  r^2 + k_2  r^4 + k_3  r^6$
* tangential: $p_1, p_2$

## 📌 6) Calibration

* 대응점 $(P_i, p_i)$를 모아서 SVD로 투영행렬 $M = K[R|T]$ 추정
* 이후 분해 → $(K, R, T)$

## 📌 7) Simplified  models

* Weak  perspective
* Orthographic

---

## 참고 자료

- [Stanford  CS231A Course  Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [01-camera-models.pdf](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf)

