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

이 포스트는 Stanford CS231A 강의의 두 번째 강의 노트인 "Single View Metrology"를 정리한 것입니다.

**원본 강의 노트**: [02-single-view-metrology.pdf](https://web.stanford.edu/class/cs231a/course_notes/02-single-view-metrology.pdf)

<!--more-->

## 1. Introduction

이전 강의에서는 3D world point들을 실제 real world로 tranform하는것을 내/외부 파라미터를 사용해서 진행을 했음. 이 과장가운데에서는 우리는 calibration을 배웠으며 이를 통해서 이미지를 사용하여 내/외부 파리미터를 추론하는 것까지 이전 시간에서 정리를 하였음. 이번에는 단일 이미지만 있을 경우 카메라의 속성(내/외부)을 알고 있따면, 3D world로 복원을 할수 있을지 알아보고자한다. 

## 2. Transformations in 2D

Image에 대해서 이해하기 위해서,  2D 공간의 다양한 transform에 대해 알고 있어야 한다. 

**Isometric transformations(등거리 변환)**은 거리를 보존하는 변환입니다. 가장 기본적인 형태에서, 등거리는 회전 $R$과 평행 이동 $t$로 설명될 수 있습니다. 따라서 수학적으로 다음과 같이 정의됩니다:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

여기서 $\begin{bmatrix} x' & y' & 1 \end{bmatrix}^T$는 등거리 변환 이후의 점임.

**Similarity transformations(유사 변환)**은 모양을 보존하는 변환입니다. 직관적으로, 등거리 변환이 할 수 있는 모든 것에 **스케일링을 더한 것**. 수학적으로 다음과 같이 표시됩니다:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} SR & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad S = \begin{bmatrix} s & 0 \\ 0 & s \end{bmatrix}$$

모양을 보존하기 때문에 길이와 각도의 비율도 보존합니다. 모든 등거리 변환은 $s = 1$일 때 유사 변환의 특정 형태. 그러나 그 역은 성립하지 않습니다.

**Affine transformations(아핀 변환)**은 점, 직선, 평행성을 보존하는 변환입니다. 어떤 벡터 $v$에 대해, 아핀 변환 $T$는 다음과 같이 정의됩니다:

$$T(v) = Av + t$$

여기서 $A$는 $\mathbb{R}^n$의 선형 변환입니다. 동차 좌표에서 아핀 변환은 종종 다음과 같이 작성됩니다:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} A & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

위 방정식에서 모든 유사 변환(따라서 등거리)이 아핀의 특정 경우임을 쉽게 알 수 있습니다.

**Projective transformations** 또는 **homographies(호모그래피)**는 직선을 직선으로 매핑하지만 평행성을 반드시 보존하지 않는 모든 변환입니다. 동차 좌표에서 투영 변환은 다음과 같이 표현됩니다:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} A & t \\ v & b \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

이 형식은 $v$의 추가로 추가 자유도가 추가되기 때문에 아핀 변환의 더 일반화된 형태임을 볼 수 있습니다.

평행성을 보존하지 않지만, 투영 변환은 직선을 직선으로 매핑하기 때문에 점의 collinearity을 보존합니다. 또한 네 개의 collinear 점의 **cross ratio(교차비)**가 투영 변환 하에서 invarient을 증명합니다. 교차비는 직선 위의 네 점 $P_1, P_2, P_3, P_4$를 취하여 다음을 계산합니다:

$$\text{cross ratio} = \frac{\|P_3 - P_1\| \|P_4 - P_2\|}{\|P_3 - P_2\| \|P_4 - P_1\|} \tag{1}$$

## 3. Points and Lines at Infinity

직선은 이미지에서 구조를 결정하는 데 중요하므로, 2D와 3D 모두에서의 정의를 아는 것이 필수적입니다. 2D의 직선은 동차 벡터 $\ell = \begin{bmatrix} a & b & c \end{bmatrix}^T$로 표현할 수 있습니다. 비율 $-\frac{a}{b}$는 직선의 기울기를 포착하고, 비율 $-\frac{c}{b}$는 y 절편을 포착합니다. 공식적으로, 2D 직선은 다음과 같이 정의됩니다:

$$\forall p = \begin{bmatrix} x \\ y \end{bmatrix} \in \ell, \quad \begin{bmatrix} a & b & c \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = 0 \tag{2}$$

일반적으로, 두 직선 $\ell$과 $\ell'$은 점 $x$에서 교차합니다. 이 점은 $\ell$과 $\ell'$ 사이의 외적으로 정의됩니다.

**증명**: 두 교차하는 직선 $\ell$과 $\ell'$이 주어지면, 교차점 $x$는 두 직선 $\ell$과 $\ell'$ 모두에 있어야 합니다. 따라서 $x^T \ell = 0$이고 $x^T \ell' = 0$입니다. $x = \ell \times \ell'$로 설정하면, 외적의 정의에 의해 벡터 $x$는 두 벡터 $\ell$과 $\ell'$에 모두 직교합니다. 직교성의 정의에 의해 $x^T \ell = 0$이고 $x^T \ell' = 0$입니다. 따라서 $x$의 이 정의는 제약 조건을 만족합니다.

평행한 두 직선의 경우에서는 다음과 같이 되어진다. 일반적으론느 교좀이 없다고 생각이 되어지지만 정의에 의해서는 무한대에서는 교차한다고 적을수 있다. 그 유는 homogeneous 좌표계에서 무한대의 점은 $\begin{bmatrix} x & y & 0 \end{bmatrix}^T$로 적성되고 이는 마지막에 있는 0이 유클리드 좌표게의 모든 좌표를 나누게 되어진다. 이 경우 좌표가 0이므로 무한대의 점을 달성합니다. 따라서 동차 좌표는 평행선의 경우에도 교차점을 결정하는 좋은 공식을 제공합니다.

평핸한 두 직선 $\ell$과 $\ell'$을 두고, 두 직선이 평행할 때, 기울기가 같으므로 $\frac{a}{b} = \frac{a'}{b'}$입니다. 동차 좌표를 사용하여 교차점을 계산하면 다음을 확인합니다:

$$\ell \times \ell' \propto \begin{bmatrix} b \\ -a \\ 0 \end{bmatrix} = x_\infty \tag{3}$$

따라서 평핸한 두 직선이 무한대에서 교차한는 정의가 되어진다. 두 평행선의 무한대에서의 교차점은 **ideal point(이상점)**이라고도 합니다. 무한대의 점의 흥미로운 속성 중 하나는 **동일한 기울기 $-\frac{a}{b}$를 가진 모든 평행선이 아래에 표시된 대로 이상점을 통과**한다는 것입니다:

$$\ell^T x_\infty = \begin{bmatrix} a & b & c \end{bmatrix} \begin{bmatrix} b \\ -a \\ 0 \end{bmatrix} = 0 \tag{4}$$

무한대의 점의 개념은 **lines at infinity(무한대의 직선)**를 정의하도록 확장될 수 있습니다. 두개이상의 평행한 두 직석에서느 평행선 쌍은 무한대의 점 $\{x_{\infty,1}, \ldots, x_{\infty,n}\}$에서 교차되어짐으로 정의할수 있다. 이러한 모든 무한대의 점을 통과하는 직선 $\ell_\infty$는 $\forall i, \ell_\infty^T x_{\infty,i} = 0$을 만족해야 합니다. 이것은 $\ell_\infty = \begin{bmatrix} 0 & 0 & c \end{bmatrix}^T$를 의미합니다. $c$는 임의의 값이므로, 단순히 $\ell_\infty = \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}^T$로 정의할 수 있습니다.

일반적인 투영 변환 $H$를 무한대의 점 $p_\infty$에 적용하면 어떻게 될까요?

$$p' = H p_\infty = \begin{bmatrix} A & t \\ v & b \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ 0 \end{bmatrix} = \begin{bmatrix} p'_x \\ p'_y \\ p'_z \end{bmatrix} \tag{5}$$

$p'$의 마지막 요소가 0이 아닐 수 있음을 주목하세요. 이것은 투영 변환이 일반적으로 무한대의 점을 더 이상 무한대가 아닌 점으로 매핑한다는 것을 시사합니다. 그러나 이것은 무한대의 점을 무한대의 점으로 매핑하는 아핀 변환의 경우는 아닙니다:

$$p' = H p_\infty = \begin{bmatrix} A & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ 0 \end{bmatrix} = \begin{bmatrix} p'_x \\ p'_y \\ 0 \end{bmatrix} \tag{6}$$

이제 투영 변환 $H$를 직선 $\ell$에 적용하여 새로운 직선 $\ell'$을 얻겠습니다. 직선을 통과하는 모든 점 $x$는 속성 $x^T \ell = 0$을 만족해야 합니다. 변환된 공간에서 직선이 여전히 직선으로 매핑된다는 것을 알고 있으므로, $x'^T \ell' = 0$을 의미합니다. 항등 속성을 사용하여 다음을 얻을 수 있습니다:

$$x^T I \ell = x^T H^T H^{-T} \ell = 0$$

투영 변환을 직선에 적용하면 모든 점도 변환되므로 $x' = H x$를 얻습니다. 따라서 $x^T H^T H^{-T} \ell = x'^T \ell'$을 얻고, 직선의 투영 변환은 $\ell' = H^{-T} \ell$임을 찾습니다. 무한대의 점에 대한 관찰과 유사하게, 무한대의 직선의 투영 변환이 반드시 다른 무한대의 직선으로 매핑되는 것은 아닙니다. 또한 아핀 변환은 여전히 무한대의 직선을 무한대의 직선으로 매핑합니다.

## 4. Vanishing Points and Lines

지금까지는 무한대에서 2D 직선과 검에 대해서 설명을 하였고, 이제는 homogenouse coordination에서늬 3D에 대한 eqivalent컴셉에 대해서 알아보자. 

3D 세계에서 이제 평면의 개념을 소개받습니다. 평면을 벡터 $\begin{bmatrix} a & b & c & d \end{bmatrix}^T$로 표현할 수 있으며, 여기서 $(a, b, c)$는 법선 벡터를 형성하고 $d$는 그 법선 벡터 방향에서 원점에서 평면까지의 거리입니다. 공식적으로, 평면은 다음을 만족하는 모든 점 $x$로 정의됩니다:

$$x^T \begin{bmatrix} a \\ b \\ c \\ d \end{bmatrix} = a x_1 + b x_2 + c x_3 + d = 0 \tag{7}$$

3D의 직선은 두 평면의 교차로 정의됩니다. 4개의 자유도(정의된 절편 위치와 세 차원 각각의 기울기)를 가지기 때문에, 3D 공간에서 잘 표현하기 어려움.

그러나 점은 2D에서와 유사하게 3D에서 정의됩니다. 3D에서 무한대의 점은 다시 3D에서 평행선의 교차점으로 정의됩니다. 또한 이러한 무한대의 점 중 하나인 $x_\infty$에 투영 변환을 적용하면, 더 이상 동차 좌표에서 무한대가 아닌 이미지 평면의 점 $p_\infty$를 얻습니다. 이 점 $p_\infty$는 **vanishing point(소실점)**로 알려져 있습니다. 그러나 소실점으로 무엇을 할 수 있을까요?

3D의 평행선, 이미지에서의 해당 소실점, 그리고 카메라 파라미터 $K, R, T$ 사이의 유용한 관계를 유도할 수 있습니다. 카메라 참조 시스템에서 3D 평행선 집합의 방향을 $d = (a, b, c)$로 정의하겠습니다. 이러한 직선들은 무한대의 점에서 교차하고, 그러한 점의 이미지로의 투영은 소실점 $v$를 반환하며, 이는 다음과 같이 정의됩니다:

$$v = K d \tag{8}$$

이 방정식은 방향 $d$를 추출하도록 다시 쓸 수 있습니다:

$$d = \frac{K^{-1} v}{\|K^{-1} v\|} \tag{9}$$

평면 $\Pi$를 평행선의 상위 집합으로 고려하면, 각 평행선 집합은 무한대의 점에서 교차합니다. 그러한 무한대의 점 집합을 통과하는 직선은 $\Pi$와 관련된 무한대의 직선 $\ell_\infty$입니다. 무한대의 직선은 또한 두 평행 평면이 교차하는 직선으로 정의됩니다. $\ell_\infty$의 이미지 평면으로의 투영 변환은 더 이상 무한대의 직선이 아니며 **vanishing line(소실선)** 또는 **horizon line(수평선)** $\ell_{\text{horiz}}$라고 불립니다. 수평선은 이미지에서 해당 소실점을 통과하는 직선입니다. 수평선은 다음과 같이 계산할 수 있습니다:

$$\ell_{\text{horiz}} = H_P^{-T} \ell_\infty \tag{10}$$

수평선의 개념은 Figure 2에서 지면의 선들이 이미지 좌표에서 평행하지 않지만, 3D 세계에서 평행하다는 자연스러운 이해를 가지고 있습니다.

또한 수평선은 세계에 대한 유용한 속성을 계산할 수 있게 해줍니다. 예를 들어, 3D에서 평면의 법선 $n$과 이미지에서의 해당 수평선 $\ell_{\text{horiz}}$ 사이의 흥미로운 관계를 유도할 수 있습니다:

$$n = K^T \ell_{\text{horiz}} \tag{11}$$

이것은 평면과 관련된 수평선을 인식할 수 있고 카메라가 보정되어 있다면, 그 평면의 방향을 추정할 수 있다는 것을 의미합니다. 소실점과 직선을 관련시키는 마지막 속성을 소개하기 전에, 먼저 **plane at infinity(무한대의 평면)** $\Pi_\infty$를 정의해야 합니다. 이 평면은 2개 이상의 소실선 집합으로 정의되며 동차 좌표에서 벡터 $\begin{bmatrix} 0 & 0 & 0 & 1 \end{bmatrix}^T$로 설명됩니다.

우리가 소개하는 마지막 속성은 3D의 직선과 평면을 이미지 평면에서의 해당 소실점과 직선과 관련시킵니다. 3D에서 두 쌍의 평행선이 방향 $d_1$과 $d_2$를 가지고 무한대의 점 $x_{\infty,1}$과 $x_{\infty,2}$와 관련되어 있다고 가정합니다. $v_1$과 $v_2$를 해당 소실점이라고 합니다. 그러면 코사인 법칙을 사용하여 $d_1$과 $d_2$ 사이의 각도 $\theta$가 다음과 같이 주어진다는 것을 찾습니다:

$$\cos \theta = \frac{d_1 \cdot d_2}{\|d_1\| \|d_2\|} = \frac{v_1^T \omega v_2}{\sqrt{v_1^T \omega v_1} \sqrt{v_2^T \omega v_2}} \tag{12}$$

여기서 $\omega = (K K^T)^{-1}$입니다.

이 아이디어를 3D 평면 경우로 더 확장할 수 있으며, 여기서 3D의 다른 평면을 관련시키고 싶습니다. 모든 평면에 대해 관련된 소실선 $\ell_{\text{horiz}}$와 그 법선 $K^T \ell_{\text{horiz}}$를 계산할 수 있다는 것을 기억하세요. 따라서 각 평면의 법선 벡터 $n_1$과 $n_2$ 사이의 각도를 계산하여 두 평면 사이의 각도 $\theta$를 결정할 수 있습니다. 각각 소실선 $\ell_1$과 $\ell_2$를 가진 두 평면 사이의 각도 $\theta$를 유도합니다:

$$\cos \theta = \frac{n_1 \cdot n_2}{\|n_1\| \|n_2\|} = \frac{\ell_1^T \omega^{-1} \ell_2}{\sqrt{\ell_1^T \omega^{-1} \ell_1} \sqrt{\ell_2^T \omega^{-1} \ell_2}} \tag{13}$$

## 5. A Single View Metrology Example

3D 세계의 이미지에서 두 평면을 식별할 수 있다고 가정합니다. 또한 이러한 각 평면에서 평행선 쌍을 식별할 수 있다고 가정합니다. 이것은 이미지에서 두 소실점 $v_1$과 $v_2$를 추정할 수 있게 해줍니다. 마지막으로, 이러한 평면이 3D에서 수직이라는 것을 알고 있다고 가정합니다. 이 경우 방정식 12에서 $v_1^T \omega v_2 = 0$임을 알고 있습니다.

그러나 $\omega$는 카메라 행렬 $K$에 의존하며, 이는 이 시점에서 잠재적으로 알려지지 않았습니다. 따라서 이 두 소실점을 아는 것이 카메라 파라미터를 정확하게 추정하기에 충분할까요? $K$가 5개의 자유도를 가지고 $v_1^T \omega v_2 = 0$이 하나의 제약만 제공한다는 것을 고려하면, $K$를 계산하기에 충분한 정보가 없습니다. 상호 직교하는 또 다른 평면에 대한 또 다른 소실점 $v_3$를 찾을 수 있다면 어떨까요? 그러면 $v_1^T \omega v_2 = v_1^T \omega v_3 = v_2^T \omega v_3 = 0$임을 알고 있습니다. 각 쌍이 제약을 제공하므로, $K$를 계산하는 데 필요한 5개 제약 중 3개만 얻습니다. 그러나 카메라가 zero-skew와 정사각형 픽셀을 가진다는 가정을 하면, 필요한 추가 두 제약을 추가할 수 있습니다. 이러한 가정에 의해, $\omega$는 다음 형식을 취한다는 것을 알고 있습니다:

$$\omega = \begin{bmatrix} \omega_1 & 0 & \omega_4 \\ 0 & \omega_1 & \omega_5 \\ \omega_4 & \omega_5 & \omega_6 \end{bmatrix} \tag{14}$$

주의 깊게 주목하면, $\omega$의 정의에 4개의 변수가 있습니다. 그러나 스케일까지만 $\omega$를 알 수 있으므로 실제 변수의 양이 3개로 줄어들어 해결할 수 있습니다. $\omega$를 얻으면 Cholesky 분해를 사용하여 $K$를 계산할 수 있습니다. 따라서 단일 이미지를 사용하여 카메라를 보정했습니다!

$K$가 알려지면 장면의 3D 기하학을 재구성할 수 있습니다. 예를 들어, 위에서 식별한 모든 평면의 방향을 계산할 수 있습니다. 따라서 단일 이미지를 사용하여 그것이 포착하는 장면에 대한 풍부한 정보를 쉽게 밝힐 수 있습니다.

---

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [02-single-view-metrology.pdf](https://web.stanford.edu/class/cs231a/course_notes/02-single-view-metrology.pdf)
