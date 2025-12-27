---
title: "[CS231A] Lecture 09: Optical and Scene Flow (광학 및 장면 흐름)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Optical Flow, Scene Flow]
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

## 1. Overview

비디오가 주어지면, **optical flow(광학 흐름)**은 카메라(관찰자)와 장면(객체, 표면, 가장자리) 사이의 상대적 움직임으로 인한 각 픽셀의 명백한 움직임을 설명하는 2D 벡터 필드로 정의됩니다. 카메라 또는 장면 또는 둘 다 움직일 수 있습니다. Figure 1은 반시계 방향으로 회전하는 파리를 보여줍니다(파리의 관점에서). 장면이 정적이지만, 명백한 움직임의 2D 광학 흐름은 원점 주위의 반대(시계 방향) 방향의 회전을 나타냅니다.

### 1.1 Motion field

광학 흐름은 **motion field(움직임 필드)**와 혼동되어서는 안 됩니다. 이것은 장면의 점에 대한 3D 움직임 벡터의 관찰자의 이미지 평면으로의 투영을 설명하는 2D 벡터 필드입니다. Figure 2는 간단한 2D 경우(위에서 내려다본 3D 장면을 상상)에서의 움직임 필드를 설명합니다. 2D 객체 점 $P_o$는 관찰자 $O$에 의해 보이는 이미지 평면에서 1D 점 $P_i$로 투영됩니다. 객체 점 $P_o$가 $V_o \cdot dt$(움직임 벡터라고 함)만큼 변위되면, 해당 투영된 1D 점은 $V_i \cdot dt$만큼 이동합니다. 여기서 1D 움직임 필드는 이미지 평면에 위치한 모든 $i$에 대한 모든 속도 값 $V_i$로 구성됩니다.

3D 장면으로 일반화하면, 픽셀 $(x, y)$에 대한 움직임 필드는 다음과 같이 주어집니다:

$$\begin{pmatrix} u \\ v \end{pmatrix} = \begin{pmatrix} \frac{dx}{dt} \\ \frac{dy}{dt} \end{pmatrix} = M x' \tag{1}$$

여기서 $x' = \begin{bmatrix} \frac{dx}{dt} & \frac{dy}{dt} & \frac{dz}{dt} \end{bmatrix}^T$는 3D 점의 움직임을 나타내고 $M \in \mathbb{R}^{2 \times 3}$는 3D 점 위치에 대한 픽셀 변위의 편미분을 포함합니다.

움직임 필드는 이미지 평면에 투영된 3D 움직임의 이상적인 2D 표현입니다. 이것은 직접 관찰할 수 없는 "ground truth"입니다. 우리는 노이즈 관측(비디오)으로부터 광학 흐름(명백한 움직임)을 추정할 수 있습니다. 광학 흐름이 항상 움직임 필드와 동일한 것은 아니라는 점에 주목하는 것이 중요합니다. 예를 들어, 고정된 광원을 가진 균일하게 회전하는 구는 광학 흐름이 없지만 0이 아닌 움직임 필드를 가집니다. 대조적으로, 주변을 움직이는 광원을 가진 고정된 균일 구는 0이 아닌 광학 흐름을 가지지만 0 움직임 필드를 가집니다. 이 두 경우는 Figure 3에 설명되어 있습니다.

## 2. Computing the optical flow

비디오를 시간에 걸쳐 캡처된 프레임의 순서 시퀀스로 정의합니다. $I(x, y, t)$는 공간과 시간 모두의 함수이며, 시간 $t$에서 프레임의 픽셀 $(x, y)$의 강도를 나타냅니다. **dense optical flow(조밀한 광학 흐름)**에서, 모든 시간 $t$와 모든 픽셀 $(x, y)$에 대해, 각각 $u(x, y, t) = \frac{\Delta x}{\Delta t}$ 및 $v(x, y, t) = \frac{\Delta y}{\Delta t}$로 주어진 $x$축과 $y$축 모두에서 픽셀의 명백한 속도를 계산하고 싶습니다. 각 픽셀에 대한 광학 흐름 벡터는 $u = \begin{bmatrix} u & v \end{bmatrix}^T$로 주어집니다. 다음 섹션에서 least-squares를 사용하여 다른 픽셀 패치에 대해 $u$를 독립적으로 해결하는 semi-local 접근 방식을 사용하는 Lucas-Kanade 방법을 설명합니다.

**brightness constancy assumption(밝기 일정성 가정)**으로부터, 이미지 평면에서 동일한 객체의 명백한 강도가 다른 프레임에 걸쳐 변하지 않는다고 가정할 수 있습니다. 이것은 시간 $t$에서 $t + \Delta t$ 사이에 $x$ 및 $y$ 방향으로 $\Delta x$ 및 $\Delta y$만큼 이동한 픽셀에 대해 방정식 (2)로 표현됩니다:

$$I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t) \tag{2}$$

일반적인 단순화는 연속 프레임을 사용하도록 $\Delta t = 1$을 사용하는 것입니다. 그러면 속도는 변위와 동일합니다: $u = \Delta x$ 및 $v = \Delta y$. 그런 다음 강의 슬라이드에서와 같이 $I(x, y, t) = I(x + u, y + v, t + 1)$을 얻을 수 있습니다.

다음으로, **small motion assumption(작은 움직임 가정)**에 의해 프레임 간 움직임 $(\Delta x, \Delta y)$가 작다고 가정합니다. 이것은 방정식 (3)에 설명된 대로 1차 테일러 급수 전개로 $I$를 선형화할 수 있게 합니다:

$$I(x + \Delta x, y + \Delta y, t + \Delta t) = I(x, y, t) + \frac{\partial I}{\partial x} \Delta x + \frac{\partial I}{\partial y} \Delta y + \frac{\partial I}{\partial t} \Delta t + \ldots$$

$$\approx I(x, y, t) + \frac{\partial I}{\partial x} \Delta x + \frac{\partial I}{\partial y} \Delta y + \frac{\partial I}{\partial t} \Delta t \tag{3}$$

$\ldots$는 테일러 급수 전개에서의 고차 항을 나타내며, 다음 줄에서 잘라냅니다. 방정식 (3)의 결과를 방정식 (2)에 대입하면 광학 흐름 제약 방정식에 도달합니다:

$$0 = \frac{\partial I}{\partial x} \Delta x + \frac{\partial I}{\partial y} \Delta y + \frac{\partial I}{\partial t} \Delta t = \frac{\partial I}{\partial x} \frac{\Delta x}{\Delta t} + \frac{\partial I}{\partial y} \frac{\Delta y}{\Delta t} + \frac{\partial I}{\partial t} = I_x u + I_y v + I_t \tag{4}$$

$I_x, I_y, I_t$는 각각 두 공간 미분과 시간 미분의 약칭입니다:

$$-I_t = I_x u + I_y v = \nabla I^T u = \nabla I \cdot \vec{u} \tag{5}$$

우리는 이것을 $A x = b$ 형식의 선형 시스템으로 인식합니다. $\nabla I = \begin{bmatrix} I_x & I_y \end{bmatrix}^T \in \mathbb{R}^{2 \times 1}$는 강도의 공간 기울기를 나타내고, $\vec{u} \in \mathbb{R}^{2 \times 1}$는 해결하려는 흐름 벡터입니다.

그러나 $\nabla I$가 fat 행렬이기 때문에, 이것은 두 미지수 $u, v$에 대해 단일 제약 방정식만 있기 때문에 미결정 시스템입니다. 이 제약은 우리를 두 개에서 하나의 자유도로만 가져갑니다. $(u, v)$ 해 집합은 방정식 (5)에 의해 주어지고 Figure 4에 설명된 선 위에 있어야 합니다. 구체적으로, 광학 흐름 제약은 우리에게 **normal flow(법선 흐름)**만 줄 수 있습니다: 공간 기울기 $\nabla I$ 방향을 따라 $u$의 구성 요소입니다. Figure 5는 예제 이미지의 $x$ 및 $y$ 방향에서의 공간 이미지 기울기를 시각화합니다. 높은 기울기 크기가 이미지 가장자리에 해당함을 볼 수 있습니다.

두 벡터 $a$와 $b$ 사이의 내적은 $a \cdot b = \|a\| \|b\| \cos(\theta)$이며, 여기서 $\theta$는 두 벡터 사이의 각도입니다. $a$를 $b$에 대한 투영은 양변을 $\|b\|$로 나누어 얻을 수 있으며, Figure 6에 설명되어 있습니다.

따라서 방정식 (6)은 광학 흐름 제약이 법선 흐름인 $\nabla I$에 대한 $u$의 투영을 어떻게 제공하는지 보여줍니다:

$$\frac{\nabla I}{\|\nabla I\|} \cdot u = -\frac{I_t}{\|\nabla I\|} \tag{6}$$

Figure 4는 $u$에 대한 해 공간을 플롯합니다. 그러나 이것 위에 이미지 공간을 시각화할 수도 있습니다. 법선 흐름(해 공간)은 주어진 픽셀에서 이미지 강도의 공간 기울기와 같은 방향입니다. 해 집합(선)은 공간 기울기에 직교하기 때문에 이미지 공간에서 가장자리와 같은 방향을 가집니다.

따라서 $\nabla I$ 방향에서 $u$의 구성 요소(법선 흐름)를 알고 있지만, 가장자리를 따라 $u$의 구성 요소를 알지 못합니다. 이것은 이 구성 요소가 가질 수 있는 모든 가능한 값을 보여주는 해 집합으로 표현됩니다. $u$의 가능한 값을 찾기 위해, 먼저 법선 흐름의 크기에 해당하는 거리만큼 공간 기울기 방향으로 이동한 다음, 가장자리 방향으로 임의의(알려지지 않은) 거리를 이동할 수 있으며, 이것은 우리의 하나의 자유도를 나타냅니다.

가장자리 방향에서 픽셀 움직임의 크기를 알지 못하는 이 문제는 **aperture problem(조리개 문제)**로도 알려져 있으며, Figure 7에 설명되어 있습니다. 파란색 사각형은 불투명하며 중심에 작은 조리개를 형성하는 사각형 구멍이 있습니다. 밝은 회색 직사각형은 우리의 움직이는 객체입니다. 왼쪽에서 회색 직사각형이 파란색 사각형 앞에 있습니다. 오른쪽에서 회색 직사각형이 파란색 사각형 뒤에 있습니다. 두 경우 모두 직사각형이 동시에 아래로 그리고 오른쪽으로 이동합니다(동시에). 빨간색 화살표는 두 경우 모두에서 인지된 움직임 방향을 나타냅니다. 왼쪽의 첫 번째 경우에서, 직사각형이 파란색 사각형에 의해 차단되지 않으므로 그 진정한 움직임 방향을 명확하게 인지할 수 있습니다. 그러나 오른쪽의 경우, 직사각형이 파란색 사각형 뒤에 있어 가려져 있습니다: 직사각형의 움직임은 왼쪽과 같은 방향이지만, 우리는 오른쪽으로 직접 움직임만 인지합니다. 이것은 우리가 이전에 유도한 것과 일치합니다. 우리는 직사각형이 공간 기울기 방향(여기서는 $x$축)으로 얼마나 이동했는지만 알고 있지만, 이미지 가장자리 방향(여기서는 $y$축)으로 얼마나 이동했는지는 알지 못합니다.

이 문제를 완화하기 위해 **spatial smoothness assumption(공간 평활성 가정)**을 사용할 수 있습니다: 인접한 점들은 장면에서 동일한 표면에 속하므로 동일한 광학 흐름 $u$를 공유합니다. 현재 픽셀 주변에 $N \times N$ 이웃을 정의하면, $N^2$ 픽셀 각각에 대해 제약 $\nabla I(p_i)^T u = -I_t(p_i)$를 얻습니다. 여기서 $p_i = \begin{bmatrix} x_i & y_i \end{bmatrix}^T$는 $i$번째 픽셀의 위치입니다. $N^2 > 2$라고 가정하면, 이제 다음 방정식 시스템을 가집니다:

$$A u = b$$

$$\begin{pmatrix} I_x(p_1) & I_y(p_1) \\ \vdots & \vdots \\ I_x(p_{N^2}) & I_y(p_{N^2}) \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix} = -\begin{pmatrix} I_t(p_1) \\ \vdots \\ I_t(p_{N^2}) \end{pmatrix} \tag{7}$$

그런 다음 오차 $\|A u - b\|$를 최소화하여 이 과결정 시스템의 최소 제곱 해를 찾습니다. 방정식 (7)의 양변에 $A^T$를 곱하여 $u_{ls} = (A^T A)^{-1} A^T b$입니다. 다음 normal equations를 검사하여 이 시스템의 해결 가능성을 고려합니다:

$$A^T A u = A^T b$$

$$\begin{pmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{pmatrix} u = -\begin{pmatrix} \sum I_x I_t \\ \sum I_y I_t \end{pmatrix} \tag{8}$$

합은 이웃의 각 픽셀에 대해 수행됩니다. 시스템이 해결 가능하려면, 노이즈의 존재로 인해 $A^T A$가 작아서는 안 됩니다. 예를 들어, 낮은 텍스처 영역은 작은 기울기 크기를 가져 작은 고유값과 쉽게 손상된 $u$ 값으로 이어집니다. $A^T A$는 또한 측정 오차에 대해 robust하도록 잘 조건화되어야 합니다. 왜냐하면 $A^T b$가 이미지 강도 데이터로부터 계산된 모든 기울기에 의존하기 때문입니다. 예를 들어, 가장자리에서 모든 큰 기울기가 같은 방향을 가리키면, 하나의 고유값이 다른 것보다 훨씬 크게 됩니다(큰 조건 수). $u$를 모호 없이 해결할 수 있는 좋은 영역은 방향적으로 다양한 공간 구조를 가진 높은 텍스처 영역입니다. 다양한 방향에서 큰 기울기를 가질 것이며, 잘 조건화된 $A^T A$로 이어집니다.

방향적으로 다양한 구조가 $u$를 해결하는 데 도움이 되는 이유를 더 직관적으로 보기 위해 Figure 8을 고려하세요. 여기서 두 가지 경우가 있습니다. 회색 원은 최적화하고 있는 픽셀 이웃을 나타냅니다. 점선은 이전 시간 단계에서의 객체를 나타내고, 실선은 현재 시간 단계에서의 객체를 나타냅니다. 이것은 객체의 진정한 움직임을 묘사합니다. (c)에서 우리는 다시 조리개 문제가 작동하는 것을 봅니다. 단일 법선 흐름 벡터(녹색)가 있으므로 제약 선(또한 녹색)을 따라 $u$(파란색)에 대한 수많은 가능한 해가 있습니다. 그러나 (b)에서 우리는 모서리 형태로 다양한 공간 구조를 가지며, 이것을 다른 방향을 가진 두 가장자리로 볼 수 있습니다. 이제 두 법선 흐름 벡터와 두 제약 선(하나는 빨간색, 다른 하나는 녹색)이 있습니다. 두 제약 선을 모두 만족하는 흐름 벡터 $u$에 대한 단 하나의 가능한 해가 있으며, 파란색으로 표시됩니다. 따라서 다양한 공간 구조를 포함할 가능성을 높이기 위해 이웃 크기 $N$을 증가시키는 것이 좋습니다. 그러나 움직임 경계를 넘어 다른 표면에 속하는 픽셀을 포함할 수 있기 때문에 트레이드오프가 있습니다. 따라서 $u$가 전체 이웃에 걸쳐 일정하지 않을 가능성이 점점 더 커집니다.

실제로 이 전통적인 공식의 Lucas-Kanade는 큰 카메라 움직임, 가림 및 앞서 언급한 가정 집합의 위반으로 인해 robust하지 않습니다.

---

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [09-optical-flow.pdf](https://web.stanford.edu/class/cs231a/course_notes/09-optical-flow.pdf)
