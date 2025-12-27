---
title: "[CS231A] Lecture 06: Fitting and Matching (피팅 및 매칭)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Fitting, Matching]
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

이 포스트는 Stanford CS231A 강의의 여섯 번째 강의 노트인 "Fitting and Matching"를 정리한 것입니다.

**원본 강의 노트**: [06-fitting-matching.pdf](https://web.stanford.edu/class/cs231a/course_notes/06-fitting-matching.pdf)

<!--more-->

## 1. Overview

**Fitting(피팅)**의 목표는 관측된 데이터를 가장 잘 설명하는 파라미터 모델을 찾는 것입니다. 우리는 데이터와 모델 파라미터의 특정 추정값 사이의 선택된 피팅 오차를 최소화하여 이러한 모델의 최적 파라미터를 얻습니다. 고전적인 예는 주어진 $(x, y)$ 점 집합에 선을 피팅하는 것입니다. 이 수업에서 본 다른 예에는 다른 이미지에서 점 대응 집합 간의 2D 호모그래피 $H$를 계산하거나 8점 알고리즘을 사용하여 fundamental 행렬 $F$를 계산하는 것이 포함됩니다.

## 2. Least-squares

$N$개의 2D 점 $\{(x_i, y_i)\}_{i=1}^N$ 시리즈가 주어지면, **least-squares(최소 제곱)** 피팅 방법은 Figure 1에 설명된 대로 $y$ 차원에서 제곱 오차가 최소화되도록 $y = mx + b$ 형식의 선을 찾으려고 합니다.

구체적으로, 모델 파라미터 $w = \begin{bmatrix} m & b \end{bmatrix}^T$를 찾아 $y_i$와 모델 추정값 $\hat{y}_i = m x_i + b$ 사이의 제곱 잔차의 합을 최소화하려고 합니다. 방정식 (1)에 주어진 대로 잔차를 $y_i - \hat{y}_i$로 정의합니다:

$$E = \sum_{i=1}^N (y_i - \hat{y}_i)^2 \tag{1}$$

$$= \sum_{i=1}^N (y_i - m x_i - b)^2 \tag{2}$$

이것을 행렬 표기법으로 작성할 수 있습니다:

$$E = \sum_{i=1}^N \left(y_i - \begin{bmatrix} x_i & 1 \end{bmatrix} \begin{bmatrix} m \\ b \end{bmatrix}\right)^2 \tag{3}$$

$$= \left\|\begin{bmatrix} y_1 \\ \vdots \\ y_N \end{bmatrix} - \begin{bmatrix} x_1 & 1 \\ \vdots & \vdots \\ x_N & 1 \end{bmatrix} \begin{bmatrix} m \\ b \end{bmatrix}\right\|^2 \tag{4}$$

$$= \|Y - X w\|^2 \tag{5}$$

잔차는 이제 $r = y - X w$이며, $X$가 skinny이고 full rank라고 가정합니다. 잔차의 제곱 노름을 최소화하는 $w$를 찾고자 하며, 다음과 같이 작성할 수 있습니다:

$$\|r\|^2 = r^T r = (y - X w)^T (y - X w) = y^T y - 2 y^T X w + w^T X^T X w \tag{6-8}$$

그런 다음 $w$에 대한 잔차의 기울기를 0과 같게 설정합니다. $X^T X$가 대칭이라는 것을 기억하세요:

$$\nabla_w \|r\|^2 = -2 X^T y + 2 X^T X w = 0 \tag{9-10}$$

이것은 normal equations로 이어집니다:

$$X^T X w = X^T y \tag{11}$$

이제 방정식 (12)에서 $w$에 대한 closed-form 해를 가집니다. $X$가 full rank이므로 $X^T X$는 가역입니다:

$$w = (X^T X)^{-1} X^T y \tag{12}$$

그러나 이 방법은 수직선을 설명하는 점을 피팅할 때 완전히 실패한다는 점에 주목하세요($m$이 정의되지 않음). 이 경우 $m$은 극도로 큰 숫자로 설정되어 수치적으로 불안정한 해로 이어집니다. 이를 수정하기 위해 $ax + by + d = 0$ 형식의 대체 선 공식을 사용할 수 있습니다. $b = 0$으로 설정하여 수직선을 얻을 수 있습니다.

이 선 표현에 대해 생각하는 한 가지 방법은 다음과 같습니다. 선 방향(기울기)은 $\vec{n}$으로 주어집니다. $(x, y) \cdot (a, b) = x a + y b = 0$을 만족하는 $(x, y)$ 집합은 $\vec{n}$에 직교하는 선입니다. 그러나 선은 임의의 위치 $(x_0, y_0)$로 이동할 수도 있으므로 다음을 가집니다:

$$a(x - x_0) + b(y - y_0) = a x + b y - a x_0 - b y_0 = a x + b y + c = 0 \tag{13-15}$$

여기서 $c = -a x_0 - b y_0$입니다. 선의 기울기는 $m = -\frac{a}{b}$이며, 이제 정의되지 않을 수 있습니다.

이전에는 우리의 잔차가 $y$축에만 있었습니다. 그러나 이제 새로운 선 파라미터화가 $x$축과 $y$축 모두의 오차를 설명하므로, 우리의 새로운 오차는 Figure 2에 설명된 대로 제곱 직교 거리의 합입니다.

2D 데이터 점 $P = (x_i, y_i)$와 선 위의 점 $Q = (x, y)$가 주어지면, $P$에서 선까지의 거리는 선에 직교하는 법선 벡터 $\vec{n}$에 대한 $\overrightarrow{QP}$의 투영 길이와 동일합니다. $\overrightarrow{QP} = (x_i - x, y_i - y)$, $\vec{n} = (a, b)$를 가지므로 다음을 제공합니다:

$$d = \frac{|\overrightarrow{QP} \cdot \vec{n}|}{\|\vec{n}\|} = \frac{|a(x_i - x) + b(y_i - y)|}{\sqrt{a^2 + b^2}} = \frac{|a x_i + b y_i + c|}{\sqrt{a^2 + b^2}} \tag{16-18}$$

$Q$가 선 위에 있으므로 $c = -a x - b y$입니다.

우리의 새로운 파라미터 집합은 이제 $w = \begin{bmatrix} a & b & c \end{bmatrix}^T$입니다. 오차를 단순화하기 위해 해를 고유하게 만들고 $\|\vec{n}\|^2 = 1$로 제약하여 분모를 제거하므로 새로운 오차는 다음과 같습니다:

$$E(a, b, x_0, y_0) = \sum_{i=1}^N (a(x_i - x_0) + b(y_i - y_0))^2 = \sum_{i=1}^N (a x_i + b y_i + c)^2 \tag{19-20}$$

여기서 $a^2 + b^2 = 1$입니다. 그러나 이것을 행렬 표기법으로 만드는 것은 제약이 $a, b$에만 있고 $c$가 존재하기 때문에 여전히 까다롭습니다. 더 단순화하기 위해, $E$를 최소화하는 최적 피팅 선이 데이터 중심값 $(\bar{x}, \bar{y})$를 통과해야 한다는 것을 주목합니다. 이것은 다음과 같이 정의됩니다:

$$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \quad \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i \tag{21-22}$$

모든 $\vec{n}$과 가능한 모든 점 집합 $\{(x_i, y_i)\}_{i=1}^N$에 대해, $c = -a \bar{x} - b \bar{y}$로 설정할 때 $E$가 최소화됩니다. 다시 말해, 모든 점 $(x_0, y_0) \in \mathbb{R}^2$에 대해 다음을 가집니다:

$$E(a, b, x_0, y_0) \geq E(a, b, \bar{x}, \bar{y}) \tag{23}$$

이것이 왜 참인지 보기 위해 벡터 $w, z$를 정의하여 $w_i = a(x_i - x_0) + b(y_i - y_0)$ 및 $z_i = a(x_i - \bar{x}) + b(y_i - \bar{y})$로 합니다. 그런 다음 오차를 다음과 같이 작성할 수 있습니다:

$$E(a, b, x_0, y_0) = \|w\|^2, \quad E(a, b, \bar{x}, \bar{y}) = \|z\|^2 \tag{26-27}$$

$w$와 $z$ 사이의 관계는 $w = z + h \mathbf{1}$이며, 여기서 $h = a(\bar{x} - x_0) + b(\bar{y} - y_0) \in \mathbb{R}$이고 $\mathbf{1}$은 모든 1의 벡터입니다. $z$는 $\mathbf{1}$에 직교합니다. 따라서 피타고라스 정리에 의해 $E(a, b, x_0, y_0) = \|w\|^2 = \|z\|^2 + h^2 N \geq \|z\|^2 = E(a, b, \bar{x}, \bar{y})$입니다.

최적 피팅 선이 $(\bar{x}, \bar{y})$를 통과해야 한다는 것을 보여주었으므로, $c = -a \bar{x} - b \bar{y}$로 $c$를 제약할 수 있습니다. 그런 다음 모든 점을 데이터 중심값 주변에 중심화하여 $c$를 제거할 수 있습니다($(x_0, y_0) = (\bar{x}, \bar{y})$로 설정), 이것은 우리가 마지막으로 오차를 행렬 곱으로 공식화할 수 있게 합니다:

$$E = \sum_{i=1}^N (a(x_i - \bar{x}) + b(y_i - \bar{y}))^2 = \left\|\begin{bmatrix} x_1 - \bar{x} & y_1 - \bar{y} \\ \vdots & \vdots \\ x_N - \bar{x} & y_N - \bar{y} \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}\right\|^2 = \|X w\|^2 \tag{37-39}$$

여기서 $w = \begin{bmatrix} a & b \end{bmatrix}^T$이고 $\|w\|^2 = 1$입니다. 이것은 이전 강의에서 본 제약된 최소 제곱 문제입니다. SVD($X$가 full rank)에 의해 다음을 가집니다:

$$X = U S V^T \tag{40}$$

$U \in \mathbb{R}^{N \times M}$, $V^T \in \mathbb{R}^{M \times M}$는 모두 정규직교 행렬이고, $S \in \mathbb{R}^{M \times M}$는 내림차순으로 $X$의 singular value를 포함하는 대각 행렬입니다. 여기서 $M = 2$입니다. $U, V$가 정규직교이므로 $\|U S V^T w\| = \|S V^T w\|$ 및 $\|V^T w\| = \|w\|$임을 알고 있습니다. $v = V^T w$로 설정하면, 새로운이지만 동등한 제약 $\|v\|^2 = 1$로 $\|S V^T w\|$를 최소화할 수 있습니다. $\|S V^T w\| = \|S v\|$는 $S$의 대각선이 내림차순으로 정렬되어 있기 때문에 $v = \begin{bmatrix} 0 & 1 \end{bmatrix}^T$일 때 최소화됩니다. 마지막으로 $w = V V^T w = V v$를 얻으므로 오차를 최소화하는 $w$는 $V$의 마지막 열입니다.

이득 관점에서 해석하면, SVD $X = U S V^T$를 다음과 같이 작성할 수 있습니다:

$$X = \sum_{i=1}^M \sigma_i u_i v_i^T \tag{43}$$

$v_1, \ldots, v_M$은 $V$의 열이고, $\sigma_i$는 $S = \text{diag}(\sigma_1, \ldots, \sigma_M)$의 대각 값이며, $u_1, \ldots, u_M$은 $U$의 열입니다. SVD로 $w$를 곱하는 것, $U S V^T w$는 먼저 입력 방향 $v_1, \ldots, v_M$을 따라 $w$의 구성 요소를 계산하고, 구성 요소를 $\sigma_i$로 스케일링한 다음 출력 방향 $u_1, \ldots, u_M$을 따라 재구성하는 것으로 볼 수 있습니다. $V^T w$는 $V$의 각 열을 따라 $w$의 투영을 제공합니다($\|v_i\|^2 = 1$을 기억하세요). 마찬가지로 $U w'$는 출력 방향의 선형 조합으로 볼 수 있습니다, $u_1 w'_1 + \cdots + u_M w'_M$. 따라서 $\|w\|^2 = 1$ 제약 하에서 $X w$를 최소화하는 $w$를 찾는 것은 단순히 출력 벡터의 크기를 최소화하는 입력 방향을 선택하는 것이며, 이것은 $V$의 마지막 열입니다.

실제로, least-squares 피팅은 노이즈 데이터를 잘 처리하지만 이상치에 취약합니다. $i$번째 데이터 점에 대한 잔차를 $u_i = a x_i + b y_i + c$로 작성하고 비용을 $C(u_i)$로 하면, 우리의 오차는 다음과 같이 일반화될 수 있습니다:

$$E = \sum_{i=1}^N C(u_i) \tag{44}$$

지금까지 사용해온 제곱 오차 $C(u_i) = u_i^2$의 이차 성장(Figure 4의 왼쪽에 설명됨)은 큰 잔차 $u_i$를 가진 이상치가 비용 최소값에 과도한 영향을 미친다는 것을 의미합니다.

큰 잔차(이상치)를 덜 처벌하기 위해 robust cost function(Figure 4의 오른쪽 절반)을 사용할 수 있습니다:

$$C(u_i, \sigma) = \frac{u_i^2}{\sigma^2 + u_i^2} \tag{45}$$

잔차 $u_i$가 클 때, 비용 $C$는 1로 포화되어 비용에 대한 기여가 제한되지만, $u$가 작을 때 비용 함수는 제곱 오차와 유사합니다. 그러나 이제 **scale parameter(스케일 파라미터)** $\sigma$를 선택해야 합니다. $\sigma$는 잠재적 이상치에 얼마나 많은 가중치가 주어지는지를 제어하며, Figure 5에 설명되어 있습니다. 큰 $\sigma$는 중심의 이차 곡선을 넓혀 다른 점에 비해 이상치를 더 많이 처벌합니다(원래 제곱 오차 함수와 유사). 작은 $\sigma$는 이차 곡선을 좁혀 이상치를 덜 처벌합니다. $\sigma$가 너무 작으면 대부분의 잔차가 이상치가 아닐 때도 이상치로 처리되어 나쁜 피팅으로 이어집니다. $\sigma$가 너무 크면 robust cost function의 이점을 얻지 못하고 least-squares 피팅으로 끝납니다.

robust cost function은 비선형이므로 반복 방법으로 최적화됩니다. 실제로 closed-form least-squares 해는 종종 시작점으로 사용되며, 그 다음 robust 비선형 비용 함수로 파라미터를 반복적으로 피팅합니다.

## 3. RANSAC

**RANSAC**이라고 불리는 또 다른 피팅 방법은 random sample consensus를 의미하며, 이상치와 누락된 데이터에 대해 robust하도록 설계되었습니다. RANSAC을 사용하여 선 피팅을 수행하는 것을 시연하지만, 많은 다른 피팅 맥락으로 일반화됩니다.

다시 $N$개의 2D 점 $X = \{(x_i, y_i)\}_{i=1}^N$ 시리즈가 있으며, Figure 6에 점으로 설명된 대로 선에 피팅하고 싶습니다. RANSAC의 첫 번째 단계는 모델을 피팅하는 데 필요한 최소 점 수를 무작위로 선택하는 것입니다. 선은 최소 두 점이 필요하므로 녹색의 두 점을 선택합니다. fundamental 행렬 $F$를 추정하는 경우 8점 알고리즘을 사용하기 위해 8개의 대응을 선택해야 합니다. 호모그래피 $H \in \mathbb{R}^{3 \times 3}$를 계산하려면 스케일까지 8개의 자유도를 다루기 위해 4개의 대응이 필요합니다(각 대응에 대해 두 개의 $x, y$ 좌표가 있음).

RANSAC의 두 번째 단계는 무작위 샘플 집합에 모델을 피팅하는 것입니다. 여기서 녹색의 두 점이 피팅됩니다(즉, 그들 사이에 선이 그려짐)하여 검은색 선을 얻습니다. 세 번째 단계는 피팅된 모델을 사용하여 전체 데이터셋에서 inlier 집합을 계산하는 것입니다. 모델 파라미터 $w$가 주어지면, inlier 집합을 $P = \{(x_i, y_i) | r(p = (x_i, y_i), w) < \delta\}$로 정의합니다. 여기서 $r$은 데이터 점과 모델 사이의 잔차이고 $\delta$는 임의의 임계값입니다. 여기서 inlier 집합은 녹색과 파란색 점으로 표현됩니다. inlier 집합의 크기 $|P|$는 전체 점 집합 중 피팅된 모델에 동의하는 양을 나타냅니다. $P$와 함께 이상치 집합도 얻을 수 있으며, $O = X \setminus P$로 정의됩니다. 여기서 이상치 집합은 빨간색 점으로 구성됩니다. inlier 집합의 크기가 최대화될 때까지 매번 새로운 무작위 샘플로 유한한 반복 횟수 $M$에 대해 이러한 단계를 반복합니다.

모든 가능한 샘플을 시도하는 것은 불필요합니다. 주어진 $s$(모델을 피팅하는 데 필요한 최소 점 수)와 $\epsilon \in [0, 1]$(이상치의 비율)에 대해 확률 $p$로 "실제" 이상치가 없는 inlier 집합을 가진 최소 하나의 무작위 샘플을 보장하기 위해 반복 횟수 $n$을 추정할 수 있습니다. 이제 $s$ 점의 단일 무작위 샘플이 모든 inlier를 포함할 확률은 $(1 - \epsilon)^s$이므로, 단일 무작위 샘플이 최소 하나의 이상치를 포함할 확률은 $1 - (1 - \epsilon)^s$입니다. 이제 모든 $n$ 샘플이 최소 하나의 이상치를 포함할 확률은 $(1 - (1 - \epsilon)^s)^n$이므로, $n$ 샘플 중 최소 하나가 어떤 이상치도 포함하지 않을 확률은 $p = 1 - (1 - (1 - \epsilon)^s)^n$입니다. 이제 $n$을 다음과 같이 유도할 수 있습니다:

$$1 - p = (1 - (1 - \epsilon)^s)^n \tag{46}$$

$$\log(1 - p) = n \log(1 - (1 - \epsilon)^s) \tag{47}$$

$$n = \frac{\log(1 - p)}{\log(1 - (1 - \epsilon)^s)} \tag{48}$$

## 4. Hough transform

**Hough transform(허프 변환)**으로 알려진 또 다른 피팅 방법을 소개합니다. 이것은 또 다른 투표 절차입니다.

다시 이미지에서 점 $\{(x_i, y_i)\}_{i=1}^N$ 시리즈에 $y = m' x + n'$ 형식의 선을 피팅하고 싶습니다. Figure 7의 왼쪽 절반에 설명된 대로입니다. 이 선을 찾기 위해 Figure 7의 오른쪽 절반에 설명된 dual parameter 또는 **Hough space(허프 공간)**를 고려합니다. 이미지 공간의 점 $(x_i, y_i)$(선 $y = m x_i + n$ 위에 있음)는 $n = -x_i m + y_i$로 정의된 파라미터 공간의 선이 됩니다. 마찬가지로 파라미터 공간의 점 $(m, n)$은 $y = m x + n$으로 주어진 이미지 공간의 선입니다.

파라미터 공간의 선 $n = -x_i m + y_i$는 이미지 공간의 점 $(x_i, y_i)$를 통과하는 이미지 공간의 모든 다른 가능한 선을 나타낸다는 것을 봅니다. 따라서 이미지 공간에서 이미지 점 $(x_1, y_1)$과 $(x_2, y_2)$ 모두에 맞는 선을 찾기 위해 두 점을 모두 Hough 공간의 선과 연관시키고 교차점 $(m', n')$을 찾습니다. Hough 공간의 이 점은 이미지 공간의 두 점을 통과하는 이미지 공간의 선을 나타냅니다. 실제로 우리는 파라미터 공간에 대해 너비 $w$의 정사각형 셀의 이산 그리드로 Hough 공간을 나눕니다. 우리는 $(m, n)$에 중심을 둔 모든 $w \times w$ 셀에 대해 카운트 그리드를 유지하며, 초기에 모든 $(m, n)$에 대해 $A(m, n) = 0$으로 표시합니다. 이미지 공간의 모든 데이터 점 $(x_i, y_i)$에 대해 $n = -x_i m + y_i$를 만족하는 모든 $(m, n)$을 찾고 카운트를 1씩 증가시킵니다. 모든 데이터 점에 대해 이 작업을 수행한 후, Hough 공간에서 가장 높은 카운트를 가진 점 $(m, n)$은 이미지 공간에서 피팅된 선을 나타냅니다. 이것이 투표 절차인 이유를 이제 봅니다: 각 데이터 요소 $(x_i, y_i)$는 이미지 공간의 각 후보 선 $(m, n)$에 대해 최대 하나의 투표를 기여할 수 있습니다.

그러나 기존 파라미터화에는 주요 제한 사항이 있습니다. least-squares에서 논의한 대로, 이미지 공간에서 선의 기울기는 무한대 $-\infty < m < \infty$입니다. 이것은 Hough 투표를 계산 및 메모리 집약적인 알고리즘으로 만듭니다. 왜냐하면 카운트를 유지하는 파라미터 공간의 크기에 제한이 없기 때문입니다. 이를 해결하기 위해 Figure 8에 설명된 대로 선의 극좌표 파라미터화로 전환합니다:

$$x \cos(\theta) + y \sin(\theta) = \rho \tag{49}$$

Figure 8의 왼쪽 절반은 $\rho$가 원점에서 선까지의 최소 거리(이미지 크기 또는 데이터셋의 임의의 두 점 사이의 최대 거리로 제한됨)이고 $\theta$는 $x$축과 선의 법선 벡터 사이의 각도(0과 $\pi$ 사이로 제한됨)임을 보여줍니다. 이전과 동일한 Hough 투표 절차를 사용하지만, 이제 특정 $(x_i, y_i)$를 통과하는 Cartesian 공간의 모든 가능한 선은 Hough 공간에서 사인파 프로필에 해당하며, Figure 8의 오른쪽 절반에 설명된 대로입니다.

실제로 노이즈 데이터 점은 이미지 공간에서 동일한 선 위의 점에 해당하는 Hough 공간의 사인파 프로필이 Hough 공간의 동일한 점에서 반드시 교차하지 않을 수 있다는 것을 의미합니다. 이를 해결하기 위해 그리드 셀의 너비 $w$를 증가시킬 수 있으며, 이것은 불완전한 교차에 대한 허용 오차를 증가시킵니다. 이것은 노이즈 데이터를 처리하는 데 도움이 될 수 있지만, 조정해야 하는 또 다른 파라미터를 도입합니다. 작은 그리드 크기는 노이즈로 인해 이미지 공간 선을 놓칠 수 있는 반면, 큰 그리드 크기는 다른 선을 병합하고 셀 내의 모든 $\rho, \theta$가 가능한 선이기 때문에 추정 정확도를 감소시킬 수 있습니다.

또한 이미지 공간의 균일한 노이즈로 인해 Hough 공간에 가짜 피크가 있고 적절한 모델에 대한 명확한 합의가 없을 것입니다.

---

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [06-fitting-matching.pdf](https://web.stanford.edu/class/cs231a/course_notes/06-fitting-matching.pdf)
