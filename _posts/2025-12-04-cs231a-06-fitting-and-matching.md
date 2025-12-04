---
title: "[CS231A] Lecture 06: Fitting and Matching (피팅 및 매칭)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Feature Matching, Fitting, RANSAC]
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
아래는 **CS231A Lecture 6 – Fitting and Matching** 내용을
처음 봐도 이해되도록 스토리랑 수식, 그리고 마지막에 **NumPy / OpenCV / Open3D 연습 코드**까지 묶어서 정리한 거다 

---

# 1. 강의 전체 스토리다

이 강의의 질문은 항상 같다.

> “데이터가 잔뜩 있는데,
> 그 데이터를 **가장 잘 설명하는 모델(직선, 평면, 호모그래피, F 등)** 을
> 어떻게 안정적이고, 아웃라이어에도 튼튼하게, 잘 찾아낼까?”

지금까지는

* 핀홀 카메라 모델,
* 에피폴라 기하,
* 스테레오, SfM 같은 **기하학적 관계**를 배웠다

이제 그 관계들을 실제 데이터에 적용하려면

* **선형/비선형 최적화**,
* **아웃라이어에 강한 추정**,
* **투표 기반 방법**

같은 도구가 필요하다.
Lecture 6이 바로 이 “**fitting & matching toolkit**”을 정리하는 시간이다.

핵심 키워드는 네 가지다.

1. Least Squares / Total Least Squares (SVD)
2. Robust cost (outlier에 덜 휘둘리는 비용함수)
3. RANSAC (Random Sample Consensus)
4. Hough Transform (parameter space에서의 voting)

---

# 2. Ordinary Least Squares – 세로 오차만 보는 직선 맞추기다

가장 기본은 **점들에 직선을 맞추기(line fitting)** 하는 예제다 

2D 점 $(x_i, y_i)$가 $N$개 있다고 하자.
우리는 직선

$$\hat{y}_i = m x_i + b$$

를 찾고 싶다. 여기서 $(m, b)$가 우리가 찾는 파라미터다.

가장 간단한 생각은

> 각 점의 **세로 방향 오차** (y_i - \hat{y}_i) 를 제곱해서
> 전부 더한 값을 최소화하자

라는 것이다.

오차합은 이렇게 쓴다.

$$E = \sum_{i=1}^N (y_i - (m x_i + b))^2$$

이걸 행렬 형태로 쓰면 더 보기 좋다.

* 파라미터 벡터: $w = \begin{bmatrix} m \\ b \end{bmatrix}$
* 입력 행렬:

$$X = \begin{bmatrix} x_1 & 1 \\ \vdots & \vdots \\ x_N & 1 \end{bmatrix}, \quad y = \begin{bmatrix} y_1 \\ \vdots \\ y_N \end{bmatrix}$$

그러면

$$E = | y - X w |^2$$

이 된다 

잔차(residual)는 $r = y - Xw$다.
우리는 $|r|^2 = r^T r$을 최소화하는 $w$를 찾고 싶다.

전개하면

$$|r|^2 = y^T y - 2 y^T X w + w^T X^T X w$$

이제 $w$에 대해 미분해서 0으로 두면 된다.

$$\nabla_w |r|^2 = -2 X^T y + 2 X^T X w = 0$$

따라서

$$X^T X w = X^T y$$

이걸 **normal equation**이라고 부른다.
$X^T X$가 역행렬을 가지면

$$w = (X^T X)^{-1} X^T y$$

로 깔끔하게 풀린다 

이게 우리가 알고 있는 “평균적으로 제일 잘 맞는 직선”이다.

하지만 여기에는 치명적인 문제가 하나 있다.

> **x축에 거의 수직인 (거의 세로인) 직선**에 매우 취약하다

이 경우 기울기 m이 거의 무한대로 가서 수치적으로 매우 불안정해진다.
게다가 이 모델은 **y 방향 오차만** 고려하기 때문에,
x 방향 오차는 완전히 무시한다는 점도 찝찝하다.

그래서 다음 단계로 넘어간다.

---

# 3. Total Least Squares – 직각 거리로 직선 맞추기다

위의 단점 때문에 **$ax + by + c = 0$** 형태의 직선 파라미터화를 사용한다 

* 여기서 $(a,b)$는 직선에 **수직인(normal) 방향**이다.
* 기울기는 $m = -\frac{a}{b}$이고, 이제 $b=0$이면 $m$이 무한대가 되는 것도 자연스럽다.

이제 오차를 "y 방향"이 아니라
**점에서 직선까지의 수직 거리**로 정의한다.

점 $P=(x_i,y_i)$, 직선 $ax + by + c = 0$에서의 거리 $d$는

$$d_i = \frac{|a x_i + b y_i + c|}{\sqrt{a^2 + b^2}}$$

이다 

우리는 이 거리의 제곱합을 최소화하고 싶다.

$$E = \sum_{i=1}^N d_i^2 = \sum_{i=1}^N \frac{(a x_i + b y_i + c)^2}{a^2 + b^2}$$

근데 $(a,b,c)$에 스케일을 곱해도 같은 직선이라
분모를 고정해서 제거하는 게 좋다.

그래서

$$a^2 + b^2 = 1$$

으로 정규화하자고 약속하고

$$E = \sum_{i=1}^N (a x_i + b y_i + c)^2$$

를 최소화하는 문제로 바꾼다.

### 3.1 “최적 직선은 반드시 데이터의 중심을 지난다”는 사실이다

하나 더 중요한 사실이 있다. 강의에서 증명을 보여준다 

데이터의 **평균점(centroid)**

$$\bar{x} = \frac{1}{N}\sum x_i, \quad \bar{y} = \frac{1}{N}\sum y_i$$

을 정의하면,
에러 $E$를 최소로 만드는 직선은 항상 **$(\bar{x},\bar{y})$을 지난다**는 사실이다.

수식적으로는

$$c = -a \bar{x} - b \bar{y}$$

라고 두었을 때 에러가 항상 최소가 된다는 뜻이다 

따라서 우리는 $c$를 없애고,
모든 점을 centroid 기준으로 평행이동한 좌표에서 문제를 풀 수 있다.

$$\tilde{x}_i = x_i - \bar{x}, \quad \tilde{y}_i = y_i - \bar{y}$$

그러면 에러는

$$E = \sum_{i=1}^N (a \tilde{x}_i + b \tilde{y}_i)^2$$

이다.

### 3.2 SVD로 푸는 constrained least squares다

이제 행렬 $X$와 벡터 $w$를 정의한다.

$$X = \begin{bmatrix} \tilde{x}_1 & \tilde{y}_1 \\ \vdots & \vdots \\ \tilde{x}_N & \tilde{y}_N \end{bmatrix}, \quad w = \begin{bmatrix} a \\ b \end{bmatrix}, \quad |w|^2 = 1$$

그러면

$$E = |X w|^2$$

가 된다 

문제는 이제

> $\min_w |Xw|^2$ subject to $|w|=1$

이다.

이건 SVD로 바로 풀 수 있다.

1. $X = U S V^T$를 구한다.
2. $S$는 singular value를 큰 순서로 가진 대각행렬이다.
3. $|Xw|^2 = |S V^T w|^2$이므로

   * $|w|=1$이라는 제약 아래
   * **가장 작은 singular value에 대응하는 방향**으로 $w$를 잡으면
   * $|Xw|$이 최소가 된다.

즉, **$V$의 마지막 컬럼**이 최적의 $w$이다 

정리하면,

* 데이터 중심을 빼고 $X$를 만든다
* SVD($X$)를 구한다
* $V$의 마지막 컬럼이 직선의 **법선 방향 $(a,b)$**이다
* $c$는 $c = -a\bar{x} - b\bar{y}$로 계산한다

이게 바로 **Total Least Squares(line)** 이고,
PCA로 보면

* **가장 큰 분산 방향 = 직선 방향**
* 그에 수직인 방향 = 법선 (a,b)

으로 해석해도 된다.

---

# 4. Least Squares vs Robust Cost – 아웃라이어와의 전쟁이다

지금까지는 항상

$$C(u_i) = u_i^2$$

를 썼다.
여기서 $u_i$는 $i$번째 데이터의 residual이다.

문제는, 이 함수는 **잔차가 커지면 비용이 제곱으로 폭발**한다는 점이다.
그래서 outlier 하나가 있으면 전체 fitting이 그놈에게 끌려가 버린다.

그래서 강의에서는 **robust cost function**을 소개한다 

예시로 든 함수는

$$C(u_i, \sigma) = \frac{u_i^2}{\sigma^2 + u_i^2}$$

이다.

* $u$가 작으면 $C \approx \frac{u^2}{\sigma^2}$ → 거의 제곱 오차처럼 행동한다
* $u$가 매우 크면 $C \to 1$로 포화되기 때문에
  거대한 outlier라도 비용 기여가 최대 1로 제한된다.

$\sigma$를 **scale parameter**라고 부른다.
$\sigma$가 클수록 "중심부의 quadratic 영역"이 넓어져서
outlier를 더 세게 처벌한다.
$\sigma$가 작으면 대부분의 점이 outlier처럼 취급돼서 모델이 이상해질 수 있다 

이런 robust cost는 대개 **비선형**이기 때문에

* closed-form이 없고
* **iterative 방법(예: IRLS, Gauss–Newton)** 으로 최적화한다.

실제로는

1. 먼저 ordinary least squares로 초기값을 구하고
2. robust cost를 넣은 비선형 최적화를 반복해서
   파라미터를 refine하는 식으로 많이 쓴다.

---

# 5. RANSAC – “랜덤 뽑기 + 다수결”로 outlier를 버리는 방법이다

RANSAC(Random Sample Consensus)는
**아웃라이어가 많이 섞여 있어도 모델을 튼튼하게 찾는 방법**이다 

라인 피팅 예제로 생각해보자.

점들이 잔뜩 있고, 그 중 일부만 진짜 직선 위에 있고, 나머지는 완전 엉뚱한 위치에 있는 상황이다.

RANSAC의 기본 루프는 이렇다.

1. **최소 샘플 수만큼 랜덤으로 뽑는다**

   * 직선: 점 2개
   * Fundamental matrix: 대응점 8개 (8-point algorithm)
   * Homography: 대응점 4개 (8 DoF up to scale)

2. **뽑은 점들로 모델을 한 번 피팅**한다

   * 두 점이면 직선 하나가 나온다

3. **전체 데이터에 대해 잔차를 계산**하고

   * residual $r(p_i, w)$가 일정 threshold $\delta$ 이하인 점들만 **inlier**로 친다
   * inlier 집합 $P$의 크기 $|P|$를 기록한다

4. 이 과정을 여러 번 반복하면서

   * **inlier가 가장 많은 모델**을 찾는다

5. 최종적으로

   * 그 inlier들만 모아서
   * least squares / robust cost로 한 번 더 피팅해서
   * 최종 모델을 얻는다.

이 알고리즘의 장점은

* outlier가 얼마든 많이 있어도,
* **운 좋게 inlier만 뽑은 샘플 세트 하나**만 있으면
* 그 한 번으로도 좋은 모델을 건질 수 있다는 점이다.

물론 단점도 있다.

* 반복 횟수 n, threshold δ 등을 **튜닝**해야 한다
* inlier 비율이 너무 낮으면 “운 좋게 뽑을 확률”이 낮아서
  n이 엄청 커져야 한다
* 이론적으로는 “언제 끝낼지 보장된 상한”이 없다
  (실제로는 적당한 n에서 멈춘다)

### 5.1 필요한 반복 횟수를 계산하는 공식이다

그래서 강의에서는
"**성공 확률 $p$ 이상을 보장하려면 몇 번 반복해야 하는가?**"라는 공식을 준다 

* $s$: 한 번 피팅에 필요한 최소 샘플 수 (직선이면 $s=2$, $F$면 8 등)
* $\epsilon$: outlier 비율 (0~1)
* $p$: 우리가 원하는 "성공 확률" (예: 0.99)

한 번 뽑을 때 **전부 inlier만 뽑힐 확률**은

$$(1-\epsilon)^s$$

이다.

반대로, 한 번 뽑을 때 **적어도 하나는 outlier가 들어갈 확률**은

$$1 - (1-\epsilon)^s$$

이다.

$n$번 뽑았을 때 **매번 outlier가 끼는(즉, 한 번도 "올인 inlier 샘플"을 못 뽑는) 확률**은

$$\big(1 - (1-\epsilon)^s\big)^n$$

따라서 반대로, **적어도 한 번은 전부 inlier인 샘플을 뽑을 확률**은

$$p = 1 - \big(1 - (1-\epsilon)^s\big)^n$$

이다. 이걸 $n$에 대해 풀면

$$n = \frac{\log(1-p)}{\log\big(1 - (1-\epsilon)^s\big)}$$

이 된다 

실제로는 ε를 대략 추정해서 이 공식으로 n을 잡고,
실험해 가면서 값을 조정한다.

---

# 6. Hough Transform – 파라미터 공간에서 투표해서 직선 찾기다

마지막 도구는 **Hough Transform**이다.
이것도 일종의 **투표 기반(line fitting)** 이다 

### 6.1 “점 ↔ 직선”의 역할을 바꾸는 발상이다

직선 $y = m x + n$을 데이터에서 찾고 싶다고 하자.

이미지 공간에서 한 점 $(x_i, y_i)$를 지나는 모든 직선을 생각해 보면

$$y_i = m x_i + n \Rightarrow n = -x_i m + y_i$$

이다.
즉, **파라미터 공간 $(m,n)$**에서는
각 데이터 점 $(x_i, y_i)$가 **하나의 직선** $n = -x_i m + y_i$에 해당한다.

여러 점이 같은 직선 위에 있다면

* 파라미터 공간에서 그 점들에 대응하는 여러 직선들이
* **한 점 $(m', n')$에서 교차**하게 된다.

그 교차점이 바로 우리가 찾는 “가장 많은 점들이 동의하는 직선 파라미터”다 

Hough 알고리즘은 이 교차를 **grid + 투표**로 근사한다.

1. 파라미터 공간 $(m,n)$을 일정 범위의 grid로 나눈다.
2. accumulator $A(m,n)=0$ 배열을 만든다.
3. 각 데이터 점 $(x_i,y_i)$에 대해

   * 가능한 $(m,n)$ 조합을 훑으며
   * 식 $n=-x_i m + y_i$를 만족하는 cell에 1표씩 더한다.
4. 모든 점에 대해 완료 후,

   * 투표수가 가장 큰 cell이
   * 가장 많은 점이 동의하는 직선 모델이 된다.

그래서 **각 점이 각 후보 직선 모델에 투표**하는 구조라서
“투표(voting) 기반”이라고 부른다.

### 6.2 (m,n) 파라미터화의 문제와 ρ–θ 표현이다

하지만 m은 -∞~+∞ 범위를 가지므로

* 파라미터 공간이 무한대
* grid를 만들기가 사실상 불가능하다.

그래서 강의에서는 **polar parameterization**을 쓴다 

$$x \cos\theta + y \sin\theta = \rho$$

* $\rho$: 원점에서 직선까지의 최소 거리 (이미지 크기만큼으로 유한)
* $\theta$: 직선의 법선과 x축 사이의 각 (0~π 사이)

이제 파라미터 공간은 $(\rho, \theta)$로

* $\theta$는 0~π
* $\rho$는 $[-\rho_{\max}, \rho_{\max}]$ 범위로
  **유한한 직사각형 영역**이 된다.

Hough voting은 똑같이 진행한다.

1. $(\rho, \theta)$ 공간을 grid로 나눈다.
2. 각 점 $(x_i, y_i)$에 대해

   * 여러 $\theta$ 후보에 대해
   * $\rho = x_i \cos\theta + y_i \sin\theta$를 계산하고
   * 해당 cell에 1표씩 더한다.
3. 가장 큰 누적값을 가진 cell이

   * 그 $(\rho,\theta)$에 해당하는 직선이다.

노이즈가 있으면

* 같은 직선 위에 있는 점들의 sin 곡선들이
* 정확히 한 점에서 교차하지 않고
* 비슷한 근처에 몰리게 된다.

그래서 grid cell 크기 w를 키우면

* 이런 “근처 교차”들을 하나의 cell로 흡수해 줄 수 있다.
  하지만 너무 키우면 서로 다른 직선들이 같은 cell에 섞여서
  정확도가 떨어진다 

결국 Hough도

* cell 크기,
* 파라미터 공간 범위,
* voting threshold

같은 하이퍼파라미터를 잘 조절해야 한다.

---

# 7. NumPy / OpenCV / Open3D로 해볼 수 있는 실습 코드들이다

이제 강의에서 나온 개념을 몸으로 느껴볼 수 있는 코드들을 정리한다.
Python 기준이다.

---

## 7.1 Ordinary Least Squares vs Total Least Squares (NumPy)다

```python
import numpy as np

def ordinary_least_squares_line(points):
    """
    points : (N, 2) array, 각 행은 (x, y)다.
    y = m x + b 형태의 least squares 직선을 구하는 함수다.
    """
    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]

    # X w ≈ y 꼴로 만들기
    X = np.column_stack([x, np.ones_like(x)])  # (N, 2)
    # 최소제곱 해
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    m, b = w
    return m, b

def total_least_squares_line(points):
    """
    points : (N, 2) array
    ax + by + c = 0 형태로 total least squares 직선을 구한다.
    반환 : (a, b, c)
    """
    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]

    # 데이터 중심
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    # 중심을 뺀 좌표 행렬 X (N x 2)
    X = np.column_stack([x - x_bar, y - y_bar])

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # V의 마지막 컬럼이 법선방향 (a, b)다.
    a, b = Vt[-1]

    # 정규화 (a^2 + b^2 = 1)
    norm = np.sqrt(a**2 + b**2)
    a /= norm
    b /= norm

    # c는 중심을 지나는 조건으로 계산한다.
    c = -a * x_bar - b * y_bar
    return a, b, c

# 예시 사용
if __name__ == "__main__":
    # 대략 y ≈ 2x + 1 근처의 noisy한 점들
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 5, 30)
    ys = 2 * xs + 1 + rng.normal(scale=0.3, size=xs.shape)
    pts = np.column_stack([xs, ys])

    m, b = ordinary_least_squares_line(pts)
    print("OLS: y = %.3f x + %.3f" % (m, b))

    a, b_, c = total_least_squares_line(pts)
    print("TLS: %.3f x + %.3f y + %.3f = 0" % (a, b_, c))
```

* 첫 함수는 강의 초반의 **ordinary least squares** 를 구현한 거다.
* 두 번째 함수는 **total least squares** 로,
  SVD와 데이터 중심을 이용해 직각 거리를 최소화하는 직선을 찾는다.

---

## 7.2 RANSAC으로 라인 피팅하기 (NumPy + Matplotlib)다

```python
import numpy as np

def fit_line_from_two_points(p1, p2):
    """
    두 점 p1, p2 ((x, y))가 주어졌을 때
    ax + by + c = 0 형태의 직선계수를 반환한다.
    """
    x1, y1 = p1
    x2, y2 = p2
    # 두 점을 지나는 직선의 법선벡터는 (y1 - y2, x2 - x1)다.
    a = y1 - y2
    b = x2 - x1
    # 한 점을 대입해서 c 구한다.
    c = -(a * x1 + b * y1)
    # 정규화
    norm = np.sqrt(a**2 + b**2)
    if norm == 0:
        return None
    return a/norm, b/norm, c/norm

def point_line_distance(a, b, c, x, y):
    # 거리 공식
    return np.abs(a*x + b*y + c)

def ransac_line(points, n_iters=1000, threshold=0.01, min_inliers=10):
    """
    RANSAC으로 2D 점 집합에 직선을 피팅한다.
    points : (N, 2)
    """
    points = np.asarray(points)
    N = points.shape[0]

    best_inliers = []
    best_model = None

    rng = np.random.default_rng()

    for _ in range(n_iters):
        # 1) 최소 샘플 2개 랜덤 뽑기
        idx = rng.choice(N, size=2, replace=False)
        p1, p2 = points[idx]

        model = fit_line_from_two_points(p1, p2)
        if model is None:
            continue
        a, b, c = model

        # 2) 모든 점에 대해 잔차(거리) 계산
        dists = point_line_distance(a, b, c, points[:, 0], points[:, 1])

        # 3) threshold 이하를 inlier로 둔다
        inliers = np.where(dists < threshold)[0]

        # 4) inlier가 더 많으면 갱신
        if len(inliers) > len(best_inliers) and len(inliers) >= min_inliers:
            best_inliers = inliers
            best_model = model

    # 마지막으로 inlier들만 모아서 total least squares로 refine해도 된다.
    if best_model is not None and len(best_inliers) >= 2:
        refined_model = total_least_squares_line(points[best_inliers])
        return refined_model, best_inliers
    else:
        return None, None
```

이 코드는

* 강의에서 설명한 **RANSAC line fitting** 절차를 그대로 구현한 것이다.
* 마지막에 inlier들의 subset으로 total least squares를 한 번 더 돌려 모델을 다듬는다.

---

## 7.3 OpenCV로 Hough Transform 이용해서 이미지에서 직선 찾기다

OpenCV는 Hough Transform을 거의 그대로 구현해 줘서
강의 내용을 바로 실습해 보기 좋다.

```python
import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread("lines.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 엣지 검출
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 표준 Hough (ρ, θ 공간에서 투표)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)

if lines is not None:
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        # 직선을 이미지 공간으로 다시 그려준다
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # (x0, y0)를 지나는 직선 상의 두 점
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Hough Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

여기서

* `HoughLines` 의 파라미터 (rho, theta)가
  강의에서 설명한 **폴라 파라미터 공간 (ρ, θ)** 에 그대로 해당한다.
* threshold는 accumulator cell의 최소 투표 수다.

---

## 7.4 Open3D로 RANSAC Plane Fitting – 3D 버전이다

강의에서는 2D 직선을 예로 들었지만,
**3D 평면**에 대해서도 RANSAC을 똑같이 쓸 수 있다.

Open3D는 이걸 편하게 제공한다.

```python
import open3d as o3d
import numpy as np

# 예시: 임의의 평면 + 노이즈 + 아웃라이어 점 생성
rng = np.random.default_rng(0)

# z = 0.5 x + 0.2 y + 1 평면 위의 점들
xs = rng.uniform(-1, 1, 500)
ys = rng.uniform(-1, 1, 500)
zs = 0.5 * xs + 0.2 * ys + 1 + rng.normal(scale=0.02, size=xs.shape)
plane_points = np.column_stack([xs, ys, zs])

# 일부 아웃라이어
outliers = rng.uniform(-2, 2, size=(100, 3))

points = np.vstack([plane_points, outliers])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# RANSAC으로 평면 세그멘테이션
distance_threshold = 0.05
ransac_n = 3
num_iterations = 1000

plane_model, inliers = pcd.segment_plane(
    distance_threshold=distance_threshold,
    ransac_n=ransac_n,
    num_iterations=num_iterations
)

a, b, c, d = plane_model
print("Estimated plane: %.3f x + %.3f y + %.3f z + %.3f = 0"
      % (a, b, c, d))

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 빨간색

outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
```

* 이 코드는 3D 점들에서 **RANSAC으로 평면을 추정**하는 예시다.
* 2D에서 직선을 찾는 RANSAC과 개념적으로 완전히 같다.

---

# 8. 마무리 정리다

Lecture 6에서 배운 도구들을 한 문장씩 요약하면 이렇다.

* **Ordinary least squares** :
  선형 모델을 세로 잔차 제곱합으로 빠르게 피팅하는 기본 도구다.

* **Total least squares (SVD)** :
  모델에 수직인 거리를 최소화해서
  기하학적으로 더 자연스러운 피팅을 제공하는 방법이다.

* **Robust cost** :
  잔차가 너무 큰 outlier의 영향을 제한해서
  전체 모델이 망가지지 않도록 하는 비선형 비용함수다.

* **RANSAC** :
  아웃라이어가 매우 많아도
  최소 샘플을 반복적으로 뽑아서
  inlier 다수가 동의하는 모델을 찾는 랜덤/투표 기반 방법이다.

* **Hough Transform** :
  데이터 공간과 파라미터 공간의 역할을 바꿔서
  점들이 **파라미터 공간에서 한 점에 교차하는 모델**을
  accumulator voting으로 찾는 방법이다.

앞에서 준 NumPy / OpenCV / Open3D 예제들을 직접 돌려보면
수식에서만 보던 개념들이
"아 이런 식으로 실제 데이터에 쓰이는구나" 하는 느낌으로
한 번에 정리될 거다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [06-fitting-matching.pdf](https://web.stanford.edu/class/cs231a/course_notes/06-fitting-matching.pdf)
