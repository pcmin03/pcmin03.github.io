---
title: "[CS231A] Lecture 03: Epipolar Geometry (에피폴라 기하학)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Epipolar Geometry, Stereo Vision]
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

이 포스트는 Stanford CS231A 강의의 세 번째 강의 노트인 "Epipolar Geometry"를 정리한 것입니다.

**원본 강의 노트**: [03-epipolar-geometry.pdf](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)

<!--more-->

아래는 **CS231A Lecture 3 – Epipolar Geometry**를
**스토리텔링+수식+실습 코드(OpenCV, Open3D)**까지 한 번에 정리한 노트이다. 

전부 처음 본다 해도, 차근차근 따라가면 이해되도록 구성했다.

---
아래는 **CS231A Lecture 3 – Epipolar Geometry**를
**스토리텔링+수식+실습 코드(OpenCV, Open3D)**까지 한 번에 정리한 노트이다. 

전부 처음 본다 해도, 차근차근 따라가면 이해되도록 구성했다.

---

# 1. 왜 Epipolar Geometry가 필요한지에 대한 이야기다

Lecture 2까지는 “**한 장의 이미지로 3D를 얼마나 알 수 있나**”를 다뤘다.
하지만 결국 한 장으로는 **깊이(Depth)** 를 완전히 복원할 수 없다는 한계가 있다.

강의 맨 앞의 피사의 사탑 사진을 떠올리면 된다.
사람이 탑을 “받치고 있는 것처럼” 보이는 사진이지만,
실제 3D 세계에서는 전혀 다른 위치에 서 있는 사람과 탑이 우연히 2D에서 겹친 것뿐이다. 

한 장만 보면:

* 사람 손이 탑에 닿는지
* 사람이 탑보다 앞에 있는지 뒤에 있는지

이런 건 **확실하게 알 수 없다**.

하지만 **카메라를 옆으로 한 발짝 옮겨서 두 번째 사진**을 찍는 순간,
그리 어렵지 않게 “아, 손은 사실 탑보다 훨씬 앞에 있고, 그냥 겹쳐 보였던 거구나”라는 걸 알게 된다.

이때 두 카메라, 그 사이의 3D점, 그리고 각 이미지에서의 픽셀 위치를 묶어주는 기하학이 바로
**Epipolar Geometry**다.

---

# 2. Epipolar Geometry의 기본 그림이다

PDF의 Figure 2·3을 떠올리면 된다. 

* 두 카메라 중심: $O_1$, $O_2$
* 두 카메라를 잇는 직선: **baseline**
* 3D 점: $P$
* 각 이미지에서의 투영: $p$ (왼쪽), $p'$ (오른쪽)
* 두 카메라 중심과 3D점 $P$가 정의하는 평면: **epipolar plane**
* 이 평면이 각 이미지 평면과 만나는 직선: **epipolar lines**
* baseline이 각 이미지 평면과 만나는 점: **epipole** ($e$, $e'$)

핵심 이야기는 이렇다:

1. 3D 점 $P$와 두 카메라 중심 $(O_1, O_2)$는 하나의 평면(epipolar plane)을 이룬다.
2. 이 평면이 각 이미지 평면을 자르면 두 개의 직선이 생긴다.
   → 왼쪽 이미지의 epipolar line, 오른쪽 이미지의 epipolar line이다.
3. 3D 점 $P$의 두 이미지 투영 $(p, p')$는 **각각 자신의 epipolar line 위에 반드시 놓인다**.

그래서,

> 한 이미지에서 한 점 $p$를 찍으면,
> 다른 이미지에서는 "어디든 있는 게 아니라, 특정 epipolar line 위 어딘가에 있다"는
> 강한 제약을 얻는다.

이 제약이 **stereo matching, 3D 재구성, SLAM**의 기반이 된다.

---

## 2.1 평행한 두 카메라의 특별한 경우다

Figure 4를 보면 두 이미지 평면이 서로 평행한 경우가 나온다. 

* 두 카메라가 baseline 방향으로만 이동하고,
* 이미지 평면은 서로 평행하다.

이때는:

* epipole $(e, e')$가 **무한대(infinity)** 로 간다.
* epipolar line들이 모두 **수평 방향**으로 정렬된다.

그래서 스테레오 카메라(예: 좌우 카메라)에서 보통

* **두 이미지가 수평으로만 disparity를 가지도록** 맞추는 것이 바로
  이 “epipoles at infinity” 상태에 해당한다.

이 상태가 되도록 이미지들을 맞추는 작업이
뒤에서 나오는 **image rectification**이다.

---

# 3. Essential Matrix 이야기다

이제 “수식”으로 epipolar geometry를 다루기 위해
**Essential Matrix (E)** 를 도입한다. 

---

## 3.1 셋팅: Canonical camera인 경우다

가장 단순한 경우를 먼저 본다:

* 두 카메라 내부 파라미터:
  $$K = K' = I$$
* 즉, 이미지 좌표가 이미 normalized camera coordinates라고 가정한다.

이럴 때 projection matrix는

$$M = [I|0],\quad M' = [R^T|-R^T T]$$

여기서:

* $R$: 두 번째 카메라가 첫 번째 카메라에 대해 갖는 회전
* $T$: 첫 번째 카메라에서 두 번째 카메라까지의 translation

---

## 3.2 Epipolar plane에서의 관계다

두 번째 카메라에서 점 $p'$는 세계 기준으로 $(R p' + T)$ 방향에 있다.

* 벡터 $T$: baseline 방향
* 벡터 $(R p' + T)$: 두 번째 카메라에서 3D점 방향

이 둘은 같은 epipolar plane 안에 있으므로,
이 둘의 외적 $T \times (R p')$는 **epipolar plane의 법선 벡터**가 된다.

3D점의 첫 번째 이미지 투영 $p$도 그 평면 위에 있으므로,
법선과의 내적이 0이어야 한다:

$$p^T [T \times (R p')] = 0$$

---

## 3.3 Cross product를 행렬로 표현한다

크로스 프로덕트는 다음과 같은 **skew-symmetric matrix**로 표현할 수 있다:

$$a \times b = [a_\times] b, \quad [a_\times] = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0 \end{bmatrix}$$

즉,

$$T \times (R p') = [T_\times] R p'$$

이를 위 식에 넣으면

$$p^T [T_\times] R p' = 0$$

여기서

$$E = [T_\times] R$$

로 두면

$$p^T E p' = 0$$

이게 바로 **Essential Matrix**의 정의다.

---

## 3.4 Essential Matrix의 의미와 성질이다

* 크기 3×3 행렬이다.
* DoF는 5개다. (R 3자유도 + T방향 2자유도; 스케일은 중요하지 않다)
* **Rank 2**인 특이 행렬이다.

그리고 epipolar line을 구하는 데 바로 쓴다:

* 첫 번째 이미지의 점 $p$에 대응하는 두 번째 이미지의 epipolar line:
  $$\ell' = E^T p$$
* 두 번째 이미지의 점 $p'$에 대응하는 첫 번째 이미지의 epipolar line:
  $$\ell = E p'$$

---

# 4. Fundamental Matrix 이야기다

현실의 카메라는 canonical이 아니므로
픽셀 좌표에서 바로 epipolar constraint를 쓰고 싶다.

이때 필요한 것이 **Fundamental Matrix (F)**다. 

---

## 4.1 Canonical로 정규화한 후 다시 되돌리는 이야기다

지금은

* 실제 픽셀 좌표: $(p, p')$
* 정규화된 camera coordinates:
  $$p_c = K^{-1} p, \quad p_c' = K'^{-1} p'$$

canonical case에서는

$$p_c^T [T_\times] R p_c' = 0$$

여기에 다시 $p_c = K^{-1}p$를 넣으면:

$$p^T K^{-T} [T_\times] R K'^{-1} p' = 0$$

여기서

$$F = K'^{-T} [T_\times] R K^{-1}$$

로 정의하면

$$p^T F p' = 0$$

이게 픽셀 좌표에서의 **기본 epipolar constraint**다.

---

## 4.2 Fundamental vs Essential 비교다

* $E = [T_\times] R$
  * normalized camera coords에서만 사용 가능하다.
  * DoF = 5
* $F = K'^{-T} [T_\times] R K^{-1}$
  * 그냥 픽셀 좌표에서 바로 쓸 수 있다.
  * DoF = 7

실무에서 가장 많이 사용하는 것은 $F$다.
왜냐하면 카메라의 $K$를 몰라도
두 이미지에서 대응점들만 있으면 $F$를 직접 추정할 수 있기 때문이다.

---

# 5. Eight-Point Algorithm 이야기다

이제 “**카메라 내부·외부 파라미터 아무 것도 모른다**”고 가정하고,
두 이미지에서 대응점만 주어졌을 때
**Fundamental matrix (F)**를 어떻게 구하는지 설명한다. 

---

## 5.1 한 쌍의 대응점이 주는 식이다

대응점 $p_i = (u_i, v_i, 1)$, $p_i' = (u_i', v_i', 1)$가 있을 때

$$p_i^T F p_i' = 0$$

$F$의 원소를

$$F = \begin{bmatrix} F_{11} & F_{12} & F_{13} \\ F_{21} & F_{22} & F_{23} \\ F_{31} & F_{32} & F_{33} \end{bmatrix}$$

이라고 쓰면, 위 식은

$$[u_i u_i', v_i u_i', u_i', u_i v_i', v_i v_i', v_i', u_i, v_i, 1] \cdot f = 0$$

형태가 된다. 여기서 $f$는 $F$의 9개 원소를 펼친 벡터다.

즉, **한 쌍의 대응점 → f에 대한 1개의 선형식**을 준다.

F는 scale까지 포함해 8개의 자유도를 갖기 때문에
**최소 8쌍**의 대응점이 필요하다.

---

## 5.2 전체 선형 시스템이다

$N \geq 8$ 쌍의 대응점에 대해 쌓으면

$$W f = 0$$

* $W$: $N \times 9$ 행렬
* $f$: $9 \times 1$ 벡터

이건 **homogeneous linear system**이다.
SVD로 최소 특이값에 해당하는 eigenvector를 찾으면
최소제곱 sense에서의 해 $\hat{F}$를 얻는다.

---

## 5.3 Rank-2 제약을 다시 강제하는 단계다

이렇게 구한 $\hat{F}$는 일반적으로 full rank(3)다.
하지만 실제 Fundamental matrix는 **rank 2**여야 한다.

그래서 다시 SVD를 한다:

$$\hat{F} = U \Sigma V^T$$

여기서 $\Sigma = \mathrm{diag}(\sigma_1,\sigma_2,\sigma_3)$
중 가장 작은 $\sigma_3$를 0으로 만들어

$$F = U \mathrm{diag}(\sigma_1,\sigma_2,0) V^T$$

로 다시 정의한다.

이렇게 하면 epipolar geometry의 성질을 만족하는
“가장 가까운 rank-2 행렬”을 얻는다.

---

## 5.4 Normalized Eight-Point Algorithm 이야기다

실제 픽셀 값은 수백~수천 단위이기 때문에,
W가 **ill-conditioned**가 되기 쉽다. 그러면 SVD 결과가 수치적으로 불안정하다. 

해결책은 간단하다:

1. 각 이미지의 점들을 **평균이 (0,0)** 이 되도록 평행이동한다.
2. 평균 제곱 거리가 2가 되도록 스케일링한다.

즉, 각 이미지마다

$$q_i = T p_i, \quad q_i' = T' p_i'$$

와 같은 정규화 행렬 $(T, T')$를 만든다.

이제 $(q_i, q_i')$를 가지고 동일한 Eight-Point Algorithm을 돌려
정규화된 Fundamental matrix $F_q$를 구한다.

마지막에 다시 역변환을 적용한다:

$$F = T'^T F_q T$$

이 방법이 **Normalized Eight-Point Algorithm**이고,
실무에서 표준으로 쓰이는 방법이다.

---

# 6. Image Rectification(영상 정렬) 이야기다

이제 강의 후반부에 나오는 “이미지 직렬화”를 다룬다. 

Rectification의 목표는 간단하다:

> 두 이미지를 적절한 **homography (H_1, H_2)** 로 변환해서
> epipolar line이 **수평**이 되도록 만드는 것이다.

이렇게 되면:

* 한 이미지에서 한 점의 대응점을 찾을 때
  **같은 row(같은 v좌표)** 만 샅샅이 찾으면 된다.
* stereo matching이 훨씬 쉬워진다.

---

## 6.1 Parallel camera case에서 직관을 얻는 이야기다

두 카메라가

* 같은 K
* 회전 없음 (R = I)
* x축 방향으로만 이동 (T = (T_x, 0, 0))

이라고 하면 Essential matrix는

$$E = [T_\times]R = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & -T_x \\ 0 & T_x & 0 \end{bmatrix}$$

이를 이용해 epipolar line을 계산하면,
모든 epipolar line의 방향이 **수평**이 된다는 걸 볼 수 있다. 

이 상태가 바로 **rectified stereo**다.

Rectification은 “임의의 두 이미지”를
위와 같은 **평행 카메라 상태인 척 보이도록** 만드는 과정이다.

---

## 6.2 F만 알고 있을 때 H1, H2를 만드는 큰 흐름이다

실제로는 K, R, T를 모르고,
단지 대응점들로부터 F만 알고 있다고 하자. 

1. **F 추정**

   * Normalized Eight-Point Algorithm으로 F를 구한다.

2. **모든 대응점에 대한 epipolar lines 계산**

   * $\ell_i' = F p_i$, $\ell_i = F^T p_i'$

3. **epipole $e$, $e'$ 추정**

   * epipole은 "모든 epipolar line이 지나는 점"이므로
   * 각 이미지에서
     $$\begin{bmatrix} \ell_1^T \\ \vdots \\ \ell_n^T \end{bmatrix} e = 0$$
     를 SVD로 풀어 얻는다.

4. **두 epipole을 "수평 무한대"로 보내는 homography 설계**

   * 두 번째 이미지의 epipole $e'$을 $(f, 0, 0)$에 보내는 $H_2$를 만든다.

     * 이미지 중심을 원점으로 옮기는 translation $T$
     * epipole을 x축 위로 올리는 rotation $R$
     * x방향 무한대로 보내는 projective transform $G$
     * $H_2 = T^{-1} G R T$

5. **첫 번째 이미지용 H1 찾기**

   * 이론적으로는 $H_1 = H_A H_2 M$ 꼴이 되며,
   * $(M, H_A)$는 $F$, $e$, 대응점들을 이용해 least-squares로 찾는다.
   * 구현은 조금 복잡하지만, OpenCV가 이미 **`stereoRectifyUncalibrated`**로 구현해 놨다.

결과적으로,

$$p^{rect} = H_1 p, \quad p'^{rect} = H_2 p'$$

가 되며, 두 rectified 이미지에서 epipolar line은
모두 수평에 가깝게 정렬된다.

---

# 7. 실전: OpenCV로 Epipolar Geometry 실습 코드이다

이제 이론을 실제 코드로 연결해 본다.

아래 코드는 다음을 한다:

1. 왼쪽/오른쪽 이미지 읽기
2. ORB 특징점 검출, 매칭
3. `cv2.findFundamentalMat` 으로 F 추정 (RANSAC 포함)
4. F로 epipolar lines를 계산해 두 이미지 위에 그리기
5. `cv2.stereoRectifyUncalibrated` 으로 rectification homography H1, H2 추정
6. `cv2.warpPerspective` 로 rectified 이미지 생성

---

## 7.1 OpenCV Python 코드 예제다

```python
import cv2
import numpy as np

# 1. 좌우 이미지 로드한다
img1 = cv2.imread("left.jpg")   # 왼쪽 이미지
img2 = cv2.imread("right.jpg")  # 오른쪽 이미지

if img1 is None or img2 is None:
    raise RuntimeError("이미지를 찾을 수 없다. left.jpg / right.jpg 경로를 확인해야 한다.")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. 특징점 검출 및 매칭한다 (ORB + BFMatcher)
orb = cv2.ORB_create(2000)
kps1, des1 = orb.detectAndCompute(gray1, None)
kps2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)

# 상위 N개만 사용한다
N = 500
matches = matches[:N]

pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

# 3. Fundamental matrix F를 RANSAC으로 추정한다
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

print("Estimated F =\n", F)

# RANSAC으로 inlier만 남긴다
pts1_in = pts1[mask.ravel() == 1]
pts2_in = pts2[mask.ravel() == 1]

print("Inliers:", len(pts1_in), "/", len(pts1))

# 4. Epipolar line을 그리는 함수이다
def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    # img1의 점들에 대응되는 img2의 epipolar lines
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    img1_c = img1.copy()
    img2_c = img2.copy()
    h2, w2 = img2_c.shape[:2]

    for r, p1, p2 in zip(lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        a, b, c = r
        # l: ax + by + c = 0 에 대해 x=0, x=w-1에서 y 계산
        x0, y0 = 0, int(-c / b) if abs(b) > 1e-6 else 0
        x1, y1 = w2-1, int(-(c + a*(w2-1)) / b) if abs(b) > 1e-6 else h2-1
        cv2.line(img2_c, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1_c, tuple(np.int32(p1)), 5, color, -1)
        cv2.circle(img2_c, tuple(np.int32(p2)), 5, color, -1)

    return img1_c, img2_c

epi1, epi2 = draw_epipolar_lines(img1, img2, pts1_in, pts2_in, F)

cv2.imshow("Left with points", epi1)
cv2.imshow("Right with epipolar lines", epi2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Rectification homography를 구한다 (uncalibrated rectification)
h, w = gray1.shape
retval, H1, H2 = cv2.stereoRectifyUncalibrated(
    pts1_in, pts2_in, F, imgSize=(w, h)
)

print("Rectification success:", retval)

if retval:
    # 6. Rectified 이미지 생성한다
    rect1 = cv2.warpPerspective(img1, H1, (w, h))
    rect2 = cv2.warpPerspective(img2, H2, (w, h))

    cv2.imshow("Rectified Left", rect1)
    cv2.imshow("Rectified Right", rect2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

이 코드를 돌려보면:

* 첫 번째 창에서는 왼쪽 이미지 위에 매칭된 점들이 찍힌다.
* 두 번째 창에서는 각 점에 대응되는 epipolar line이 오른쪽 이미지 위에 그려진다.
* rectified 이미지에서는
  **동일한 물체가 거의 같은 y좌표(수평선)에서만 옮겨져 있는** 모습을 볼 수 있다.

이게 바로 이론에서 말한
“epipolar line이 수평이 되도록 만드는 rectification”의 구현이다.

---

# 8. Open3D + OpenCV로 3D 점 구해서 시각화하는 코드이다

이제 한 단계 더 나아가서,
**두 장의 이미지 + 카메라 내부 파라미터 K**가 주어졌을 때

1. F를 구하고
2. E를 만들고
3. R, T를 복원해서
4. 일부 keypoint들에 대해 3D 위치를 triangulation으로 구한 뒤
5. Open3D로 point cloud를 띄워보는 코드를 적어본다.

---

## 8.1 필요한 라이브러리 설치이다

```bash
pip install opencv-python open3d numpy
```

---

## 8.2 코드 예제다

```python
import cv2
import numpy as np
import open3d as o3d

# 1. 이미지와 카메라 내부 파라미터를 로드한다
img1 = cv2.imread("left.jpg")
img2 = cv2.imread("right.jpg")

if img1 is None or img2 is None:
    raise RuntimeError("이미지를 찾을 수 없다.")

h, w = img1.shape[:2]

# 예시용 K (실제 값으로 바꾸는 것이 좋다)
fx = fy = 1200.0
cx = w / 2.0
cy = h / 2.0
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# 2. 특징점을 검출하고 매칭한다 (앞 예시와 비슷)
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

# 3. F와 E를 계산한다
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

E = K.T @ F @ K   # 픽셀 좌표 F -> Essential matrix

print("F =\n", F)
print("E =\n", E)

# 4. Essential matrix에서 R, t를 복원한다
#    (cv2.recoverPose가 epipolar constraint p^T E p' = 0을 이용한다)
pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)

_, R, t, mask_pose = cv2.recoverPose(
    E, pts1_norm, pts2_norm, K
)

print("R =\n", R)
print("t =\n", t)

# 5. Triangulation으로 3D 점들을 계산한다
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, t))

pts1_h = pts1.reshape(-1, 1, 2)
pts2_h = pts2.reshape(-1, 1, 2)

pts4D_h = cv2.triangulatePoints(
    P1, P2,
    pts1_h.transpose(1, 0, 2).reshape(2, -1),
    pts2_h.transpose(1, 0, 2).reshape(2, -1)
)

# 동차좌표 -> 3D
pts3D = (pts4D_h[:3, :] / pts4D_h[3, :]).T  # (N, 3)

# 6. Open3D로 포인트클라우드를 시각화한다
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts3D)

# 색상도 대충 넣고 싶으면
colors = np.zeros_like(pts3D)
colors[:, 0] = 1.0  # 빨간색
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])
```

이 코드는 **Lecture 3 이론의 전체 흐름**을 그대로 따른다:

1. 두 이미지에서 correspondence를 모은다.
2. Fundamental matrix (F) 를 추정한다.
3. 카메라 내부 파라미터 (K) 로 Essential matrix (E = K^T F K)를 만든다.
4. (E)로부터 상대적인 (R, T)를 복원한다.
5. 두 카메라 projection matrix (P_1, P_2) 를 만들고
   `cv2.triangulatePoints` 로 3D 점을 복원한다.
6. Open3D로 3D point cloud를 띄워서 눈으로 확인한다.

이 과정을 직접 돌려보면,
강의에서 말한 “한 쌍의 이미지와 Epipolar Geometry만으로도
이미 상당한 3D 구조를 복원할 수 있다”는 사실을 체감하게 된다.

---

# 9. 정리다

지금까지 Lecture 3의 핵심을 스토리로 풀어봤다:

1. 한 장의 사진으로는 깊이에 대한 모호성이 크다.
2. 두 장의 사진과 카메라들 사이의 기하관계가 있는 순간,
   **Epipolar Geometry**로 강력한 제약을 얻는다.
3. Essential matrix ($E$)는 정규화 좌표에서
   $$p^T E p' = 0, \quad E = [T_\times] R$$
   를 만족한다.
4. Fundamental matrix ($F$)는 픽셀 좌표에서
   $$p^T F p' = 0, \quad F = K'^{-T}[T_\times]R K^{-1}$$
   를 만족한다.
5. Eight-Point Algorithm(정규화 포함)으로
   캘리브레이션 없이도 F를 추정할 수 있다.
6. F에서 epipole과 epipolar line들을 얻고,
   homography (H_1, H_2)로 rectification을 수행할 수 있다.
7. K를 알고 있다면, F → E → (R,T) → triangulation으로
   실제 3D 포인트들을 복원할 수 있다.

## 1. 서론 (Introduction)

에피폴라 기하학(Epipolar Geometry)은 **두 개의 카메라 뷰 간의 기하학적 관계**를 설명하는 수학적 프레임워크입니다. 스테레오 비전, 구조화된 빛, 그리고 다중 뷰 3D 재구성의 기초가 됩니다.

## 2. 에피폴라 기하학의 기본 개념

### 2.1. 에피폴라 평면 (Epipolar Plane)

두 카메라 중심 $O_1, O_2$와 3D 점 $P$를 포함하는 평면을 **에피폴라 평면(Epipolar Plane)**이라고 합니다.

![Epipolar Geometry Overview](/assets/images/posts/cs231a-03/figures/page_1_img_1.jpeg)
![Epipolar Geometry Diagram](/assets/images/posts/cs231a-03/figures/page_2_img_1.png)

### 2.2. 에피폴라 선 (Epipolar Line)

에피폴라 평면이 이미지 평면과 만나는 선을 **에피폴라 선(Epipolar Line)**이라고 합니다. 한 이미지의 점 $p_1$에 대응하는 다른 이미지의 점 $p_2$는 항상 에피폴라 선 위에 있습니다.

### 2.3. 에피폴라 점 (Epipole)

두 카메라 중심을 연결하는 선이 이미지 평면과 만나는 점을 **에피폴라 점(Epipole)**이라고 합니다.

## 3. Essential Matrix (본질 행렬)

### 3.1. Essential Matrix의 정의

Essential Matrix $E$는 두 카메라 뷰 간의 기하학적 관계를 나타내는 3×3 행렬입니다:

$$E = [t]_{\times} R$$

여기서 $R$은 회전 행렬, $t$는 이동 벡터, $[t]_{\times}$는 $t$의 외적을 나타내는 반대칭 행렬입니다.

### 3.2. Essential Matrix의 성질

Essential Matrix는 다음 관계를 만족합니다:

$$p_2^T E p_1 = 0$$

여기서 $p_1, p_2$는 정규화된 이미지 좌표입니다.

### 3.3. Essential Matrix의 추정

Essential Matrix는 최소 5개의 대응점(correspondence)으로 추정할 수 있습니다 (5-point algorithm). 8개의 대응점을 사용하면 더 안정적인 추정이 가능합니다 (8-point algorithm).

![Essential Matrix](/assets/images/posts/cs231a-03/figures/page_3_img_1.png)
![Essential Matrix Derivation](/assets/images/posts/cs231a-03/figures/page_3_img_2.png)

## 4. Fundamental Matrix (기본 행렬)

### 4.1. Fundamental Matrix의 정의

Fundamental Matrix $F$는 카메라 내부 파라미터를 고려한 Essential Matrix의 일반화입니다:

$$F = K_2^{-T} E K_1^{-1}$$

여기서 $K_1, K_2$는 각각 두 카메라의 내부 파라미터 행렬입니다.

### 4.2. Fundamental Matrix의 성질

Fundamental Matrix는 다음 관계를 만족합니다:

$$p_2^T F p_1 = 0$$

여기서 $p_1, p_2$는 픽셀 좌표입니다.

### 4.3. Fundamental Matrix의 추정

Fundamental Matrix는 최소 7개의 대응점으로 추정할 수 있습니다 (7-point algorithm). 8개의 대응점을 사용하면 선형 최소제곱법으로 추정할 수 있습니다 (8-point algorithm).

![Fundamental Matrix](/assets/images/posts/cs231a-03/figures/page_4_img_1.png)

## 5. 삼각 측량 (Triangulation)

두 이미지에서 대응점을 찾으면, 에피폴라 기하학을 이용하여 3D 점의 위치를 복원할 수 있습니다. 이 과정을 **삼각 측량(Triangulation)**이라고 합니다.

### 5.1. 삼각 측량 방법

1. 두 이미지에서 대응점 찾기
2. Essential Matrix 또는 Fundamental Matrix 추정
3. 카메라 파라미터 복원
4. 3D 점 위치 계산

## 6. 응용 분야 (Applications)

에피폴라 기하학은 다음과 같은 분야에서 활용됩니다:

- **스테레오 비전 (Stereo Vision)**: 두 카메라를 이용한 깊이 추정
- **구조화된 빛 (Structured Light)**: 프로젝터와 카메라를 이용한 3D 스캔
- **다중 뷰 3D 재구성 (Multi-view 3D Reconstruction)**: 여러 이미지로부터 3D 모델 생성
- **증강 현실 (Augmented Reality)**: 카메라 추적 및 3D 객체 배치

## 요약

이 강의에서는 다음과 같은 내용을 다뤘습니다:

1. **에피폴라 기하학**: 두 카메라 뷰 간의 기하학적 관계
2. **Essential Matrix**: 정규화된 좌표에서의 기하학적 관계
3. **Fundamental Matrix**: 픽셀 좌표에서의 기하학적 관계
4. **삼각 측량**: 대응점으로부터 3D 점 복원

에피폴라 기하학은 스테레오 비전과 다중 뷰 3D 재구성의 수학적 기초를 제공합니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [03-epipolar-geometry.pdf](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)
