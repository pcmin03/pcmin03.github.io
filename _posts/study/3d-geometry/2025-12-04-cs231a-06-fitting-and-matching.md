---
title: "[CS231A] Lecture 06: Fitting and Matching (피팅 및 매칭)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Fitting, and, Matching]
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

이 포스트는 Stanford CS231A 강의의 06번째 강의 노트인 "Fitting and Matching"를 한글로 정리한 것입니다.

**원본 강의 노트**: [06-fitting-matching.pdf](https://web.stanford.edu/class/cs231a/course_notes/06-fitting-matching.pdf)

<!--more-->

## 강의 개요

이 강의에서는 피팅 및 매칭에 대해 다룹니다.


## 강의 노트 페이지 이미지

<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
  <div><img src='/assets/images/posts/cs231a-06/page_1.png' alt='Page 1' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_2.png' alt='Page 2' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_3.png' alt='Page 3' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_4.png' alt='Page 4' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_5.png' alt='Page 5' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_6.png' alt='Page 6' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_7.png' alt='Page 7' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_8.png' alt='Page 8' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_9.png' alt='Page 9' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_10.png' alt='Page 10' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_11.png' alt='Page 11' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_12.png' alt='Page 12' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-06/page_13.png' alt='Page 13' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
</div>


## 주요 수식

이 강의에서 다루는 주요 수식들:

**수식 1:**

$$fitting attempts to find a line y = mx+b such that the squared error in the y dimension is minimized, as illustrated in Figure (1).$$

**수식 2:**

$$E = (y −yˆ)2 (1)$$

**수식 3:**

$$E = (y − x 1 )2 (3)$$

**수식 4:**

$$satisfying (x,y)·(a,b) = xa+by = 0 is the line orthogonal to ⃗n. However, the line can also be arbitrarily shifted to location (x ,y ), so we have$$

**수식 5:**

$$Given a 2D data point P = (x ,y ) and a point on the line Q = (x,y),$$

**수식 6:**

$$of QP onto the normal vector ⃗n orthogonal to the line. We have QP = (x −x,y −y),⃗n = (a,b), which gives:$$

**수식 7:**

$$z = a(x −x¯)+b(y −y¯) (25)$$

**수식 8:**

$$E = (a(x −x¯)+b(y −y¯))2 (37)$$

**수식 9:**

$$X = USVT (40)$$

**수식 10:**

$$Interpreting this from a gain perspective, we can write the SVD X =$$

**수식 11:**

$$X = σ u vT (43)$$

**수식 12:**

$$E = C(u ) (44)$$

**수식 13:**

$$We again have a series of N 2D points X = {(x ,y )}N that we want to i i i=1 fit a line to, illustrated by the dots in Figure (6). The first step in RANSAC$$

**수식 14:**

$$w, we define the inlier set as P = {(x ,y ) | r(p = (x ,y ),w) < δ}, where$$

**수식 15:**

$$set, defined as O = X \ P. The outlier set here is comprised of the red$$

**수식 16:**

$$We again want to fit a line of the form y = m′x+n′ to a series of points {(x ,y )}N in an image, illustrated on the left half of Figure (7). To find i i i=1$$

**수식 17:**

$$y = mx +n)becomesalineintheparameterspacedefinedbyn = −x m+y .$$

**수식 18:**

$$given by y = mx+n. We see that a line in the parameter space n = −x m+y represents all$$


## 1. Overview

The goal of 피팅 is to find a parametric model that best describes the observed data. We obtain the 최적 parameters of such a model by mini- mizing a chosen 피팅 error betwee the data and a particular estimate of the model parameters. A classic example is 피팅 a line to a set of given (x,y) points. Other examples we’ve seen in this class include computing a 2D homography H between set of point correspondences in different images or computing the fundamental matrix F using the eight-point algorithm.


## 2. Least-squares

Given a series of N 2D points {(x ,y )}N , the method of least-squares i i i=1 피팅 attempts to find a line y = mx+b such that the squared error in the y dimension is minimized, as illustrated in Figure (1). Figure 1: Ordinary least squares. 1 T Specifically, we want to find model parameters w = m b to minimize the sum of squared residuals between y and the model estimate yˆ = mx +b, i i i given in Equation (1). We define the residual as y −yˆ. i i N (cid:88) E = (y −yˆ)2 (1) i i i=1 N (cid:88) = (y −mx −b)2 (2) i i i=1 We can write this in matrix notation as: N (cid:20) (cid:21) (cid:88) m E = (y − x 1 )2 (3) i i b i=1     y x 1


## 1. 1 (cid:20) (cid:21)

. . . m = ∥ . . − . . . .  ∥2 (4)     b y x 1 N N = ∥Y −Xw∥2 (5) The residual is now r = y−Xw, we assume X to be skinny and full rank. We want to find the B that minimizes the norm of the residual squared, which 우리는 할 수 있습니다 write as: ∥r∥2 = rTr (6) = (y −Xw)T(y −Xw) (7) = yTy −2yTXw+wTXTXw (8) We then set the gradient of the residual with respect to w equal to 0. Recall XTX is symmetric. ∇ ∥r∥2 = −2XTy +2XTXw (9) w = 0 (10) This leads to the normal equations. XTXw = XTy (11) 2 We now have a closed-form solution for w in Equation (12). A is full rank so ATA is invertible. w = (XTX)−1XTy (12) However, note that this method fails completely for 피팅 points that describe a vertical line (m undefined). In this case, m would be set to ex- tremely large number, leading to numerically unstable solutions. To fix this, 우리는 할 수 있습니다 use an alternate line formulation of the form ax+by+d = 0. We can obtain a vertical line by setting b = 0. Here’s one way to think about this line 표현. The line direction (slope) is given by ⃗n; the set of (x,y) satisfying (x,y)·(a,b) = xa+by = 0 is the line orthogonal to ⃗n. However, the line can also be arbitrarily shifted to location (x ,y ), so we have


## 0. 0

a(x−x )+b(y −y ) = ax+by −ax −by (13)


## 0. 0 0 0

= ax+by +c (14) = 0 (15) where c = −ax −by . The slope of line is then m = −a, which now may be


## 0. 0 b

undefined. Earlierourresidualwasonlyinthey-axis. However, nowthatour new line parameterization accounts for error in the both the x- and y-axes, our new error is the sum of squared orthogonal distances, as illustrated in Figure (2). Figure 2: Total least squares. Given a 2D data point P = (x ,y ) and a point on the line Q = (x,y), i i the distance from P to the line is equivalent to the length of the 투영 3 −→ −→ of QP onto the normal vector ⃗n orthogonal to the line. We have QP = (x −x,y −y),⃗n = (a,b), which gives: i i −→ |QP ·⃗n| d = (16) ∥⃗n∥ |a(x −x)+b(y −y)| = i √ i (17) a2 +b2 |ax +by +c| = √i i (18) a2 +b2 Recall Q lies on the line, so c = −ax−by. Figure 3: Distance between a point and a line. T Our new set of parameters is now w = a b c . To simplify the error, we make the solution unique and remove the denominator by constraining ∥⃗n∥2 = 1, so the new error is N (cid:88) E(a,b,x ,y ) = (a(x −x )+b(y −y ))2 (19)


## 0. 0 i 0 i 0

i=1 N (cid:88) = (ax +by +c)2 (20) i i i=1 where a2 + b2 = 1. However, putting this into matrix notation is still tricky due to the presence of c when the constraint is only on a,b. To simplify 4 further, we note that the resulting line of best fit that minimizes E must pass through the data centroid (x¯,y¯), defined as N


## 1. (cid:88)

x¯ = x (21) i N i=1 N


## 1. (cid:88)

y¯= y (22) i N i=1 For every ⃗n and every possible set of points {(x ,y )}N , E is minimized i i i=1 when we set c = −ax¯−bx¯. In other words, given every point (x ,y ) ∈ R2,


## 0. 0

we have E(a,b,x ,y ) ≥ E(a,b,x¯,y¯) (23)


## 0. 0

To see why 이것은 true, we define vectors w,z such that w = a(x −x )+b(y −y ) (24) i i 0 i 0 z = a(x −x¯)+b(y −y¯) (25) i i i We can then write the errors as E(a,b,x ,y ) = ∥w∥2 (26)


## 0. 0

E(a,b,x¯,y¯) = ∥z∥2 (27) The relationship between w and z is then w = z +h1 (28) where h = a(x¯ −x ) +b(y¯−y ) ∈ R and 1 is a vector of all ones. z is


## 0. 0

orthogonal to 1 since N (cid:88) z ·1 = z (29) i i=1 N N (cid:88) (cid:88) = a (x −x¯)+b (y −y¯) (30) i i i=1 i=1 N N N N (cid:88) 1 (cid:88) (cid:88) 1 (cid:88) = a( x −N( x ))+b( y −N( y )) (31) i i i i N N i=1 i=1 i=1 i=1 = 0a+0b = 0 (32) 5 Thus, by the Pythagorean theorem, we have E(a,b,x ,y ) = ∥w∥2 (33)


## 0. 0

= ||z||2 +h2N (34) ≥ ||z||2 = E(a,b,x¯,y¯) (35) We have shown that the line of best fit must pass through (x¯,y¯), so 우리는 할 수 있습니다 constrain c as c = −ax¯−by¯ (36) We can then eliminate c by shifting all points to be centered around the data centroid (setting (x ,y ) = (x¯,y¯)), which allows us to finally formulate


## 0. 0

the error as a matrix product. N (cid:88) E = (a(x −x¯)+b(y −y¯))2 (37) i i i=1   x −x¯ y −y¯


## 1. 1 (cid:20) (cid:21)

. . a = ∥ . . . .  ∥2 (38)   b x −x¯ y −y¯ N N = ∥Xw∥2 (39) where w = a b T and ∥w∥2 = 1. This is a constrained least-squares problem that we’ve seen before in previous lectures. By SVD (X full rank), we have X = USVT (40) U ∈ RN×M,VT ∈ RM×M are both orthonormal matrices, while S ∈ RM×M is a diagonal matrix containing the singular values of X in descending order. M = 2 here. Since U,V are orthonormal, we know that ∥USVTw∥ = ∥SVTw∥ (41) ∥VTw∥ = ∥w∥ (42) Setting v = VTw, 우리는 할 수 있습니다 now minimize ∥SVTw∥ with the new but equivalent constraint that ∥v∥2 = 1. ∥SVTw∥ = ∥Sv∥ is minimized when 6 T v = 0 1 since the diagonal of S is sorted in descending order. Finally, we obtain w = VVTw = Vv, so the w that minimizes the error is the last column in V. Interpreting this from a gain perspective, 우리는 할 수 있습니다 write the SVD X = USVT as M (cid:88) X = σ u vT (43) i i i i=1 v ,...,v arecolumnsofV,σ arethediagonalvaluesofS = diag(σ ,...,σ ),


## 1. M i 1 M

and u ,...,u are the columns of U. Multiplying w by the SVD, USVTw,


## 1. M

can then be viewed as first computing the components of w along the input directions v ,...,v , scaling the components by σ , and then reconstituting


## 1. M i

along the output directions u ,...,u . VTw gives the 투영 of w along


## 1. M

each column in V (recall ∥v ∥2 = 1). Similarly, Uw′ 될 수 있습니다 seen as a linear i combination of the output directions, u w′ +···+u w′ . Thus, to find w


## 1. 1 M M

that minimizes Xw subject to ∥w∥2 = 1 is simply choosing the input direc- tion that minimizes the magnitude of the output vector, which is the last column of V. In practice, least-squares 피팅 handles noisy data well but is susceptible to outliers. To see why, if we write the residual for the i-th data point as u = ax +by +c, and the cost as C(u ), our error 될 수 있습니다 generalized to i i i i N (cid:88) E = C(u ) (44) i i=1 The quadratic growth of the squared error C(u ) = u2 that we’ve been i i using so far (illustrated on the left side of Figure (4) means that outliers with large residuals u exert an outsized influence on cost minimum. i 7 Figure 4: Cost functions: squared residual (left) vs. robust cost function (right). The x-axis error is the residual u , and the y-axis is the cost C(u ). i i We can penalize large residuals (outliers) less by a robust cost function (right half of Figure (4)), such as u2 C(u ,σ) = i (45) i σ2 +u2 i When the residual u is large, the cost C saturates to 1 such that their i contribution to the cost is limited, but when u is small, the cost function resembles the squared error. However, now we need to choose σ, also known as the scale parameter. σ controls how much weight is given to potential outliers, illustrated in Figure (5). A large σ widens the quadratic curve in the center, penalizing outliers more relative to other points (similar to the original squared error function). A small σ narrows the quadratic curve, penalizing outliers less. If σ is too small, then most of the residuals will be treated as outliers even when they are not, leading to a poor fit. If σ is too large, then we do not benefit from the robust cost function and end up with the least-squares fit. Figure 5: Comparison of different scale parameters. 8 Since the robust cost functions are non-linear, they are optimized with iterative methods....


## 3. RANSAC

Another 피팅 method called RANSAC, which stands for random sample consensus, is designed to be robust to outliers and missing data. We demon- strate using RANSAC to perform line 피팅, but it generalizes to many different 피팅 contexts. Figure 6: RANSAC procedure for line 피팅. We again have a series of N 2D points X = {(x ,y )}N that we want to i i i=1 fit a line to, illustrated by the dots in Figure (6). The first step in RANSAC is to randomly select the minimum number of points needed to fit a model. A line requires at least two points, so we choose the two points in green. If we were estimating the fundamental matrix F, we would need to choose 8 correspondences to use the eight-point algorithm. If we wanted to compute a 9 homography H ∈ R3×3, we would need 4 correspondences (we have two x,y coordinates for each correspondence) to cover the 8 degrees of freedom up to scale. ThesecondstepinRANSACistofitamodeltotherandomsampleset. Here, the two points in green are fitted (i.e., a line is drawn between them) to obtain the line in black. The third step step is to use the fitted model to compute the inlier set from the entire dataset. Given the model parameters w, we define the inlier set as P = {(x ,y ) | r(p = (x ,y ),w) < δ}, where i i i i the r is the residual between a data point and the model and δ is some arbitrary threshold. Here, the inlier set is represented by the green and blue points. The size of the inlier set, |P|, indicates how much of the entire set of points agrees with the fitted model. With P, 우리는 할 수 있습니다 also obtain the outlier set, defined as O = X \ P....


## 4. Hough transform

We introduce another 피팅 method known as the Hough transform, which is another voting procedure. 10 Figure 7: Hough transform. We again want to fit a line of the form y = m′x+n′ to a series of points {(x ,y )}N in an image, illustrated on the left half of Figure (7). To find i i i=1 this line, we consider the dual parameter, or Hough space, illustrated on the right half of Figure (7). A point (x ,y ) in the image space (on the line i i y = mx +n)becomesalineintheparameterspacedefinedbyn = −x m+y . i i i i Similarly, a point in the parameter space (m,n) is a line in the image space given by y = mx+n. We see that a line in the parameter space n = −x m+y represents all i i of the different possible lines in the image space that pass through the point (x ,y ) in the image space. Thus, to find the line in the image space that fits i i both image points (x ,y ) and (x ,y ), we associate both points with lines in


## 1. 1 1 1

the Hough space and find the point of intersection (m′,n′). This point in the Hough space represents the line in the image space that passes through both points in the image space. In practice, we would divide the Hough space into a discrete grid of square cells with width w for the parameter space. We would maintain a grid of counts for every w ×w cell centered at (m,n) denoted A(m,n) = 0 for all (m,n) initially. For every data point (x ,y ) i i in the image space, we would find all (m,n) satisfying n = −x m + y and i i increment the count by 1. After we do this for all data points, the point (m,n) in the Hough space with the highest count represent the fitted lines in the image space. We now see why 이것은 a voting procedure: each data element (x ,y ) can contribute up to one vote for each candidate line in the i i image space (m,n). However, there is a major limitation with the existing parameterization. As we discussed before with least-squares, the slope of a line in the image space is unbounded −∞ < m < ∞. This makes Hough voting an computa- 11 tionally and memory intensive algorithm in practice since there is no limit on the size of the parameter space that we are maintaining counts for. To solve this, we turn to the polar parameterization of a line, illustrated in Figure (8). xcos(θ)+ysin(θ) = ρ (49) Figure 8: Polar 표현 of the parameter space. The left half of Figure (8) shows that ρ is minimum distance from the origin to the line (bounded by the image size or maximum distance between any two points in the dataset), and θ is the angle between the x-axis and and normal vector of the line (bounded between 0 and π). We use the same Hough voting procedure as before, but now all possible lines in the Cartesian space going through a specific (x ,y ) corresond to sinusodial profile in the i i Hough space, as illustrated in the right half of of Figure (8). In practice, noisy data points means that sinusoidal profiles in the Hough space that correspond to points on the same line in the image space, may not necessarily intersect at the same point in the Hough space....


## 요약

이 강의에서는 피팅 및 매칭의 주요 개념과 방법론을 다뤘습니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [06-fitting-matching.pdf](https://web.stanford.edu/class/cs231a/course_notes/06-fitting-matching.pdf)
