---
title: "[CS231A] Lecture 04: Stereo Systems (스테레오 시스템)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Stereo, Systems]
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

이 포스트는 Stanford CS231A 강의의 04번째 강의 노트인 "Stereo Systems"를 한글로 정리한 것입니다.

**원본 강의 노트**: [Course_Notes_4.pdf](https://web.stanford.edu/class/cs231a/course_notes/Course_Notes_4.pdf)

<!--more-->

## 강의 개요

이 강의에서는 스테레오 시스템에 대해 다룹니다.


## 강의 노트 페이지 이미지

<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
  <div><img src='/assets/images/posts/cs231a-04/page_1.png' alt='Page 1' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_2.png' alt='Page 2' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_3.png' alt='Page 3' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_4.png' alt='Page 4' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_5.png' alt='Page 5' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_6.png' alt='Page 6' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_7.png' alt='Page 7' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_8.png' alt='Page 8' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_9.png' alt='Page 9' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_10.png' alt='Page 10' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_11.png' alt='Page 11' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_12.png' alt='Page 12' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_13.png' alt='Page 13' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_14.png' alt='Page 14' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_15.png' alt='Page 15' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_16.png' alt='Page 16' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_17.png' alt='Page 17' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-04/page_18.png' alt='Page 18' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
</div>


## 주요 수식

이 강의에서 다루는 주요 수식들:

**수식 1:**

$$images that correspond to each other p = MP = (x,y,1) and p′ = M′P = (x′,y′,1). By the definition of the cross product, p × (MP) = 0. We can$$

**수식 2:**

$$formulate a linear equation of the form AP = 0 where$$

**수식 3:**

```
P to the previous estimation of AP = 0 will correspond to a solution HP for the transformed problem (AH−1)(HP) = 0. Recall that SVD solves for the constraint that ∥P∥ = 1, which is not invariant under a projective transfor-
```

**수식 4:**

$$error. At each step we want to update our estimate P by some δ : P =$$

**수식 5:**

$$x = MX =  0 0 m 1 2 0 1     X X 2 3    = m 1 1 2 X (3.2)$$

**수식 6:**

$$m 1 X = A b X = AX +b (3.3)$$

**수식 7:**

$$x = A X +b (3.5)$$

**수식 8:**

$$We then find a perspective transformation H such that M H = [I 0] and$$

**수식 9:**

$$M H = [A B]$$

**수식 10:**

$$M H−1 = [I 0] M H−1 = [A b] (4.2)$$

**수식 11:**

$$p = M P = M H−1HP = [I | 0]P(cid:101)$$

**수식 12:**

$$p′ = M P = M H−1HP = [A | b]P(cid:101)$$

**수식 13:**

$$p′ = [A|b]P(cid:101) = A[I|0]P(cid:101)+b (4.4) = Ap+b$$

**수식 14:**

$$= p′T[b] Ap$$

**수식 15:**

$$the Fundamental matrix p′TFp = 0. If we set F = [b] A, then extracting A$$

**수식 16:**

$$F⊤b = [[b] A]⊤b = 0 (4.7)$$

**수식 17:**

$$Once b is known, we can now compute A. If we set A = −[b] F, then we$$

**수식 18:**

$$can verify that this definition satisfies F = [b] A:$$

**수식 19:**

$$[b ]A′ = −[b ][b ]F$$

**수식 20:**

$$M = [I 0] M = [−[b ]F b] (4.9)$$

**수식 21:**

$$M = [I 0] M = [−[e ]F e] (4.10)$$

**수식 22:**

$$E = KTFK (4.11)$$

**수식 23:**

$$E = [t] R (4.12)$$

**수식 24:**

$$W = 1 0 0, Z = −1 0 0 (4.13)$$

**수식 25:**

$$One important property we will use later is that Z = diag(1,1,0)W up to a sign. Similarly, we will also use the fact that ZW = ZWT = diag(1,1,0) up$$

**수식 26:**

$$[t] = UZUT (4.14)$$

**수식 27:**

$$E = Udiag(1,1,0)(WUTR) (4.15)$$

**수식 28:**

$$singular value decomposition E = UΣVT, where Σ contains two equal sin-$$

**수식 29:**

$$E = Udiag(1,1,0)VT, then we arrive at the following factorizations of E: [t] = UZUT, R = UWVT or UWTVT (4.16)$$

**수식 30:**

$$values in, we get ZX = diag(1,1,0) up to scale. Thus, X must be equal to$$

**수식 31:**

$$R = (detUWVT)UWVT or (detUWTVT)UWTVT (4.17)$$

**수식 32:**

$$t×t = [t] t = UZUTt = 0 (4.18)$$

**수식 33:**

$$Knowing that U is unitary, we can find that the ∥[t] ∥F = 2. Therefore,$$

**수식 34:**

$$t = ±U 0 = ±u$$

**수식 35:**

$$get the same results by reformatting [t] = UZUT into the vector t known$$


## 1. Introduction

In the previous notes, we covered how adding additional viewpoints of a scene can greatly enhance our knowledge of the said scene. We focused on the 에피폴라 geometry setup in order to relate points of one 이미지 평면 to points in the other without extracting any information about the 3D scene. In these lecture notes, we will discuss how to recover information about the 3D scene from multiple 2D images.


## 2. Triangulation

One of the most fundamental problems in multiple view geometry is the problem of triangulation, the process of determining the location of a 3D 점 given its 투영s into two or more images. Figure 1: The setup of the triangulation problem when given two views. 1 In the triangulation problem with two views, we have two cameras with known camera intrinsic parameters K and K′ respectively. We also know the relative orientations and offsets R,T of these cameras with respect to each other. Suppose that we have a point P in 3D, which 될 수 있습니다 found in the images of the two cameras at p and p′ respectively. Although the location of P is currently unknown, 우리는 할 수 있습니다 measure the exact locations of p and p′ in the image. Because K,K′,R,T are known, 우리는 할 수 있습니다 compute the two lines of sight ℓ and ℓ′, which are defined by the camera centers O ,O and the image


## 1. 2

locations p,p′. Therefore, P 될 수 있습니다 computed as the intersection of ℓ and ℓ′. Figure 2: The triangulation problem in real-world scenarios often involves minimizing the re투영 error. Although this process appears both straightforward and mathematically sound, it does not work very well in practice. In the real world, because the observations p and p′ are noisy and the camera calibration parameters are not precise, finding the intersection point of ℓ and ℓ′ may be problematic. In most cases, it will not exist at all, as the two lines may never intersect.


## 2.1. A linear method for triangulation

In this section, we describe a simple linear triangulation method that solves the lack of an intersection point between rays. We are given two points in the images that correspond to each other p = MP = (x,y,1) and p′ = M′P = (x′,y′,1). By the definition of the cross product, p × (MP) = 0. We can 2 explicitly use the equalities generated by the cross product to form three constraints: x(M P)−(M P) = 0


## 3. 1

y(M P)−(M P) = 0 (2.1)


## 3. 2

x(M P)−y(M P) = 0


## 2. 1

where M is the i-th row of the matrix M. Similar constraints 될 수 있습니다 for- i mulated for p′ and M′. Using the constraints from both images, 우리는 할 수 있습니다 formulate a linear equation of the form AP = 0 where   xM −M


## 3. 1

A =   yM 3 −M 2  (2.2) x′M′ −M′ 


## 3. 1

y′M′ −M′


## 3. 2

This equation 될 수 있습니다 solved using SVD to find the best linear estimate of the point P. Another interesting aspect of this method is that it can actu- ally handle triangulating from multiple views as well. To do so, one simply appends additional rows to A corresponding to the added constraints by the new views. This method, however is not suitable for projective reconstruction, as 그것은 not projective-invariant. For example, suppose we replace the camera matri- ces M,M′ with ones affected by a projective transformation MH−1,M′H−1. The matrix of linear equations A then becomes AH−1. Therefore, a solution P to the previous 추정 of AP = 0 will correspond to a solution HP for the transformed problem (AH−1)(HP) = 0. Recall that SVD solves for the constraint that ∥P∥ = 1, which is not invariant under a projective transfor- mation H. Therefore, this method, although simple, is often not the 최적 solution to the triangulation problem. -


## 2.2. A nonlinear method for triangulation

Instead, the triangulation problem for real-world scenarios is often mathe- matically characterized as solving a minimization problem: min∥MP ˆ −p∥2 +∥M′P ˆ −p′∥2 (2.3) Pˆ ˆ In the above equation, we seek to find a P in 3D that best approximates P ˆ by finding the best least-squares estimate of the re투영 error of P in bothimages. There투영errorfora3Dpointinanimageisthedistance between the 투영 of that point in the image and the corresponding 3 observed point in the 이미지 평면. In the case of our example in Figure 2, since M is the projective transformation from 3D space to image 1, the ˆ ˆ ˆ projected point of P in image 1 is MP. The 매칭 observation of P in image 1 is p. Thus, the re투영 error for point P in image 1 is the ˆ distance ∥MP − p∥. The overall re투영 error found in Equation 2.3 is the sum of the re투영 errors across all the points in the image. For cases with more than two images, we would simply add more distance terms to the objective function. (cid:88) min ∥MP ˆ −p ∥2 (2.4) i i Pˆ i In practice, there exists a variety of very sophisticated optimization tech- niques that result in good approximations to the problem. However, for the scope of the class, we will focus on only one of these techniques, which is the Gauss-Newton algorithm for nonlinear least squares. The general nonlinear least squares problem is to find an x ∈ Rn that minimizes m (cid:88) ∥r(x)∥2 = r (x)2 (2.5) i i=1 where r is any residual function r : Rn → Rm such that r(x) = f(x) − y for some function f, input x, and observation y. The nonlinear least squares problem reduces to the regular, linear least squares problem when the function f is linear. However, recall that, in general, our camera matrices are not affine. Because the 투영 into the 이미지 평면 often involves a division by the homogeneous coordinate, the 투영 into the image is generally nonlinear. ˆ Notice that if we set e to be a 2×1 vector e = MP −p , then 우리는 할 수 있습니다 i i i i reformulate our optimization problem to be: (cid:88) min e (P ˆ )2 (2.6) i Pˆ i which 될 수 있습니다 perfectly represented as a nonlinear least squares problem. In these notes, we will cover how 우리는 할 수 있습니다 use the popular Gauss-Newton algorithm to find an approximate solution to this nonlinear least squares problem....


## 1. 1 1

. . e =  . .  =  . .  (2.10)     e p −M P ˆ N n n and ∂e ∂e ∂e 


## 1. 1 1

ˆ ˆ ˆ ∂P 1 ∂P 2 ∂P 3  . . .  J =  . . . . . .  (2.11)   ∂e ∂e ∂e  N N N ˆ ˆ ˆ ∂P ∂P ∂P


## 1. 2 3

Recall that the residual error vector of a particular image e is a 2 × 1 i vector because there are two dimensions in the 이미지 평면. Consequently, in the simplest two camera case (N = 2) of triangulation, this results in the residual vector e being a 2N ×1 = 4×1 vector and the Jacobian J being a 2N×3 = 4×3 matrix. Notice how this method handles multiple views seam- lessly, as additional images are accounted for by adding the corresponding rows to the e vector and J matrix. After computing the update δ , 우리는 할 수 있습니다 P simply repeat the process for a fixed number of steps or until it numerically converges. One important property of the Gauss-Newton algorithm is that our assumption that the residual function is linear near our estimate gives us no guarantee of convergence. Thus, 그것은 always useful in practice to put an upper bound on the number of updates made to the estimate. 5


## 3. Affine structure from motion

Attheendoftheprevioussection, wehintedhowwecangobeyondtwoviews of a scene to gain information about the 3D scene. We will now explore the extension of the geometry of two cameras to multiple cameras. By combining observations of points from multiple views, we will be able to simultaneously determine both the 3D structure of the scene and the parameters of the camera in what is known as structure from motion. Figure 3: The setup of the general structure from motion problem. Here, we formally introduce the structure from motion problem. Suppose we have m cameras with camera transformations M encoding both the in- i trinsic and extrinsic parameters for the cameras. Let X be one of the n 3D j points in the scene. Each 3D 점 may be visible in multiple cameras at the location x , which is the 투영 of X to the image of the camera i using ij j the projective transformation M . The aim of structure from motion is to i recover both the structure of the scene (the n 3D 점s X ) and the motion j of the cameras (the m 투영 matrices M ) from all the observations x . i ij


## 3.1. The affine structure from motion problem

Before tackling the general structure from motion problem, we will first start with a simpler problem, which assumes the cameras are affine or weak per- spective. Ultimately, the lack of the perspective scaling operation makes the 6 mathematical derivation easier for this problem. Previously, we derived the above equations for perspective and weak per- spective cases. Remember that in the full perspective model, the 카메라 행렬 is defined as (cid:20) (cid:21) A b M = (3.1) v 1 where v is some non-zero 1 × 3 vector. On the other hand, for the weak perspectivemodel, v = 0. Wefindthatthispropertymakesthehomogeneous coordinate of MX equal to 1:    m  X 1  m X  x = MX =  0 0 m 1 2 0 1     X X 2 3    = m 1 1 2 X (3.2) 1 Consequently, the nonlinearity of the projective transformation disap- pears as we move from homogeneous to Euclidean coordinates, and the weak perspective transformation acts as a mere magnifier. We canmore compactly represent the 투영 as: (cid:20) (cid:21) m 1 X = A b X = AX +b (3.3) m X 2 and represent any 카메라 행렬 in the format M = A b . Thus, we affine now use the affine camera model to express the relationship from a point X j in 3D and the corresponding observations in each affine camera (for instance, x in camera i). ij Returning to the structure from motion problem, we need to estimate m matrices M , and the n world coordinate vectors X , for a total of 8m+3n i j unknowns, from mn observations. Each observation creates 2 constraints per camera, so there are 2mn equations in 8m + 3n unknowns. We can use this equation to know the lower bound on the number of corresponding observations in each of the images that we need to have. For example, if we have m = 2 cameras, then we need to have at least n = 16 points in 3D. However, once we do have enough corresponding points labeled in each image, how do we solve this problem?


## 3.2. The Tomasi and Kanade factorization method

In this part, we outline Tomasi and Kanade’s factorization method for solving the affine structure from motion problem. This method consists of two major steps: the data centering step and the actual factorization step. 7 Figure 4: When applying the centering step, we translate all of the image points such that their centroid (denoted as the lower left red cross) is located at the origin in the 이미지 평면. Similarly, we place the world coordinate system such that the origin is at the centroid of the 3D 점s (denoted as the upper right red cross). Let’s begin with the data centering step. In this step, 주요 idea is center the data at the origin. To do so, for each image i, we redefine new coordinates xˆ for each image point x by subtracting out their centroid x¯ : ij ij i n


## 1. (cid:88)

xˆ = x −x¯ = x − x (3.4) ij ij i ij ij n j=1 Recall that the affine structure from motion problem allows us to define the relationship between image points x , the 카메라 행렬 variables A and ij i b , and the 3D 점s X as: i j x = A X +b (3.5) ij i j i After this centering step, 우리는 할 수 있습니다 combine definition of the centered image 8 points xˆ in Equation 3.4 and the affine expression in Equation 3.5: ij n


## 1. (cid:88)

xˆ = x − x ij ij ik n k=1 n


## 1. (cid:88)

= A X − A X i j i k n k=1 (3.6) n


## 1. (cid:88)

= A (X − X ) i j k n k=1 ¯ = A (X −X) i j ˆ = A X i j As we see from Equation 3.6, if we translate the origin of the world ref- ¯ erence system to the centroid X, then the centered coordinates of the image ˆ points xˆ and centered coordinates of the 3D 점s X are related only by ij ij a single 2×3 matrix A . Ultimately, the centering step of the factorization i methodallowsustocreateacompactmatrixproduct표현torelate the 3D structure with their observed points in multiple images. ˆ However, notice that in the matrix product xˆ = A X , we only have ij i j access to the values on the left hand side of the equation. Thus, we must somehow factor out the motion matrices A and structure X . Using all i j the observations for all the cameras, 우리는 할 수 있습니다 build a measurement matrix D, made up of n observations in the m cameras (remember that each xˆ entry ij is a 2x1 vector):   xˆ xˆ ... xˆ


## 11. 12 1n

xˆ xˆ ... xˆ 


## 21. 22 2n

D =   ...   (3.7)   xˆ xˆ ... xˆ m1 m2 mn Now recall that because of our affine assumption, D 될 수 있습니다 expressed as the product of the 2m × 3 motion matrix M (which comprises the camera matrices A ,...A ) and the 3×n structure matrix S (which comprises the


## 1. m

3DpointsX ,...X ). Animportantfactthatwewilluseisthatrank(D) = 3


## 1. n

since D is the product of two matrices whose max dimension is 3. TofactorizeDintoM andS,wewillusethesingularvaluedecomposition, D = UΣVT. Since we know the rank(D) = 3, so there will only be 3 non- zero singular values σ ,σ , and σ in Σ. Thus, 우리는 할 수 있습니다 further reduce the


## 1. 2 3

9 expression and obtain the following decomposition: D = UΣVT   σ 0 0 0 ... 0 1 0 σ 0 0 ... 0   2  vT 0 0 σ 0 ... 0 1 3 . = u 1 ... u n  0 0 0 0 ... 0     . .      ...  v n T (3.8)  


## 0. 0 0 0 ... 0

   σ 0 0 vT


## 1. 1

 = u 1 u 2 u 3 0 σ 2 0v 2 T 


## 3. 3

= U Σ VT


## 3. 3 3

In this decomposition, Σ is defined as the diagonal matrix formed by the 3 non-zero singular values, while U and VT are obtained by taking the corre-


## 3. 3

sponding three columns of U and rows of VT respectively. Unfortunately, in practice, rank(D) > 3 because of measurement noise and the affine camera approximation. However, recall that when rank(D) > 3, U W VT is still


## 3. 3 3

the best possible rank-3 approximation of MS in the sense of the Frobenius norm. Upon close inspection, we see that the matrix product Σ VT forms a 3×n


## 3. 3

matrix, which exactly the same size as the structure matrix S. Similarly, U 3 isa2m×3matrix, whichisthesamesizeasthemotionmatrixM. Whilethis way of associating the components of the SVD decomposition to M and S leads to a physically and geometrical plausible solution of the affine structure from motion problem, this choice is not a unique solution. For example, we could also set the motion matrix to M = U Σ and the structure matrix to


## 3. 3

S = VT, since in either cases the observation matrix D is the same. So what 3 factorization do we choose? In their paper, Tomasi and Kanade concluded √ √ that a robust choice of the factorization is M = U Σ and S = Σ VT.


## 3.3. Ambiguity in reconstruction

Nevertheless, we find inherent ambiguity in any choice of the factorization D = MS, as any arbitrary, invertible 3 × 3 matrix A may be inserted into the decomposition: D = MAA−1S = (MA)(A−1S) (3.9) This means that the camera matrices obtained from motion M and the 3D 점s obtained from structure S are determined up to a multiplication by a 10 common matrix A. Therefore, our solution is underdetermined, and requires extra constraints to resolve this affine ambiguity. When a reconstruction has affine ambiguity, it means that parallelism is preserved, but the metric scale is unknown. Another important class of ambiguities for reconstruction is the similarity ambiguity, which occurs when a reconstruction is correct up to a similarity transform (회전, 평행 이동 and scaling). A reconstruction with only similarity ambiguity is known as a metric reconstruction. This ambiguity exists even when the camera are intrinsically calibrated. The good news is that for calibrated cameras, the similarity ambiguity is the only ambiguity1. The fact that there is no way to recover the absolute scale of a scene from images is fairly intuitive. An object’s scale, absolute position and canonical orientation will always be unknown unless we make further assumptions (e.g, we know the height of the house in the figure) or incorporate more data. This is because some attributes may compensate for others. For instance, to get the same image, 우리는 할 수 있습니다 simply move the object backwards and scale it accordingly. One such example of removing similarity ambiguity occurred during the camera calibration procedure, where we made the assumption that we know the location of the calibration points with respect to the world reference system. This enabled us to know the size of the squares of the checkerboard to learn a metric scale of the 3D structure.


## 4. Perspective structure from motion

After studying the simplified affine structure from motion problem, let us now consider the general case for projective cameras M . In the general i case with projective cameras, each 카메라 행렬 M contains 11 degrees of i freedom, as 그것은 defined up to scale:   a a a b


## 11. 12 13 1

M i = a 21 a 22 a 23 b 2 (4.1) a a a 1


## 31. 32 33

Moreover, similar to the affine case where the solution 될 수 있습니다 found up to an affine transformation, solutions for structure and motion 될 수 있습니다 de- termined up a projective transformation in the general case: 우리는 할 수 있습니다 always arbitrarilyapplya4×4projectivetransformationH tothemotionmatrix, as long as we also transform the structure matrix by the inverse transformation H−1. The resulting observations in the 이미지 평면 will still be the same. 1See [Longuet-Higgins ’81] for more details. 11 Similar to the affine case, 우리는 할 수 있습니다 set up the general structure from motion problem as estimating both the m motion matrices M and n 3D 점s X i j from mn observations x . Because cameras and points can only be recovered ij up to a 4 × 4 projective transformation up to scale (15 parameters), we have 11m+3n−15 unknowns in 2mn equations. From these facts, 우리는 할 수 있습니다 determine the number of views and observations that are required to solve for the unknowns.


## 4.1. The algebraic approach

Figure 5: In the algebraic approach, we consider sequential, camera pairs to determine camera matrices M and M up to a perspective transformation.


## 1. 2

We then find a perspective transformation H such that M H = [I 0] and 1 M H = [A B] 2 . We will now cover the algebraic approach, which leverages the concept of fundamental matrix F for solving the structure from motion problem for two cameras. As shown in Figure 5, 주요 idea of the algebraic approach is to compute two camera matrices M and M , which can only be computed


## 요약

이 강의에서는 스테레오 시스템의 주요 개념과 방법론을 다뤘습니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [Course_Notes_4.pdf](https://web.stanford.edu/class/cs231a/course_notes/Course_Notes_4.pdf)
