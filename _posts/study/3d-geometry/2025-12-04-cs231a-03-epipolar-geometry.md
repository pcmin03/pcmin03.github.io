---
title: "[CS231A] Lecture 03: Epipolar Geometry (에피폴라 기하학)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Epipolar, Geometry]
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

이 포스트는 Stanford CS231A 강의의 03번째 강의 노트인 "Epipolar Geometry"를 한글로 정리한 것입니다.

**원본 강의 노트**: [03-epipolar-geometry.pdf](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)

<!--more-->

## 강의 개요

이 강의에서는 에피폴라 기하학에 대해 다룹니다.


## 강의 노트 페이지 이미지

<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
  <div><img src='/assets/images/posts/cs231a-03/page_1.png' alt='Page 1' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_2.png' alt='Page 2' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_3.png' alt='Page 3' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_4.png' alt='Page 4' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_5.png' alt='Page 5' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_6.png' alt='Page 6' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_7.png' alt='Page 7' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_8.png' alt='Page 8' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_9.png' alt='Page 9' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_10.png' alt='Page 10' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_11.png' alt='Page 11' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_12.png' alt='Page 12' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_13.png' alt='Page 13' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-03/page_14.png' alt='Page 14' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
</div>


## 주요 수식

이 강의에서 다루는 주요 수식들:

**수식 1:**

```
which K = K' = I. This reduces Equation 1 to M = I 0 M' = RT −RTT (2) Furthermore, this means that the location of p' in the first camera’s ref- erence system is Rp'+T. Since the vectors Rp'+T and T lie in the epipolar plane, then if we take the cross product of T ×(Rp'+T) = T ×(Rp'), we will
```

**수식 2:**

$$pT ·[T ×(Rp')] = 0 (3)$$

**수식 3:**

$$a×b =  a z 0 −a xb y = [a × ]b (4)$$

**수식 4:**

$$pT ·[T ](Rp') = 0$$

**수식 5:**

$$pT[T ]Rp' = 0$$

**수식 6:**

$$The matrix E = [T ]R is known as the Essential Matrix, creating a com-$$

**수식 7:**

$$Thus e' satisfies e'T(ETx) = (e'TET)x = 0 for all the x, so Ee' = 0. Similarly ETe = 0.$$

**수식 8:**

$$pT[T ]Rp' = 0 (8)$$

**수식 9:**

$$pTK−T[T ]RK'−1p' = 0 (9)$$

**수식 10:**

$$The matrix F = K'−T[T ]RK−1 is known as the Fundamental Matrix,$$

**수식 11:**

$$Wf = 0 (12)$$

**수식 12:**

$$same K and that there is no relative rotation between the cameras (R = I).$$

**수식 13:**

$$E = [T × ]R = 0 0 −T x (17)$$

**수식 14:**

$$coordinates) are in the set {x|(cid:96)Tx = 0}. If we define each epipolar line as (cid:96) = (cid:96) (cid:96) (cid:96) , then we can we formulate a linear system of equations$$

**수식 15:**

$$R = −α√ e' 2 α√ e' 1 0 (21)  e'2+e'2 e'2+e'2 $$

**수식 16:**

$$H = T−1GRT (23)$$

**수식 17:**

$$H = H H M (25)$$

**수식 18:**

$$where F = [e] M and$$

**수식 19:**

$$F = [e] M = [e] [e] [e] M = [e] [e] F (27)$$

**수식 20:**

$$M = [e] F (28)$$

**수식 21:**

$$the F = [e] M still holds up to scale. Therefore, the more general case of$$

**수식 22:**

$$M = [e] F +evT (29)$$


## 1. Introduction

Previously, we have seen how to compute the intrinsic and extrinsic param- eters of a camera using one or more views using a typical camera calibration procedure or single view metrology. This process culminated in deriving properties about the 3D world from one image. However, in general, 그것은 not possible to recover the entire structure of the 3D world from just one image. This is due to the intrinsic ambiguity of the 3D to the 2D mapping: some information is simply lost. Figure 1: A single picture such as this picture of a man holding up the Leaning Tower of Pisa can result in ambiguous scenarios. Multiple views of the same scene help us resolve these potential ambiguities. For example, in Figure 1, we may be initially fooled to believe that the man is holding up the Leaning Tower of Pisa. Only by careful inspection can we tell that 이것은 not the case and merely an illusion based on the 투영 of different 깊이s onto the 이미지 평면. However, if we were able to view this scene from a completely different angle, this illusion immediately disappears and we would instantly figure out the correct scene layout. 1 Thefocusoftheselecturenotesistoshowhowhavingknowledgeofgeom- etrywhenmultiplecamerasarepresentcanbeextremelyhelpful. Specifically, we will first focus on defining the geometry involved in two viewpoints and then present how this geometry can aid in further understanding the world around us.


## 2. Epipolar Geometry

Figure 2: The general setup of 에피폴라 geometry. The gray region is the 에피폴라 plane. The orange line is the baseline, while the two blue lines are the 에피폴라 lines. Often in multiple view geometry, there are interesting relationships be- tween the multiple cameras, a 3D 점, and that point’s 투영s in each of the camera’s 이미지 평면. The geometry that relates the cameras, points in 3D, and the corresponding observations is referred to as the 에피폴라 geometry of a 스테레오 pair. As illustrated in Figure 2, the standard 에피폴라 geometry setup involves two cameras observing the same 3D 점 P, whose 투영 in each of the 이미지 평면s is located at p and p' respectively. 카메라는 centers are located at O and O , and the line between them is referred to as the


## 1. 2

baseline. We call the plane defined by the two camera centers and P the 에피폴라 plane. Thelocationsofwherethebaselineintersectsthetwoimage 2 Figure 3: An example of 에피폴라 lines and their corresponding points drawn on an image pair. planes are known as the the epipoles e and e'. Finally, the lines defined by the intersection of the 에피폴라 plane and the two 이미지 평면s are known as the 에피폴라 lines. The 에피폴라 lines have the property that they intersect the baseline at the respective epipoles in the 이미지 평면. Figure 4: When the two 이미지 평면s are parallel, then the epipoles e and e' are located at infinity. Notice that the 에피폴라 lines are parallel to the u axis of each 이미지 평면. An interesting case of 에피폴라 geometry is shown in Figure 4, which occurs when the 이미지 평면s are parallel to each other. When the 이미지 평면s are parallel to each other, then the epipoles e and e' will be located at infinity since the baseline joining the centers O ,O is parallel to the image


## 1. 2

planes. Another important byproduct of this case is that the 에피폴라 lines are parallel to an axis of each 이미지 평면. This case is especially useful 3 and will be covered in greater detail in the subsequent section on image rectification. In real world situations, however, we are not given the exact location of the3DlocationP, butcandetermineits투영inoneoftheimageplanes p. We also should be able to know the cameras locations, orientations, and camera matrices. What can we do with this knowledge? With the knowledge of camera locations O ,O and the image point p, 우리는 할 수 있습니다 define the 에피폴라


## 1. 2

plane. With this 에피폴라 plane, 우리는 할 수 있습니다 then determine the 에피폴라 lines1. By definition, P’s 투영 into the second image p' must be located on the 에피폴라 line of the second image. Thus, a basic understanding of 에피폴라 geometry allows us to create a strong constraint between image pairs without knowing the 3D structure of the scene. Figure 5: The setup for determining the essential and fundamental matrices, which help map points and 에피폴라 lines across views. Wewillnowtrytodevelopseamlesswaystomappointsand에피폴라lines across views. If we take the setup given in the original 에피폴라 geometry framework(Figure5),thenweshallfurtherdefineM andM' tobethecamera 투영 matrices that map 3D 점s into their respective 2D 이미지 평면 locations. Let us assume that the world reference system is associated to the first camera with the second camera offset first by a 회전 R and then by a 평행 이동 T. This specifies the camera 투영 matrices to be: M = K I 0 M' = K' RT −RTT (1) 1This means that 에피폴라 lines 될 수 있습니다 determined by just knowing the camera centers O ,O and a point in one of the images p


## 1. 2

4


## 3. The Essential Matrix

In the simplest case, let us assume that we have canonical cameras, in which K = K' = I. This reduces Equation 1 to M = I 0 M' = RT −RTT (2) Furthermore, this means that the location of p' in the first camera’s ref- erence system is Rp'+T. Since the vectors Rp'+T and T lie in the 에피폴라 plane, then if we take the cross product of T ×(Rp'+T) = T ×(Rp'), we will get a vector normal to the 에피폴라 plane. This also means that p, which lies in the 에피폴라 plane is normal to T × (Rp'), giving us the constraint that their dot product is zero: pT ·[T ×(Rp')] = 0 (3) From linear algebra, 우리는 할 수 있습니다 introduce a different and compact expression for the cross product: 우리는 할 수 있습니다 represent the cross product between any two vectors a and b as a matrix-vector multiplication:   


## 0. −a a b

z y x a×b =  a z 0 −a xb y = [a × ]b (4) −a a 0 b y x z Combining this expression with Equation 3, 우리는 할 수 있습니다 convert the cross product term into matrix multiplication, giving pT ·[T ](Rp') = 0 × (5) pT[T ]Rp' = 0 × The matrix E = [T ]R is known as the Essential Matrix, creating a com- × pact expression for the 에피폴라 constraint: pTEp' = 0 (6) The Essential matrix is a 3×3 matrix that contains 5 degrees of freedom. It has rank 2 and is singular. The Essential matrix is useful for computing the 에피폴라 lines associated with p and p'. For instance, (cid:96)' = ETp gives the 에피폴라 line in the 이미지 평면ofcamera2. Similarly(cid:96) = Ep' givesthe에피폴라lineintheimageplane of camera 1. Other interesting properties of the essential matrix is that its dot product with the epipoles equate to zero: ETe = Ee' = 0. Because for any point x (other than e) in the image of camera 1, the corresponding 에피폴라 line in the image of camera 2, l' = ETx, contains the epipole e'. Thus e' satisfies e'T(ETx) = (e'TET)x = 0 for all the x, so Ee' = 0. Similarly ETe = 0. 5


## 4. The Fundamental Matrix

Although we derived a relationship between p and p' when we have canoni- cal cameras, we should be able to find a more general expression when the camerasarenolongercanonical. Recallthatgivesusthe투영matrices: M = K I 0 M' = K' RT −RTT (7) First, we must define p = K−1p and p' = K'−1p' to be the 투영s of c c P to the corresponding camera images if the cameras were canonical. Recall that in the canonical case: pT[T ]Rp' = 0 (8) c × c By substituting in the values of p and p', we get c c pTK−T[T ]RK'−1p' = 0 (9) × The matrix F = K'−T[T ]RK−1 is known as the Fundamental Matrix, × which acts similar to the Essential matrix from the previous section but also encodes information about the camera matrices K,K' and the relative 평행 이동 T and 회전 R between the cameras. Therefore, 그것은 also usefulincomputingthe에피폴라linesassociatedwithpandp', evenwhenthe camera matrices K,K' and the transformation R,T are unknown. Similar to theEssentialmatrix, wecancomputethe에피폴라lines(cid:96)' = FTpand(cid:96) = Fp' from just the Fundamental matrix and the corresponding points. One main difference between the Fundamental matrix and the Essential matrix is that the Fundamental matrix contains 7 degrees of freedom, compared to the Essential matrix’s 5 degrees of freedom. But how is the Fundamental matrix useful? Like the Essential matrix, if we know the Fundamental matrix, then simply knowing a point in an image gives us an easy constraint (the 에피폴라 line) of the corresponding point in the other image. Therefore, without knowing the actual position of P in 3D space, or any of the extrinsic or intrinsic characteristics of the cameras, 우리는 할 수 있습니다 establish a relationship between any p and p'.


## 4.1. The Eight-Point Algorithm

Still, the assumption that 우리는 할 수 있습니다 have the Fundamental matrix, which is defined by a matrix product of the camera parameters, seems rather large. However, 그것은 possible to estimate the Fundamental matrix given two images ofthesamesceneandwithoutknowingtheextrinsicorintrinsicparametersof thecamera. ThemethodwediscussfordoingsoisknownastheEight-Point 6 Figure 6: Corresponding points are drawn in the same color on each of the respective images. Algorithm, which was proposed by Longuet-Higgins in 1981 and extended by Hartley in 1995. As the title suggests, the Eight-Point Algorithm assumes that a set of at least 8 pairs of corresponding points between two images is available. Each correspondence p = (u ,v ,1) and p' = (u',v',1) gives us the epipo- i i i i i i lar constraint pTFp' = 0. We can reformulate the constraint as follows: i i   F 11 F 12   F 13   F 21   u u' v'u u u'v v v' v u' v' 1 F  = 0 (10) i i i i i i i i i i i i  22  F  23   F  31   F 32 F 33 Since this constraint is a scalar equation, it only constrains one degree of freedom. Since 우리는 할 수 있습니다 only know the Fundamental matrix up to scale, we require eight of these constraints to determine the Fundamental matrix: 7    u u' v'u u' u'v v v' v u' v' 1  F 11       u u u 1 2 3 4 u u u 1 ' 2 ' 3 ' 4 v v v 1 2 3 4 ' ' ' u u u 1 2 3 4 u u u 1 ' 2 ' 3 ' 4 u u u 1 ' 2 ' 3 ' 4 v v v 1 2 3 4 v v v 1 2 3 4 v v v 1 2 3 4 ' ' ' v v v 1 2 3 4 u u u 1 ' 2 ' 3 ' 4 v v v 1 2 3 4 ' ' ' 1 1 1              F F F F 1 1 2 2 3 1        = 0 (11) u u' v'u u' u'v v v' v u' v' 1 22   5 5 5 5 5 5 5 5 5 5 5 5 F  u u' v'u u' u'v v v' v u' v' 1 23   6 6 6 6 6 6 6 6 6 6 6 6 F  u u 7 u u '


## 7. (cid:48)

v v 7 ' ' u u 7 u u '


## 7. (cid:48)

u u '


## 7. (cid:48)

v v 7 v v 7 v v 7 ' ' v v 7 u u '


## 7. (cid:48)

v v 7 ' ' 1 1  F 3 3 1 2  


## 8. 8 8 8 8 8 8 8 8 8 8 8 F

33 This 될 수 있습니다 compactly written as Wf = 0 (12) where W is an N ×9 matrix derived from N ≥ 8 correspondences and f is the values of the Fundamental matrix we desire. In practice, it often is better to use more than eight correspondences and createalargerW matrixbecauseitreducestheeffectsofnoisymeasurements. The solution to this system of homogeneous equations 될 수 있습니다 found in the least-squares sense by Singular Value Decomposition (SVD), as W is rank- ˆ deficient. SVD will give us a estimate of the Fundamental matrix F, which may have full rank. However, we know that the true Fundamental matrix has rank 2. Therefore, we should look for a solution that is the best rank-2 ˆ approximation of F. To do so, we solve the following optimization problem: ˆ minimize (cid:107)F −F(cid:107) F F (13) subject to detF = 0 This problem is solved again by SVD, where F ˆ = UΣVT, then the best rank-2 approximation is found by   Σ 0 0 1 F = U  0 Σ 2 0VT (14)


## 4.2. The Normalized Eight-Point Algorithm

In practice, the standard least-squares approach to the Eight-Point Algo- rithm is not precise. Often, the distance between a point p and its corre- i sponding 에피폴라 line (cid:96) = Fp' will be very large, usually on the scale of i 10+ pixels. To reduce this error, 우리는 할 수 있습니다 consider a modified version of the Eight-Point Algorithm called the Normalized Eight-Point Algorithm. 8 The main problem of the standard Eight-Point Algorithm stems from the fact that W is ill-conditioned for SVD. For SVD to work properly, W should have one singular value equal to (or near) zero, with the other singular values being nonzero. However, the correspondences p = (u ,v ,1) will often have i i i extremely large values in the first and second coordinates due to the pixel range of a modern camera (i.e. p = (1832,1023,1)). If the image points i used to construct W are in a relatively small region of the image, then each of the vectors for p and p' will generally be very similar. Consequently, the i i constructed W matrix will have one very large singular value, with the rest relatively small. To solve this problem, we will normalize the points in the image before constructingW. Thismeanswepre-conditionW byapplyingbothatrans- lation and scaling on the image coordinates such that two requirements are satisfied. First, the origin of the new coordinate system should be located at the centroid of the image points (평행 이동). Second, the mean square distance of the transformed image points from the origin should be 2 pix- els (scaling). We can compactly represent this process by a transformation matrices T,T' that translate by the centroid and scale by the scaling factor 2N ( )1/2 ΣN ||x −x¯||2 i=1 i , for each respective image. Afterwards, we normalize the coordinates: q = Tp q' = T'p' (15) i i i i Using the new, normalized coordinates, 우리는 할 수 있습니다 compute the new F using q the regular least-squares Eight Point Algorithm....


## 5. Image Rectification

Recall that an interesting case for 에피폴라 geometry occurs when two images are parallel to each other. Let us first compute the Essential matrix E in the case of parallel 이미지 평면s. We can assume that the two cameras have the 9 same K and that there is no relative 회전 between the cameras (R = I). In this case, let us assume that there is only a 평행 이동 along the x axis, giving T = (T ,0,0). This gives x  


## 0. 0 0

E = [T × ]R = 0 0 −T x (17)


## 0. T 0

x Once E is known, 우리는 할 수 있습니다 find the directions of the 에피폴라 lines associ- ated with points in the 이미지 평면s. Let us compute the direction of the 에피폴라 line (cid:96) associated with point p':     


## 0. 0 0 u(cid:48) 0

(cid:96) = Ep' = 0 0 −T xv'  = −T x (18)


## 0. T 0 1 T v(cid:48)

x x We can see that the direction of (cid:96) is horizontal, as is the direction of (cid:96)', which is computed in a similar manner. Figure 7: The process of image rectification involves computing two homo- graphies that 우리는 할 수 있습니다 apply to a pair of images to make them parallel. If we use the 에피폴라 constraint pTEp' = 0, then we arrive at the fact that v = v', demonstrating that p and p' share the same v-coordinate. Con- sequently, there exists a very straightforward relationship between the cor- responding points. Therefore, rectification, or the process of making any two given images parallel, becomes useful when discerning the relationships between corresponding points in images. 10 Figure 8: The rectification problem setup: we compute two homographies that 우리는 할 수 있습니다 apply to the 이미지 평면s to make the resulting planes parallel. Rectifying a pair of images does not require knowledge of the two camera matricesK,K' ortherelativetransformationR,T betweenthem. Instead,우리는 할 수 있습니다 use the Fundamental matrix estimated by the Normalized Eight Point Algorithm. Upon getting the Fundamental matrix, 우리는 할 수 있습니다 compute the 에피폴라 lines (cid:96) and (cid:96)' for each correspondence p and p'. i i i i Fromthesetof에피폴라lines,wecanthenestimatetheepipoleseande' of each image. This is because we know that the epipole lies in the intersection of all the 에피폴라 lines. In the real world, due to noisy measurements, all the 에피폴라 lines will not intersect in a single point. Therefore, computing the epipole 될 수 있습니다 found by minimizing the least squared error of 피팅 a point to all the 에피폴라 lines. Recall that each 에피폴라 line 될 수 있습니다 represented as a vector (cid:96) such that all points on the line (represented in 동차 좌표) are in the set {x|(cid:96)Tx = 0}. If we define each 에피폴라 line as T (cid:96) = (cid:96) (cid:96) (cid:96) , then 우리는 할 수 있습니다 we formulate a linear system of equations i i,1 i,2 i,3 and solve using SVD to find the epipole e:   (cid:96)T 1 ....


## 1. 2

to map the epipoles to infinity. Let us start by finding a homography H that 2 11 mapsthesecondepipolee' toapointonthehorizontalaxisatinfinity(f,0,0). Since there are many possible choices for this homography, we should try to choose something reasonable. One condition that leads to good results in practice is to insist that the homography acts like a transformation that applies a 평행 이동 and 회전 on points near the center of the image. Thefirststepinachievingsuchatransformationistotranslatethesecond image such that the center is at (0,0,1) in 동차 좌표. We can do so by applying the 평행 이동 matrix 


## 1. 0

−width 2 T = 0 1 −height  (20) 2


## 0. 0 1

After applying the 평행 이동, we apply a 회전 to place the epipole on the horizontal axis at some point (f,0,1). If the translated epipole Te' is located at 동차 좌표 (e',e',1), then the 회전 applied is


## 1. 2

 α√ e' 1 α√ e' 2 0  e'2+e'2 e'2+e'2  1 2 1 2  R = −α√ e' 2 α√ e' 1 0 (21)  e'2+e'2 e'2+e'2 


## 0. 0 1

where α = 1 if e' ≥ 0 and α = −1 otherwise. After applying this 회전, 1 notice that given any point at (f,0,1), bringing it to a point at infinity on the horizontal axis (f,0,0) only requires applying a transformation  


## 1. 0 0

G =  0 1 0  (22) −1 0 1 f After applying this transformation, we finally have an epipole at infinity, so 우리는 할 수 있습니다 translate back to the regular image space. Thus, the homography H 2 that we apply on the second image to rectify 그것은 H = T−1GRT (23) 2 Now that a valid H is found, we need to find a 매칭 homography H


## 2. 1

for the first image. We do so by finding a transformation H that minimizes 1 the sum of square distances between the corresponding points of the images (cid:88) argmin (cid:107)H p −H p'(cid:107)2 (24)


## 1. i 2 i

H1 i 12 Although the derivation2 is outside the scope of this class, 우리는 할 수 있습니다 actually prove that the 매칭 H is of the form: 1 H = H H M (25)


## 1. A 2

where F = [e] M and ×   a a a


## 1. 2 3

H A = 0 1 0 (26)


## 0. 0 1

with (a ,a ,a ) composing the elements of a certain vector a that will be


## 1. 2 3

computed later. First, we need to know what M is. An interesting property of any 3×3 skew-symmetric matrix A is A = A3 up to scale. Because any cross product matrix [e] is skew-symmetric and that 우리는 할 수 있습니다 only know the Fundamental × matrix F up to scale, then F = [e] M = [e] [e] [e] M = [e] [e] F (27) × × × × × × By grouping the right terms, 우리는 할 수 있습니다 find that M = [e] F (28) × Notice that if the columns of M were added by any scalar multiple of e, then the F = [e] M still holds up to scale. Therefore, the more general case of × defining M is M = [e] F +evT (29) × for some vector v. In practice, defining M by setting vT = 1 1 1 works very well. TofinallysolveforH ,weneedtocomputetheavaluesofH . Recallthat


## 1. A

wewanttofindaH ,H tominimizetheproblemposedinEquation24. Since


## 1. 2

we already know the value of H and M, then 우리는 할 수 있습니다 substitute pˆ = H Mp


## 2. i 2 i

and pˆ' = H p' and the minimization problem becomes i 2 i (cid:88) argmin (cid:107)H pˆ −pˆ'(cid:107)2 (30) A i i HA i In particular, if we let pˆ = (xˆ ,yˆ,1) and pˆ' = (xˆ',yˆ',1), then the mini- i i i i i i mization problem 될 수 있습니다 replaced by: (cid:88) argmin (a xˆ +a yˆ +a −xˆ')2 +(yˆ −yˆ')2 (31)


## 1. i 2 i 3 i i i

a i 2If you are interested in the details, please see Chapter 11 of Hartley & Zisserman’s textbook Multiple View Geometry 13 Since yˆ −yˆ' is a constant value, the minimization problem further reduces i i to (cid:88) argmin (a xˆ +a yˆ +a −xˆ')2 (32)


## 1. i 2 i 3 i

a i Ultimately, this breaks down into solving a least-squares problem Wa = b for a where     xˆ yˆ 1 xˆ'


## 1. 1 1

. . W =  . .  b =  . .  (33)     xˆ yˆ 1 xˆ' n n n After computing a, 우리는 할 수 있습니다 compute H and finally H . Thus, we gen- A 1 erated the homographies H ,H to rectify any image pair given a few corre-


## 1. 2

spondences. 14


## 요약

이 강의에서는 에피폴라 기하학의 주요 개념과 방법론을 다뤘습니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [03-epipolar-geometry.pdf](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)
