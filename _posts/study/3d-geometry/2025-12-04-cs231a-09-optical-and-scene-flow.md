---
title: "[CS231A] Lecture 09: Optical and Scene Flow (광학 흐름 및 장면 흐름)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Optical, and, Scene, Flow]
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

이 포스트는 Stanford CS231A 강의의 09번째 강의 노트인 "Optical and Scene Flow"를 한글로 정리한 것입니다.

**원본 강의 노트**: [09-optical-flow.pdf](https://web.stanford.edu/class/cs231a/course_notes/09-optical-flow.pdf)

<!--more-->

## 강의 개요

이 강의에서는 광학 흐름 및 장면 흐름에 대해 다룹니다.


## 강의 노트 페이지 이미지

<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
  <div><img src='/assets/images/posts/cs231a-09/page_1.png' alt='Page 1' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-09/page_2.png' alt='Page 2' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-09/page_3.png' alt='Page 3' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-09/page_4.png' alt='Page 4' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-09/page_5.png' alt='Page 5' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-09/page_6.png' alt='Page 6' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-09/page_7.png' alt='Page 7' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-09/page_8.png' alt='Page 8' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
</div>


## 주요 수식

이 강의에서 다루는 주요 수식들:

**수식 1:**

$$where x′ = [dx,dy,dz]T represents the motion of a 3D point and M ∈ R2×3$$

**수식 2:**

$$flow vector for each pixel is then given as u=[u,v]T. In the following sections, we describe the Lucas-Kanade method, which uses a semi-local approach to independently solve for u different pixel patches using least-squares.$$

**수식 3:**

```
One common simplification is to use ∆t=1 (consecutive frames), such that the velocities are equivalent to the displacements u=∆x and v =∆y. We can then obtain I(x,y,t)=I(x+u,y+u,t+1) as in the lecture slides. Next, by the small motion assumption, we assume the motion (∆t,∆y) is small from frame to frame. This allows us to linearize I with a first-order Taylor series expansion, illustrated by Equation (3).
```

**수식 4:**

$$We recognize this as linear system in the form of Ax=b. ∇I =[I ,I ]T ∈$$

**수식 5:**

$$Figure 4: Solution set for (u,v) as a line in the form of y =mx+b.$$

**수식 6:**

$$pixels, where p =[x ,y ]T is the location of the i−th pixel. Assuming N2 >2,$$

**수식 7:**

$$[1] https://commons.wikimedia.org/w/index.php?curid=10588443 By Njw000$$


## 1. Overview

Given a video, 광학 흐름 is defined as a 2D vector field describing the ap- parent movement of each pixel due to relative motion between the camera (ob- server)andthescene(objects,surfaces,edges). Thecameraorthesceneorboth may be moving. Figure 1 shows a fly rotating in a counter-clockwise direction (from the fly’s point-of-view). Although the scene is static, the 2D 광학 흐름 of apparent motion indicates a 회전 in the opposite (clockwise) direction around the origin. Figure 1: Optical flow example [3].


## 1.1. Motion field

Optical flow is not to be confused with the motion field, a 2D vector field describing the 투영 of 3D motion vectors for points in the scene onto the 이미지 평면 of the observer. Figure 2 illustrates the motion field in a simple 2D case (imagine viewing a 3D scene from a top-down perspective). The 2D object point P is projected to a a1D point P in the 이미지 평면 as viewed o i by observer O. If the object point P is displaced by V ·dt (called the motion o o vector), thecorrespondingprojected1DpointmovesbyV ·dt. The1Dmotion i field here consists of all velocity values V for all i located in the 이미지 평면. i 1 Figure 2: Example of a motion field for a 2D scene [5]. Generalizing to a 3D scene, the motion field for pixel (x,y) is given by  u  dx dt  = =Mx′ (1) v dy dt where x′ = [dx,dy,dz]T represents the motion of a 3D 점 and M ∈ R2×3 dt dt dt contains the partial derivatives of pixel displacement with respect to the 3D 점 locations. Themotionfieldisanideal2D표현of3Dmotionasprojectedonto the 이미지 평면. It is the “ground truth” that 우리는 할 수 있습니다not observe directly; 우리는 할 수 있습니다onlyestimatetheopticalflow(apparentmotion)fromournoisyobservations (video). It is important to note that the 광학 흐름 is not always the same as the motion field. For instance, a uniform rotating sphere with a fixed light source has no 광학 흐름 but a non-zero motion field. In contrast, a fixed uniformspherewithalightsourcemovingaroundithasanon-zeroopticalflow but a zero motion field. These two cases are illustrated in Figure 3. 2 Figure 3: Physical vs. optical correspondence [4].


## 2. Computing the optical flow

Wedefineavideoasanorderedsequenceofframescapturedovertime. I(x,y,t), a function of both space and time, represents the intensity of pixel (x,y) in the frame at time t. In dense 광학 흐름, at every time t and for every pixel (x,y), we want to compute the apparent velocity of the pixel in both the x-axis and y-axis, given by u(x,y,t) = ∆x and v(x,y,t) = ∆y, respectively. The optical ∆t ∆t flow vector for each pixel is then given as u=[u,v]T. In the following sections, we describe the Lucas-Kanade method, which uses a semi-local approach to independently solve for u different pixel patches using least-squares. From the brightness constancy assumption, 우리는 할 수 있습니다 assume that the apparentintensityintheimageplaneforthesameobjectdoesnotchangeacross differentframes. ThisisrepresentedbyEquation(2)forapixelthatmoved∆x and ∆y in the x and y directions between times t to t+∆t. I(x,y,t)=I(x+∆x,y+∆y,t+∆t) (2) One common simplification is to use ∆t=1 (consecutive frames), such that the velocities are equivalent to the displacements u=∆x and v =∆y. We can then obtain I(x,y,t)=I(x+u,y+u,t+1) as in the lecture slides. Next, by the small motion assumption, we assume the motion (∆t,∆y) is small from frame to frame. This allows us to linearize I with a first-order Taylor series expansion, illustrated by Equation (3). 3 ∂I ∂I ∂I I(x+∆x,y+∆y,t+∆t)=I(x,y,t)+ ∆x+ ∆y+ ∆t+... ∂x ∂y ∂t (3) ∂I ∂I ∂I ≈I(x,y,t)+ ∆x+ ∆y+ ∆t ∂x ∂y ∂t The ... represents the higher-order terms in the Taylor series expansion which we subsequently truncate out in the next line. Substituting the result from Equation (3) into Equation (2), we arrive at the 광학 흐름 constraint equation: ∂I ∂I ∂I 0= ∆x+ ∆y+ ∆t ∂x ∂y ∂t ∂I ∆x ∂I ∆y ∂I (4) = + + ∂x ∆t ∂y ∆t ∂t =I u+I v+I x y t I ,I ,I are short-hand for the two spatial derivatives and time derivative, x y t respectively....


## 요약

이 강의에서는 광학 흐름 및 장면 흐름의 주요 개념과 방법론을 다뤘습니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [09-optical-flow.pdf](https://web.stanford.edu/class/cs231a/course_notes/09-optical-flow.pdf)
