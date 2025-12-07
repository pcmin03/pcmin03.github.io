---
title: "[CS231A] Lecture 05: Active and Volumetric Stereo (능동 및 볼륨 스테레오)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Active, and, Volumetric, Stereo]
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

이 포스트는 Stanford CS231A 강의의 05번째 강의 노트인 "Active and Volumetric Stereo"를 한글로 정리한 것입니다.

**원본 강의 노트**: [05-active-volumetric-stereo.pdf](https://web.stanford.edu/class/cs231a/course_notes/05-active-volumetric-stereo.pdf)

<!--more-->

## 강의 개요

이 강의에서는 능동 및 볼륨 스테레오에 대해 다룹니다.


## 강의 노트 페이지 이미지

<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
  <div><img src='/assets/images/posts/cs231a-05/page_1.png' alt='Page 1' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_2.png' alt='Page 2' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_3.png' alt='Page 3' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_4.png' alt='Page 4' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_5.png' alt='Page 5' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_6.png' alt='Page 6' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_7.png' alt='Page 7' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_8.png' alt='Page 8' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_9.png' alt='Page 9' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_10.png' alt='Page 10' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_11.png' alt='Page 11' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-05/page_12.png' alt='Page 12' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
</div>


## 1. Introduction

In traditional 스테레오, 주요 idea is to use corresponding points p and p' to estimate the location of a 3D 점 P by triangulation. A key challenge here, is to solve the correspondence problem: how do we know whether a point p actually corresponds to a point p' in another image? This problem is further accentuated by the fact that we need to handle the many 3D 점s that are present in the scene. The focus of these notes will discuss alternative techniques that work well in reconstructing the 3D structure.


## 2. Active stereo

Figure 1: The active 스테레오 setup that projects a point into 3D space. First, wewillanintroduceatechniqueknownasactive 스테레오thathelps mitigate the correspondence problem in traditional 스테레오. The main idea of active 스테레오 is to replace one of the two cameras with a device that interacts 1 with the 3D environment, usually by projecting a pattern onto the object thatiseasilyidentifiablefromthesecondcamera. Thisnewprojector-camera pair defines the same 에피폴라 geometry that we introduced for camera pairs, wherebytheimageplaneofthereplacedcameraisreplacedwithaprojector virtual plane. In Figure 1, the projector is used to project a point p in the virtual plane onto the object in 3D space, producing a point in 3D space P. This 3D 점 P should be observed in the second camera as a point p'. Because we know what we are projecting (e.g. the position of p in the virtual plane, the color and intensity of the 투영, etc.), 우리는 할 수 있습니다 easily discover the corresponding observation in the second camera p'. Figure 2: The active 스테레오 setup that projects a line into 3D space. A common strategy in active 스테레오 is to project from the virtual plane a verticalstripesinsteadofasinglepoint. Thiscaseisverysimilartothepoint case, where the line s is projected to a stripe in 3D space S and observed as a line in the camera as s'. If the projector and camera are parallel or rectified, then 우리는 할 수 있습니다 discover the corresponding points easily by simply intersecting s' with the horizontal 에피폴라 lines. From the correspondences, 우리는 할 수 있습니다 use the triangulation methods introduced in the previous course notes to reconstruct all the 3D 점s on the stripe S. By swiping the line across the scene and repeating the process, 우리는 할 수 있습니다 recover the entire shape of all visible objects in the scene. Notice that one requirement for this algorithm to work is that the pro- jector and the camera need to be calibrated....


## 3. Volumetric stereo

Figure 4: The setup of volumetric 스테레오, which takes points from a limited, working volume and performs consistency checks to determine 3D shape. An alternative to both the traditional 스테레오 and active 스테레오 approach is volumetric 스테레오, which inverts the problem of using correspondences to find 3D structure. In volumetric 스테레오, we assume that the 3D 점 we are trying to estimate is within some contained, known volume. We then project the hypothesized 3D 점 back into the calibrated cameras and validate whether these 투영s are consistent across the multiple views. Figure 4 illustrates the general setup of the volumetric 스테레오 problem. Because these techniques assume that the points we want to reconstruct are contained by a limited volume, these techniques are mostly used for recovering the 3D models of specific objects as opposed to recovering models of a scene, which may be unbounded. The main tenet of any volumetric 스테레오 method is to first define what it means to be “consistent” when we reproject a 3D 점 in the contained volume back into the multiple image views. Thus, depending on the defi- nition of the concept of consistent observations, different techniques 될 수 있습니다 introduced. In these notes, we will briefly outline three major techniques, which are known as space carving, shadow carving, and voxel coloring. 4


## 3.1. Space carving

Figure 5: The silhouette of an object we want to reconstruct contains all pixels of the visible portion of the object in the image. The visual cone is the set of all possible points that can project into the silhouette of the object in the image. The idea of space carving is mainly derived from the observation that the contours of an object provide a rich source of geometric information about the object. In the context of multiple views, let us first set up the problem illustrated in Figure 5. Each camera observes some visible portion of an object, from which a contour 될 수 있습니다 determined. When projected into the 이미지 평면, this contour encloses a set of pixels known as the silhouette of the object in the 이미지 평면. Space carving ultimately uses the silhouettes of objects from multiple views to enforce consistency. However, if we do not have the information of the 3D object and only im- ages, then how can we obtain silhouette information? Luckily, one practical advantage of working with silhouettes is that they 될 수 있습니다 easily detected in images if we have control of the background behind the object that we want to reconstruct. For example, 우리는 할 수 있습니다 use a “green screen” behind the object to easily segment the object from its background. Now that we have the silhouettes, how can we actually use them? Recall that in volumetric 스테레오, we have an estimate of some volume that we guar- antee that the object can reside within. We now introduce the concept of a visual cone, which is the enveloping surface defined by the camera center and the object contour in the 이미지 평면. By construction, 그것은 guaranteed 5 that the object will lie completely in both the initial volume and the visual cone. Figure 6: The process of estimating the object from multiple views involves recovering the visual hull, which is the intersection of visual cones from each camera. Therefore, if we have multiple views, then 우리는 할 수 있습니다 compute visual cones for each view. Since, by definition, the object resides in each of these visual cones, then it must lie in the intersection of these visual cones, as illustrated in Figure 6....


## 3.2. Shadow carving

Tocircumventtheconcavityproblemposedbyspacecarving, weneedtolook to other forms of consistency checks. One important cue for determining the 3D shape of an object that 우리는 할 수 있습니다 use is the presence of self-shadows. Self- shadows are the shadows that an object projects on itself. For the case of concave objects, an object will often cast self-shadows in the concave region. Shadow carving at its core augments space carving with the idea of using self-shadows to better estimate the concavities. As shown in Figure 9, the general setup of shadow carving is very similar to space carving. An ob- ject is placed in a turntable that is viewed by a calibrated camera. However, there is an array of lights in known positions around the camera who states 될 수 있습니다 appropriately turned on and off. These lights will be used to make 8 Figure 9: The setup of shadow carving, which augments space carving by adding a new consistency check from an array of lights surrounding the cam- era. the object cast self-shadows. As shown in Figure 10, the shadow carving process begins with an initial voxel grid, which is trimmed down by using the same approach as in space carving. However, in each view, 우리는 할 수 있습니다 turn on and off each light in the array surrounding the camera. Each light will produce a different self-shadow on the object. Upon identifying the shadow in the 이미지 평면, 우리는 할 수 있습니다 then find the voxels on the surface of our trimmed voxel grid that are in the visual cone of the shadow. These surface voxels allow us to then make a new visual cone with the image source....


## 3.3. Voxel coloring

The last technique we cover in volumetric 스테레오 is voxel coloring, which uses color consistency instead of contour consistency in space carving. As illustrated in Figure 11, suppose that we are given images from multi- ple views of an object that we want to reconstruct. For each voxel, we look at its corresponding 투영s in each of the images and compare the color of each of these 투영s. If the colors of these 투영s sufficiently match, then we mark the voxel as part of the object. One benefit of voxel coloring not present in space carving is that color associated with the 투영s 될 수 있습니다 transferred to the voxel, giving a colored reconstruction. Overall, there are many methods that one could use for the color con- sistency check. One example would be to set a threshold between the color similarity between the 투영s. However, there exists a critical assump- tion for any color consistency check used: the object being reconstructed must be Lambertian, which means that the perceived luminance of any part of the object does not change with viewpoint location or pose. For 10 Figure 11: The setup of voxel coloring, which makes a consistency check of the color of all 투영s of a voxel. non-Lambertian objects, such as those made of highly reflective material, 그것은 easy to conceive that the color consistency check would fail on voxels that are actually part of the object. Figure 12: An example of an ambiguous case of vanilla voxel coloring. One drawback of vanilla voxel coloring is that it produces a solution that is not necessarily unique, as shown in Figure 12. Finding the true, unique solution complicates the problem of reconstruction by voxel coloring. 11 It is possible to remove the ambiguity in the reconstruction by introducing a visibility constraint on the voxel, which requires that the voxels be traversed in a particular order. In particular, we want to traverse the voxels layer by layer, starting with voxels closer to the cameras and then progress to further away voxels....


## 요약

이 강의에서는 능동 및 볼륨 스테레오의 주요 개념과 방법론을 다뤘습니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [05-active-volumetric-stereo.pdf](https://web.stanford.edu/class/cs231a/course_notes/05-active-volumetric-stereo.pdf)
