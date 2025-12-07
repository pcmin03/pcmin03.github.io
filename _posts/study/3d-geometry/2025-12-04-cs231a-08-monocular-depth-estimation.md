---
title: "[CS231A] Lecture 08: Monocular Depth Estimation (단안 깊이 추정)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Monocular, Depth, Estimation]
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

이 포스트는 Stanford CS231A 강의의 08번째 강의 노트인 "Monocular Depth Estimation"를 한글로 정리한 것입니다.

**원본 강의 노트**: [08-monocular_depth_estimation.pdf](https://web.stanford.edu/class/cs231a/course_notes/08-monocular_depth_estimation.pdf)

<!--more-->

## 강의 개요

이 강의에서는 단안 깊이 추정에 대해 다룹니다.


## 강의 노트 페이지 이미지

<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
  <div><img src='/assets/images/posts/cs231a-08/page_1.png' alt='Page 1' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_2.png' alt='Page 2' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_3.png' alt='Page 3' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_4.png' alt='Page 4' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_5.png' alt='Page 5' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_6.png' alt='Page 6' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_7.png' alt='Page 7' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_8.png' alt='Page 8' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_9.png' alt='Page 9' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_10.png' alt='Page 10' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_11.png' alt='Page 11' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_12.png' alt='Page 12' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_13.png' alt='Page 13' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_14.png' alt='Page 14' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-08/page_15.png' alt='Page 15' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
</div>


## 주요 수식

이 강의에서 다루는 주요 수식들:

**수식 1:**

$$We can use similar triangles as illustrated in Figure 3 to obtai z = fb, where$$


## 1. Overview

In the previous section, we discussed the idea of 표현 학습 which leverages unsupervised and self-supervised methods to learn an intermediate, low-dimensional표현ofhigh-dimensionalsensorydata. Theselearned features can then be used to solve downstream visual inference tasks. Here, we will examine how 표현 학습 in the context of two common 컴퓨터 비전 problems: monocular 깊이 추정 and feature tracking.


## 2.1. Background

Depth 추정 is common 컴퓨터 비전 building block that is crucial to tackling more complex tasks, such as 3D reconstruction and spatial perception for grasping in robotics or navigation for autonomous vehicles. There are nu- merous active methods for 깊이 추정 such as structured light 스테레오 and LIDAR (3D 점 clouds), but we focus on 깊이 추정 through passive means here since it does not require specialized, possibly expensive hardware and can work better in outdoor situations. We can view 깊이 추정 as a special case of the correspondence problem, which is fundamental in 컴퓨터 비전. It involves finding the 2D locations corresponding to the 투영s of a physical 3D 점 onto multiple 2D images taken of a 3D scene. The 2D frames 될 수 있습니다 captured from multiple viewpoints either using a monocular or a 스테레오 camera. 1 Figure 1: Epipolar geometry setup with a 스테레오 camera. One way to solve for correspondences is through 에피폴라 geometry, illus- trated in Figure 1, as we have seen earlier in the course. Recall that given the camera centers O and O and a 3D 점 in the scene called P, p and p′ rep-


## 1. 2

resent the 투영 of P into the 이미지 평면s for the left and right cameras, respectively. Given p in the left image, we know that the corresponding point in the right image p′ must lie somewhere on the 에피폴라 line of the right cam- era, which we defined as the intersection of the 이미지 평면s with the 에피폴라 plane. This is known as the 에피폴라 constraint, which is encapsulated by thefundamental(oressential)matrixbetweenthetwocamerassinceF givesus the known 에피폴라 lines. In the context of 깊이 추정, we often assume that we are dealing with a 스테레오 setup and rectified images. The 에피폴라 lines are then horizontal and the disparity is defined as the (horizontal) distance between the two corresponding points such that d = p′ −p and p +d = p′ u u u u (note that p′ >p for all P). u u 2 Figure 2: Rectified setup with parallel 이미지 평면s and epipoles at infinity. We then see that there is simple inverse relationship between disparity and 깊이, whichisdefinedasthez-coordinateofP relativetothecameracenters. We can use similar triangles as illustrated in Figure 3 to obtai z = fb, where d f is the 초점 거리 of the cameras and b is the length of the baseline between the two cameras (yellow dashed line in Fig 2). Assuming b and the camera intrinsics K are known, we see that if we are able to find correspondences between two rectified images, illustrated in Figure 2, we know their disparity and thus their 깊이. One approach to identify correspondences p′ for p is to runasimple1D-searchalongthe에피폴라lineintheotherimage,usingpixelor patchsimilaritiestodeterminethelocationofthemostlikelyp′. However, such a naive method would run into issues such as occlusions, repetitive patterns, and homogeneous regions (i.e. lack of texture) on real-world images. We turn to modern 표현 학습 methods instead. Figure 3: Relationship between 깊이 and disparity. 3


## 2.2. Supervised Estimation

Here, we focus on the task of monocular (single-view) 깊이 추정: weonlyhaveasingleimageavailableattesttime,andnoassumptionsaboutthe scenecontentsaremade. Incontrast,스테레오(multi-view)깊이추정 methods perform inference with multiple images. Monocular 깊이 추정 is an underconstrained problem, i.e. geometrically 그것은 impossible to determine the깊이ofeachpixelintheimage. However, humanscanestimate깊이well withasingleeyebyexploitingcuessuchasperspective,scaling,andappearance via lighting and occlusion. Therefore, when exploiting these cues, computers shouldbeabletoinfer깊이withjustasingleimage. Fullysupervised학습 methods, illustrated in Figure 4, rely on training models (CNNs) to learn to predict pixel-wise disparity over pairs of ground truth 깊이 and RGB camera frames [8, 11]. The training loss captures the similarity between the predicted and ground-truth 깊이 and the 학습 method aims to minimize that loss. Sincemonocularmethodscanonlycapture깊이up-to-scale,[1]proposesusing a scale-invariant error to prior monocular methods. Figure 4: Vanilla supervised 학습 setup used in [1, 8, 11]. Figure from [3].


## 2.3. Unsupervised Estimation

While supervised 학습 methods achieved decent results, they are limited to scene types where large quantities of ground 깊이 data are available. This motivates unsupervised 학습 methods which only require the input RGB frame data and a 스테레오 camera with known intrinsics, and thereby avoid the need for expensive labeling efforts. Here, we examine the approach proposed in [3] as a case study. Instead of using the difference between reconstructed andgroundtruth깊이astheloss,thebaseunsupervisedformulationcaststhe problemasimagereconstructionthroughtheuseofanautoencodertominimize thedifferencebetweentheinputreferenceimageandareconstructedversionI˜l. 4 Figure 5: Unsupervised baseline network. The differentiable sampler enables end-to-end optimization. The baseline network, shown in Figure 5, only reconstructs the left image Il. The input to the network is Il, the left frame. A CNN maps the left frame to an output to dl, the disparity (displacement) values required to warp the right image Ir into the left image. The disparity values are then used as an intermediate 표현 to reconstruct the left image, I˜l, by sampling from the right image. We could sample from the right image as I˜l(u,v) = Ir(u−dl(u,v),v), but dl(u,v) is not necessarily an integer so the pixel at the exact new location may not exist. To perform end-to-end optimization to train the network, a fully (sub-) differentiable bilinear sampler [6] is used, illustrated in Figure ??. Note that both the left and right images are used to train the network, but only the left image is required to infer the left-aligned 깊이 at test time. The network architecture is fully convolutional, consisting of an encoder followed by adecoder,whichoutputsmultipledisparitymapsatdoublingspatialscales. For instance, if the first disparity map is of resolution (D ,D ), the second output h w disparity map would be of resolution (2D ,2D )....


## 1. (cid:88)

Cl = |dl −dr | (4) lr N ij ij+dl ij i,j Theintuitionhereisthattheabsolutedistancebetweencorrespondingpixels intwoimageplanes,i.e. disparity,shouldbethesamewhethercomputedright to left or left to right. Therefore, we should penalize differences between the predicted disparities for the left and right images that are outputted from the network. To achieve this, we iterate through each pixel i,j and calculate the L1 distance between left-aligned disparity d˜l and corresponding right-aligned i,j disparity dr . Note here that we are using the disparities to ”sample” from i,j+dl ij the disparity maps, not the images. In the results, the authors demonstrate that this unsupervised setup outperforms both the baseline and existing state- of-the-art fully supervised methods.


## 2.4. Self-Supervised Estimation

Unsupervised methods have been followed by self-supervised 학습 for 깊이 추정; here we review a follow-up paper [4]. Here, the 깊이 추정 problem is framed as novel view synthesis prob- lem. Given a target image of a scene, the learned pipeline aims to predict what the scene would look like from another viewpoint. Depth is used as an interme- diate표현toobtainthenovelview,sothe깊이mapcanbeextracted fromthepipelineattesttimeforuseinothertasks. Inthemonocularsetup,we rely on monocular video as our supervisory signal: our target image is a color frame I at time t, and the images from other viewpoints, or the source views, t are the temporally adjacent frames I ∈ {I ,I }. Since we are predicting t′ t−1 t+1 futureandpastinputsfromthecurrentinput, asopposedtoreconstructingthe entire original input, 이것은 an example of self-supervised 학습. 7 Figure 7: Self-supervised 깊이 추정 pipeline. (a) is the 깊이 network architecture for extracting the 깊이 map, (b) predicts the pose transformation from frame to frame, and (c) describes the ambiguity in 깊이 for re투영 in all frames. The pipeline is illustrated in Figure 7. First, we obtain the intermediate 깊이 표현: given the color input I , we run it through a convolution t encoder-decoderarchitecturetoobtainthe깊이mapD ,asshowninFigure7a. t In parallel, as shown in Figure 7b, we iterate over the source past and future frames I and compute the relative pose T indicating the transformation t′ t→t′ fromIt toIt′. AssumingthesamecameraintrinsicsK foralltargetandsource views, 우리는 할 수 있습니다 obtain our novel views for t−1,t+1 through re투영, as shown in Figure 7c. Given K and the predicted 깊이 D for all pixels, we t can backproject specific 2D image coordinates (u,v) to the 3D 점 location. Since we know the relative pose T , 우리는 할 수 있습니다 then project our 3D 점 in the t→t′ 이미지 평면 of source view I to obtain the 2D coordinates (u′,v′) in the novel t 이미지 평면 I . Since we have the monocular video, the ground truth I is t′ t′ known....


## 4. Motivation

Givenasequenceofimages,thetaskof feature tracking involvestrackingthe locations of a set of 2D points across all the images, illustrated in Figure ??. Justlike깊이추정,wecanviewfeaturetrackingasyetanotherinstance of solving for correspondence problem across an image sequence. Figure 8: Feature point tracking over time. Featuretrackingcanbeusedtotracethemotionofobjectsinthescene. We make no assumptions about scene contents or camera motion; the camera may be moving or stationary, and the scene may contain multiple moving or static objects. Thechallengeinfeaturetrackingliesidentifyingwhichfeaturepointswecan efficientlytrackoverframes. Theappearanceofimagefeaturescanchangedras- tically over frames due to camera movement (feature completely disappears), shadows, or occlusion. Small errors can also accumulate as the appearance model for feature tracking is updated, leading to drift. Our goal is to identify distinct regions (called features or sometimes keypoints) that 우리는 할 수 있습니다 track eas- ilyandconsistently,andthenapplysimpletrackingmethodstocontinuallyfind these correspondences. 9 Figure 9: Descriptors for feature tracking. Traditionally, distinct features in images that are easy to track have been detected and tracked using hand-designed method [5, 10, 9, 12, 7]. Specifically, these good features then need to be encoded into a so-called descriptor that lends itself well for fast 매칭 with features in other images, i.e. finding correspondences. These methods are also sparse, only yielding descriptors for a subsetofpixelsintheimage. Inthissection,wewilllookathow표현 학습 can also be used to learn descriptors of image features rather than hand-designing them....


## 5. Learned Dense Descriptors

We examine the method for 학습 dense descriptors proposed in [2]. 10 Figure 10: Representation of dense descriptors. Givenaninputcolorimage, wewanttolearnamappingf(·)thatoutputsa D-dimensionaldescriptorforeverypixelinthecolorimage. ”Dense”heremeans that we have a descriptor for every point in the input image, not just a sparse set. For visualization purposes in Figure 10, the D-dimensional descriptors are mappedtoRGBthroughdimensionalityreduction. Inpractice,f(·)isalearned neuralnetworkwithaconvolutionalencoder-decoderarchitecture. Thenetwork istrainedonpairsofimagesofthesameobjectfromdifferentviews(I ,I )using a b apixel-contrastive loss,whichattemptstomirrorthe“contrast”inpixelsby minimizing the distance for similar descriptors and maximizing the distance for different descriptors. Figure 11: Loss for matches. The blue arrow indicates a 매칭 correspon- dence between the two points at the end of the arrow. Weassumethatwearegivenalistofcorrespondences(herecalledmatches) for an image pair. We run the network to compute the descriptors for all pointsintheimage. Foreachgroundtruthmatch, wecalculatetheL2distance between the descriptors at the two corresponding points. We want to minimize the distance in descriptor space D(I ,u ,I ,u )2. a a b b 11


## 1. (cid:88)

L (I ,I )= D(I ,u ,I ,u )2 (7) matches a b N a a b b matches Nmatches Figure 12: Loss for non-matches. The blue arrow a pair of non-corresponding points. For the contrastive part, we also compute the loss term for non-matches (pairs of points that do not correspond to each other). Here, we want to maxi- mize the distance between non-corresponding points (the max operation maxi- mizes this distance up to M, the maximum distance), given by


## 1. (cid:88)

L (I ,I )= max(0,M −D(I ,u ,I ,u )2) non-matches a b N a a b b non-matches Nnon-matches (8) Assuming the true correspondence is known for u , 그것은 easy to find pairs a of non-matches: 우리는 할 수 있습니다 just sample arbitrary points from I that are not the b corresponding point. Note that in Figure 12, u refers to a non-matched point, b whileinFigure7,u referstothematchedpoint. Thetotallossisthenthesum b of the two L(I ,I )=L (I ,I )+L (I ,I ) (9) a b matches a b non-matches a b The challenge lies in cheaply obtaining ground truth correspondences at scale with minimal human assistance. To this end, the authors propose using a robotic setup to perform autonomous and self-supervised data collection. 12 Figure 13: Robotic arm capturing different views and corresponding poses of a stationary object for training. A robotic arm is used to capture images of a stationary object at various poses, illustrated in Figure 13. Since the forward kinematics of this precise robotic arm are known, we have matched pairs of camera pose and the corre- sponding view. 3D reconstruction is performed using all views to obtain a 3D model of the object. Using the camera poses, 3D 점s, and images, 우리는 할 수 있습니다 now generate as many ground-truth correspondences as we want. The network is trained using Equation 9 in combination with several other tricks such as background randomization, data augmentation, and hard-negative scaling. 13 Figure 14: Cross-object loss. The two images at the bottom correspond to the distinct cluster in the plot on the right (blue, orange) when using the cross- object loss. If we only train on pairs of the same object, the learned descriptors for dif- ferent objects overlap when they shouldn’t since they correspond to completely different entities. If we incorporate a cross-object loss (pixels from images of two different objects are all non-matches), then we see distinct clusters forming in the descriptor space, as show in Figure 14. Figure 15: Class-consistent descriptors....


## 요약

이 강의에서는 단안 깊이 추정의 주요 개념과 방법론을 다뤘습니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [08-monocular_depth_estimation.pdf](https://web.stanford.edu/class/cs231a/course_notes/08-monocular_depth_estimation.pdf)
