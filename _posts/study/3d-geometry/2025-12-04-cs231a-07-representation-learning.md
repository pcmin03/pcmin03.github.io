---
title: "[CS231A] Lecture 07: Representation Learning (표현 학습)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Representation, Learning]
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

이 포스트는 Stanford CS231A 강의의 07번째 강의 노트인 "Representation Learning"를 한글로 정리한 것입니다.

**원본 강의 노트**: [07-representation-learning.pdf](https://web.stanford.edu/class/cs231a/course_notes/07-representation-learning.pdf)

<!--more-->

## 강의 개요

이 강의에서는 표현 학습에 대해 다룹니다.


## 강의 노트 페이지 이미지

<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
  <div><img src='/assets/images/posts/cs231a-07/page_1.png' alt='Page 1' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_2.png' alt='Page 2' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_3.png' alt='Page 3' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_4.png' alt='Page 4' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_5.png' alt='Page 5' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_6.png' alt='Page 6' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_7.png' alt='Page 7' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_8.png' alt='Page 8' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_9.png' alt='Page 9' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_10.png' alt='Page 10' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
  <div><img src='/assets/images/posts/cs231a-07/page_11.png' alt='Page 11' style='width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'></div>
</div>


## 1.1. States and Representations

Thestateofadynamicalsystemisbasicallyacompresseddescriptionandcap- turesthekeyaspectsofthissystem. Theseaspectsmaybetime-varyingorfixed parameters. Consider the system shown in Figure 1: a walking robot moving along a vertical plane. In this example, the time-varying quantities may con- sist of information about the robot’s current pose (position and orientation) in space, joint configuration and velocity. Fixed parameters may be the length of the robot’s limbs, the weight of each limb or joint friction. Numerous represen- tations are possible for the system’s state; for instance, we could choose to use any selection of joint angles, velocities, positions or the pose of the robot. Fur- thermore, these quantities could be represented in different ways. For example, orientations of the robot body could be given as Euler angles, in the angle-axis 표현 or as a quaternion. Positions could be provided as spatial or po- lar coordinates. Which 표현 you choose often depends on the task at hand. Finally, a state 표현 could also be learned which we will come back to later. The state of a dynamical system is not static and evolves over time. For the example of the walker, its pose and joint configuration are time-varying. We denote the state at time t as x . A common assumption for such dynamical t Figure 1: Walker2D robot from the DeepMind Control Suite [4] 1 Figure 2: Probabilistic Graphical Model of a Markov chain. Figure 3: Probabilistic Graphical Model of a Hidden Markov Model. systems is that they possess the Markov property: future states only depend on the current state, not past states. In other words, state x is conditionally t+1 independent of past states x ,x ,...,x given the current state x :


## 1. 2 t−1 t

P(x |x ,x ,...,x ,x )=P(x |x ). (1) t+1 t t−1 2 1 t+1 t The Markov property has implications when predicting future state. If this property holds, future states only depend on the present - not on the past. Therefore,wecanpredictthestateatthenexttimestampgiventhecurrentstate according to specific dynamics of the system, or f(x ) = x . For instance, if t t+1 thestatecontainsthecurrentangularvelocityofajoint,wecanpredictthejoint angle at the next time step. f(·) may be a non-linear function, and we often assumethatthedynamicsareconstantovertimeforsimplicity. Wemodelsuch systems as Markov chains. Figure 2 visualizes the conditional dependence structure of a Markov chain as a probabilistic graphical model in which the nodes are the random variables and the edges indicate conditional dependence. Markov chains model dynamical systems as fully observable, i.e. the state is known. In practice, the state of a system at time t is often unknown because it cannot be directly observed. Our job is to estimate the true state from noisy sensor observations: a process called state 추정. Such systems are partially observable and 될 수 있습니다 modelled by hidden Markov models (or HMMs) since the state is ”hidden”. We assume that the states x behave as t a Markov process. We also assume that observation z at time t only depends t on the state x at that time....


## 1.2. Generative and Discriminative Approaches

How can we estimate the state from observations? Let us consider the task of estimating an object’s pose (state x) from an input RGB image (observation z). A 6D pose refers to both the objects 3D 평행 이동 and 3D 회전 (6 3 Figure 5: PoseCNN [6] is an example of a discriminative model. parameters total). Wedescribetwoapproachesatahigh-levelandwilldelveintospecificslater in the course. First, a generative model describes the joint probability dis- tribution p(z,x) given the observation z and state x. We compute this joint distribution using Bayes rule with the likelihood p(z|x) (see Eq. 2) and prior p(x). Forourexampleofobjectpose추정,wecouldsampleanobjectpose x(i) from p(x). This prior 될 수 있습니다 as simple as a uniform distribution in space or more complex and capture likely locations of this object (e.g. cars on roads rather than on the side of buildings). Given this pose sample, 우리는 할 수 있습니다 use the observationmodeltogeneratethemostlikelyobservationh(x(i))=z(i),i.e. the 2D 투영 of the object in the sampled pose onto the 이미지 평면. p(z|x(i)) then provides a likelihood of the actual observation z given the hypothesized pose x(i) by comparing the differences between z(i) and z. Second, a discriminative model describes the conditional probability p(x|z) of the state x given the observation z. For instance, we could train a neural network, such as PoseCNN in Figure 5, to directly map an input image z to the most likely output pose x. To learn more about the differences between discriminativeandgenerativemodels,pleaserefertotheCS229notesathttps: //cs229.stanford.edu/notes-spring2019/cs229-notes2.pdf.


## 2.1. Representations in Computer Vision

We turn our attention to how the term 표현s is used in Computer Vision. Consider the high-level, semantic segmentation pipeline in Figure 6. It illustrates that data at any stage within that pipeline comes in a specific repre- 4 Figure 6: Visualization of input, intermediate, and output 표현s in a high-level 컴퓨터 비전 pipeline for semantic segmentation [2]. sentation. Wewillthereforeintroducecontexttoqualifytheterm표현 more clearly. On one end of the pipeline, we have raw sensor data captured by some sensor. The input 표현 describes the raw sensor data format. In Figure 6, our input is an observation of a 3D scene within the ocean containing afish. Dependingonsensorchoice, rawsensordatacouldberepresentedby2D images, 깊이 images, or point clouds. Even within a specific form, there are subtle differences in the input 표현. For instance, color vs. grayscale, 스테레오 vs. monocular, RGB vs. HSV image color space. On the other end of the pipeline, the information being inferred from the raw sensor data is pro- vided in an output 표현s that succinctly describes the high-level key aspects of the scene that are necessary for the considered decision-making task....


## 2.2. Traditional CV and Interpretable Representations

Ahigh-levelvisualizationofthetraditional(≈pre-2012)computervisionpipeline is shown in Figure 7. The input data is compressed into an intermediate repre- sentationwhichusedtobecombinationsofhand-craftedfeaturesextractedfrom the input that comes in a specific input 표현. The specific choices of features (descriptors) is flexible and can eb made task specific. For instance, if we believe that specific stripes or designs on a fish’s body may distinguish it fromothertypesoffishorthingsinthesea,thenwecouldexploitthosespecific patternsbyextractingedgesandcolorstoformourintermediate표현 6 of the raw input data. In the example shown in Figure 7, the 표현 of the output is a class label. To infer this class label, the intermediate repre- sentation is fed into a classifier that can either be learned or is based on more manually-defined heuristics. As a simple example, I can define a heuristic that if the histogram of colors in the image contains a high amount of oranges and whites, we may say that the image is likely to contain a clownfish. Much clas- sical literature has focused on developing new feature extractors, often based on image processing and filtering methods. While specific methods are out of this class’s scope, we provide several references for further reading at the end of this document. The primary benefit of classical feature extraction methods is that they result in interpretable 표현s: because we designed these methods by hand, 우리는 할 수 있습니다 easily explain why specific output represen- tations were chosen. This 될 수 있습니다 especially important for downstream tasks with high stakes, perhaps in legal or medical domains. However, the downside is that coming up with these feature extractors is a tedious process, requiring significant amounts of time and domain expertise that may not be available.


## 2.3. Modern CV and Learned Representations

Modern 컴퓨터 비전 methods (≈ 2012-present) replace manually extracted features with learned intermediate 표현s as visualized in Fig- ure 8. One of the most common model architecture for this purpose are Con- volutional Neural Networks (CNN) consisting of layers of convolutional filters applied to images. These filters are learned typically using labeled training data provided a learned intermediate 표현 of the input data. The learned 표현s turn out to be much more powerful than the previously hand-designedfeaturesinprovidingtheessentialinformationtothesubsequent classifiers. While learned 표현s have shown higher accuracy in downstream taskssuchasforexampleimageclassification,thedownsideisthattheylackthe interpretability of classical 표현s. Several methods aim to interpret learned 표현s to understand why they perform so well and how to improve them. Zeiler and Fergus [7] analyzed the specific image patches that activate filters the most strongly for each layer in a CNN trained on ImageNet. What they discovered was that the learned 표현s shown in Figure 9 resembled the traditionally extracted features (such as edges, textures, body parts shown in Figure 7). Another approach to understanding learned intermediate 표현s of the input data is projecting them to a lower dimensional space so that 우리는 할 수 있습니다 plot and interpret them. One popular technique for achieving 이것은 tSNE [5], which performs dimensionality reduction on high-dimensional intermediate 표현s by minimizing the KL-divergence of joint probabilities (based on data similarity) between the low-dimensional embeddings and original high- dimensional 표현s. We expect that data points from the same class will be geographically clustered near each other as visualized in Figure 10. We canusetSNEtosanitycheckthatourneuralnetworkindeedlearnsthatimages 7 Figure 8: Learned intermediate 표현s of input data [2]. Components of the intermediate feature 표현s are not manually designed, hence the blank boxes. Figure9: Differentlevelsoflearnedintermediate표현srepresentthose of a traditional pipeline from Figure 7. 8 Figure 10: Visualization of tSNE embeddings on the MNIST dataset [2]....


## 2.4. Unsupervised and Self-Supervised Learning

In a traditional supervised 학습 formulation, 우리는 할 수 있습니다 train a model for a specific inference task such that it minimizes a loss function (e.g. image clas- sification accuracy or pose 추정) on a training dataset of D datapoints {(x ,y )} , where x is the ith datapoint in a given input 표현 (e.g. i i D i images or 3D 점 clouds) and y is the label in the output 표현 i (e.g. object categories or 6D poses). In practice, data is abundant. However, labels are expensive to acquire and can require specialized knowledge. Can we learn meaningful 표현s from data without labels? Consider the au- toencoder architecture in Figure 11. The goal of an autoencoder is to learn to perfectly reconstruct the input image. How well such an encoder is able to achieve 이것은 measured by the reconstruction loss, which is defined as the difference between the input data X and the output F(X), both in the same 표현. Here, the intermediate 표현 z of the input data is a lower-dimensional vector, and is located at the middle of the network in Fig- ure 11. Intuitively, we expect that if the second half of the network F is able 9 Figure 11: Autoencoder architecture for unsupervised 학습 [2]. F is the autoenconder, X is the input image, Xˆ is the reconstructed input image. to reconstruct the input from z, then z is a useful and informative compressed version of the input data. For this reason, we say that z is the bottleneck. We say that an autoencoder performs unsupervised 학습 since it re- quires no external labels aside from the input image (which may be viewed as the label itself)....


## 요약

이 강의에서는 표현 학습의 주요 개념과 방법론을 다뤘습니다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [07-representation-learning.pdf](https://web.stanford.edu/class/cs231a/course_notes/07-representation-learning.pdf)
