---
title: "[CS231A] Lecture 05: Active and Volumetric Stereo (능동 및 볼륨 스테레오)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Active Stereo, Volumetric Stereo, Structured Light]
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

이 포스트는 Stanford CS231A 강의의 다섯 번째 강의 노트인 "Active and Volumetric Stereo"를 정리한 것입니다.

**원본 강의 노트**: [05-active-volumetric-stereo.pdf](https://web.stanford.edu/class/cs231a/course_notes/05-active-volumetric-stereo.pdf)

<!--more-->
아래는 **CS231A Course Notes 5 – Active and Volumetric Stereo**를
처음 봐도 이해할 수 있도록 스토리 + 직관 + 핵심 개념 + 실습 코드까지 정리한 내용이다 

---

## 1. 이 강의가 다루는 큰 그림이다

지금까지 배운 건 “두 카메라가 같은 장면을 찍었을 때, **픽셀 대응을 맞춰서 깊이를 구한다**”는 전통적인 스테레오 구조였다

* 두 이미지의 점 $(p, p')$를 대응시킨 뒤
* 에피폴라 기하 + 삼각측량으로 3D 점 $P$를 구하는 구조였다

여기서 항상 등장하는 핵심 난제가 하나 있었다

> **대응점(correspondence)을 어떻게 정확히 찾을 것인가**라는 문제였다

텍스처가 없거나, 반복 패턴이 많거나, 조명이 안 좋으면
“여기가 진짜 매칭인지” 헷갈리는 경우가 너무 많다

Lecture 5는 이 문제를 완전히 다른 방향에서 접근한다

1. **Active Stereo**

   * 카메라가 환경에 직접 빛 패턴을 쏴서 **의도적으로 텍스처를 만들어 주는 방법**이다
   * Kinect, 구조광(Structured Light) 깊이 센서 같은 게 여기에 속한다

2. **Volumetric Stereo**

   * “점마다 대응을 찾는 대신, **공간 전체를 하나의 부피(volume)**로 보고
     그 안에 어떤 점이 실제 물체인지 검사하는 방식”이다
   * **Space Carving / Shadow Carving / Voxel Coloring** 세 가지 기법이 등장한다

기본 스테레오가 “픽셀→3D”를 고민한다면,
Volumetric Stereo는 “3D 후보→여러 이미지에서 일관적인가”를 검사하는 방식이라고 보면 된다

---

## 2. Active Stereo 이야기다

### 2.1 “카메라 둘 → 카메라 + 프로젝터”로 바꾸는 발상이다

전통적 스테레오에서는 카메라 두 대가 장면을 찍는다
Active Stereo에서는 이 구조를 바꾼다

* 왼쪽 카메라 자리에 **프로젝터**를 둔다
* 오른쪽에는 **카메라 한 대**만 남긴다

프로젝터는 "화면"에 해당하는 **가상 이미지 평면(virtual image plane)**에서
점 $p$를 선택해 장면으로 쏜다
이 빛이 물체 표면의 어떤 3D 위치 $P$에 닿고
그 점 $P$가 카메라 이미지에서 $p'$로 보인다 

여기서 중요한 포인트는 다음과 같다

* 우리가 **뭘 쐈는지 정확히 알고 있다**는 점이다

  * 가상 평면에서의 좌표 $p$
  * 색, 밝기, 패턴의 모양 등
* 그러면 카메라 이미지에서 **해당 패턴을 찾는 게 훨씬 쉽게 된다**
* 즉, correspondence 문제를 거의 “디자인”으로 해결하는 구조가 된다

이 프로젝터–카메라 쌍은
실질적으로 두 카메라의 에피폴라 기하와 완전히 같은 구조를 갖는다
다만 **왼쪽 카메라 이미지 평면 대신 “프로젝터 가상 평면”이 쓰일 뿐**이다

---

### 2.2 점 하나 대신 “선 한 줄”을 쏘는 이유이다

점 하나씩 쏘면 정확하지만, 한 점만 얻어서 물체 전체를 재구성하려면 너무 오래 걸린다다

그래서 보통은 **세로 줄 하나(스트라이프, line)** 를 쏜다다 

* 프로젝터 가상 평면에서 세로줄 $s$를 쏜다
* 이 줄이 3D 장면에서 어떤 곡선/띠 $S$로 투영된다
* 카메라 이미지에서는 이 띠가 다시 세로 혹은 곡선 줄 $s'$로 보인다

카메라–프로젝터가 **평행하거나 rectified** 되어 있으면

* 카메라 이미지에서 수평 epipolar line 들이 정의되고
* 각 줄 $s'$와 epipolar line의 교차로 **점별 대응**을 쉽게 찾을 수 있다
* 그 점들에 대해 앞에서 배운 삼각측량(triangulation)으로
  3D 표면 위의 점들을 하나씩 복원해 나간다

이 세로줄을 조금씩 옆으로 “쓸어가며(swipe)”
계속해서 동일한 작업을 반복하면
결국 전체 물체 표면을 매우 정확하게 스캔할 수 있다

고급 3D 스캐너, 레이저 스캐너들이 이런 방식을 변형해 사용한다

---

### 2.3 프로젝터–카메라 캘리브레이션이다

Active Stereo가 제대로 동작하려면

* **카메라 K, R, T**
* **프로젝터의 “내부/외부 파라미터”**

를 모두 알아야 한다다

방법은 다음과 같다 

1. 먼저 일반 카메라처럼 **캘리브레이션 패턴(체스보드 등)** 으로 카메라를 캘리브레이션한다
2. 이제 프로젝터가 이 패턴에 여러 스트라이프를 쏜다
3. 카메라 이미지에서 스트라이프가 패턴 어디에 떨어졌는지 측정한다
4. 그 대응으로부터 “프로젝터 가상 평면 ↔ 패턴 3D 위치”의 관계를 추정한다
   → 결국 프로젝터의 K, R, T도 일반 카메라와 동일한 수식으로 풀 수 있다

이렇게 세팅된 Active Stereo는 **매우 높은 정밀도의 3D 스캔**을 제공할 수 있다
예를 들어 Stanford의 Marc Levoy 그룹은 레이저 스캐너로 미켈란젤로 조각을 **sub-mm 단위**로 복원했다는 예시가 나온다다 

---

### 2.4 값싼 대안: 그림자(Shadow) 기반 Active Stereo이다

고급 프로젝터·레이저는 비싸고 세팅도 복잡하다
그래서 더 저렴한 대안이 소개된다다 

* 물체와 조명 사이에 **막대기(stick)** 를 둔다
* 막대기가 만드는 **그림자(stripe)** 가 곧 프로젝터의 줄과 같은 역할을 한다
* 막대기를 움직이면 그림자 줄도 움직이므로
  Active Stripe 스캐닝과 동일한 아이디어로 3D를 복원할 수 있다

값은 싸지만

* 막대기–카메라–조명 위치를 아주 정확히 알아야 하고
* 그림자 폭을 적당히 얇게 유지해야 하며
* 전체적으로 정밀도가 떨어진다는 단점이 있다

---

### 2.5 한 번에 많은 줄을 쏘는 Structured Light이다

linewidth를 하나씩 스캔하는 방식은 매우 느리다

* 물체 전체를 커버하려면
  셔터를 수십~수백 번 열어야 하고
* 중간에 물체가 움직이면 바로 깨진다

그래서 나온 방법이 **"한 장의 이미지에 색/패턴을 잔뜩 encode해서 한 방에 쏘는 구조광(Structured Light)"**이다 

* 예: 각 세로줄마다 서로 다른 색/코드 패턴을 줘서
  카메라 이미지에서 “이 픽셀이 어떤 줄에서 왔는지”를 **유일하게 식별**하게 한다
* 그림 3에서 보듯, 알록달록한 stripe 패턴 전체를 한 프레임으로 쏴서
  한 번에 물체 전체의 3D를 복원하는 구조다

이 아이디어가 바로 **초기 Kinect** 등 많은 깊이 센서의 핵심이다

* 실제로는 RGB 대신 **적외선 패턴**을 쓰기 때문에
  실내 조명이나 태양광에 관계없이 3D 정보를 뽑아낼 수 있다

---

## 3. Volumetric Stereo 이야기다

이제 완전히 다른 발상의 3D 복원법이 등장한다

### 3.1 방향을 뒤집는 발상이다

전통 스테레오/Active Stereo는

> “이미지에서 대응점을 찾고 → 삼각측량으로 3D 점을 구한다”

라는 **2D→3D 방향**이었다

Volumetric Stereo는 정반대 방향을 쓴다다 

1. 먼저, 우리가 복원하려는 물체가 들어 있을 **작업 부피(working volume)** 를 정한다

   * 예: 어떤 상자 안에 물체를 올려놓고, 주변에 카메라 여러 대를 둔다
2. 이 부피를 **작은 큐브(voxel)**로 촘촘히 나눈다
3. 각 voxel의 중심을 3D 가설 점 $P$라고 본다
4. 이 $P$를 여러 카메라로 투영해서

   * 각 이미지에서 어떤 픽셀에 떨어지는지 확인하고
   * 여러 기준과 “일관성(consistency)”을 검사한다
5. 여러 뷰에서 일관된 voxel만 남기고 나머지는 “깎아낸다(carving)”

결국 **처음에는 큰 상자 전체가 물체 후보였다가,
조건에 안 맞는 voxel을 하나씩 제거하면서
진짜 물체 모양만 남기는 방법**이라고 보면 된다

이때 “일관성”을 어떻게 정의하느냐에 따라

* **Space Carving**
* **Shadow Carving**
* **Voxel Coloring**

세 가지 주요 기법으로 나뉜다

이 방법은 장면 전체보다는

* “손에 들고 돌려가며 찍을 수 있는 물체”
* “작업 부피가 제한된 Product Scan”

같은 상황에서 많이 쓰인다

---

## 4. Space Carving – 실루엣으로 공간을 깎는 방법이다

### 4.1 실루엣(silhouette)과 Visual Cone 개념이다

그림 5를 보면, 한 카메라가 물체를 본다다 

* 물체의 외곽선(contour)을 따라가면
* 그 안쪽 픽셀들이 **실루엣(silhouette)** 이 된다
* 이 컨투어와 카메라 중심을 이으면
  **“광선이 갈 수 있는 모든 점의 집합”**이 생긴다
* 이 집합이 바로 **Visual Cone** 이다

직관적으로는

> 실루엣 안에 있는 모든 픽셀은
> 그 뒤에 “어딘가에” 물체 표면이 있다는 말이다

따라서 카메라 중심과 실루엣을 잇는 모든 광선이 만들어내는
원뿔/부채 모양의 3D 영역 안에는 **반드시 물체가 들어 있어야 한다**

### 4.2 여러 뷰의 Visual Cone을 교집합 내는 구조이다

카메라가 여러 대라면 어떻게 될까

* 각 카메라는 자기 실루엣에서 **자신만의 Visual Cone**을 갖는다
* 물체는 모든 카메라에서 동시에 보이므로
  **모든 Visual Cone의 교집합** 안에 있어야 한다

이 교집합이 바로 **Visual Hull**이다 

그림 6처럼 생각하면 된다

1. 작업 부피 전체를 voxel grid로 깐다
2. 각 voxel을 모든 카메라에 투영해 본다
3. 어떤 카메라에서든

   * 투영 위치가 실루엣 밖이면 “이 voxel은 물체가 아니다”
   * 그 voxel을 제거한다
4. 모든 카메라에 대해 반복하면

   * 결국 **모든 실루엣 내부에 공통으로 들어가는 voxel만 남는다**
   * 이게 바로 Visual Hull 근사 3D 모델이다

그림 7에서, 빨간 실제 물체를 둘러싼 격자 중
남은 격자들이 Visual Hull로 남는 모습을 볼 수 있다 

Visual Hull은 항상 **“보수적인(conservative)”** 결과를 준다

* 실제 물체보다 항상 같거나 더 크게 나온다
* 실루엣이 물체를 감싸는 껍데기 같은 느낌이라서
* 특히 “홈 파인 부분(concavity)”을 잘 잡지 못한다

### 4.3 장단점이다

장점이다

* 대응점 찾을 필요가 없다
* 실루엣만 있어도 된다
* 구현이 상대적으로 단순하다

단점이다

* voxel 해상도에 따라 계산량이 **voxel 수에 선형, 즉 공간 해상도에 세제곱**으로 늘어난다

  * voxel 크기를 절반으로 줄이면 voxel 개수는 8배가 된다
  * Octree 같은 자료구조로 조금 줄일 수는 있다
* 뷰 수가 적으면 Visual Hull이 매우 거칠어진다
* 실루엣 추출이 조금만 오염되어도 결과가 크게 망가질 수 있다
* 가장 큰 한계는 **일반적인 오목한(concave) 부분을 복원할 수 없다**는 점이다

  * 그림 8처럼 오목한 부분은 모든 실루엣에서 “겉으로만 보이기 때문에”
    그 안을 carve 하려면 겉도 깎여버리게 된다다 

---

## 5. Shadow Carving – Self-shadow로 concavity를 파고드는 방법이다

Space Carving은 실루엣 기반이라 concavity를 못 본다
그 오목한 부분을 잘 보고 싶으면 무엇을 써야 할까

강의는 **Self-shadow(자가 그림자)**를 사용한다 

### 5.1 Self-shadow란 무엇인가이다

* 물체가 빛을 받으면

  * 바닥이나 다른 물체 위에 생기는 그림자는 “투사 그림자”이고
  * 자기 자신 표면 위에 생기는 그림자가 **Self-shadow** 이다
* 오목한 부분은 외부 빛이 잘 들어오지 않기 때문에
  Self-shadow가 잘 생기는 영역이다

Self-shadow를 잘 측정할 수 있으면
“어디가 빛이 막혀 있는지 → 어디가 concavity인지”를 유추할 수 있다

### 5.2 Shadow Carving의 세팅이다

그림 9를 보면, 세팅은 이렇다 

* 중앙에 물체가 있고
* 그 앞에 **캘리브레이션된 카메라**가 있다
* 그리고 카메라 주변에 여러 개의 **점광원(light)** 이 원형으로 배치되어 있다
* 물체는 회전판(turntable)에 올라가 있을 수도 있다

### 5.3 알고리즘 개요다

1. 먼저 Space Carving처럼

   * 실루엣 기반으로 초기 voxel grid를 깎아 **대략적인 Visual Hull** 을 만든다
2. 이제 각 빛 하나씩 켜 보면서

   * 물체가 스스로 드리우는 shadow를 관찰한다
3. 한 light를 켠 상태에서

   * 카메라 이미지에서 물체 실루엣 중 “어두운 영역(self-shadow)"을 찾는다
   * 그 영역에서 나오는 광선(visual cone of shadow)과
   * 기존 Visual Hull 표면에 위치한 voxel들을 연결해 본다 
4. 어떤 voxel이

   * 빛과 카메라 둘 다에서 shadow cone 안에 있다면
   * 그 voxel은 **실제 물체가 아니고 오목한 공간 안의 점**일 가능성이 크다
   * 이런 voxel을 제거하면서 concavity를 표현할 수 있게 된다

결과적으로 Shadow Carving은

* Space Carving의 “겉껍데기”에서 출발해
* Self-shadow 정보를 이용해서
* 오목한 부분을 더 정확하게 carve 하는 방식이다

계산량은

* voxel 수에 비례하고
* 사용하는 light 개수가 (N)개라면 Space Carving보다 대략 (N+1)배 정도 느리다

장점이다

* concavity 복원이 훨씬 잘 된다

단점이다

* 여전히 conservative volume이다
* **조명 + 물체 반사 특성**에 민감하다

  * 반짝이는 재질, 매우 어두운 재질 등에서는
    그림자 검출이 어렵다

---

## 6. Voxel Coloring – 색 일관성으로 공간을 채우는 방법이다

마지막 기법은 이름 그대로 **색(color) 일관성**을 사용하는 방법이다 

### 6.1 기본 아이디어이다

Voxel Coloring의 설정은 다음과 같다

* 물체를 여러 카메라에서 찍은 이미지들이 있다
* 카메라들은 모두 캘리브레이션되어 있다
* 특정 working volume 안에 물체가 들어 있다고 가정한다

이제 각 voxel을 하나씩 본다

1. voxel 중심 $P$를 모든 카메라에 투영한다
2. 각 카메라 이미지에서 얻은 픽셀 색을 모은다
3. 이 색들이 서로 “충분히 비슷” 하면

   * 이 voxel은 **물체 표면에 있는 점**일 확률이 높다
   * voxel을 “살려 두고”, 그 voxel의 색을 이 평균 색으로 설정한다
4. 색이 너무 달라 일관성이 없으면

   * 이 voxel은 물체가 아닌 것으로 간주하고 제거한다

이 과정을 **모든 voxel에 대해 수행**하면

* Space Carving처럼 모양만 얻는 게 아니라
* **색까지 가진 컬러 3D 모델**을 얻게 된다

### 6.2 Lambertian 가정이다

Voxel Coloring이 잘 동작하려면
아주 중요한 가정을 하나 쓴다 

> 물체가 Lambertian이다

Lambertian이란

* 표면이 보는 방향에 따라 밝기/색이 변하지 않는 특성이다
* 어느 방향에서 보든 동일한 색·밝기를 가지는 이상적인 확산 반사체이다

만약 물체가

* 유광 금속처럼 반사해서 하이라이트가 생기거나
* 각도에 따라 색이 달라지는 재질이라면

같은 voxel을 서로 다른 카메라에서 봤을 때
색이 크게 달라져 버린다

그러면 color consistency check가 실패해서
실제 물체임에도 voxel이 제거될 수 있다

그래서 Voxel Coloring은

* “대충 matte한, diffuse한 물체”에 가장 잘 맞는다

### 6.3 기본 Voxel Coloring의 모호성 문제이다

단순한 Voxel Coloring은
색만 보고 voxel을 살릴지 말지만 판단한다

그런데 그림 12처럼, 서로 다른 voxel들이
여러 카메라에서 **같은 픽셀과 align 되면**
"모두 색이 맞는 것처럼" 보이는 모호한 경우가 생길 수 있다 

* 이 경우 “어떤 voxel이 진짜 표면이고, 어떤 voxel은 뒤에 있는가”를
  색만으로는 구분하기 어렵다

### 6.4 Visibility 제약과 가까운 voxel부터 처리하는 순서이다

이 모호성을 줄이기 위해 **visibility constraint**를 도입한다 

아이디어는 다음과 같다

1. voxel을 처리하는 순서를 바꾼다

   * **카메라에 더 가까운 voxel부터**
   * **멀리 있는 voxel 순으로** 처리한다
   * 즉, 카메라–voxel 거리에 따라 layer-by-layer로 순서화한다
2. 한 voxel을 color-consistent라고 보고 살려 두면

   * 이 voxel은 나중에 그 뒤쪽 voxel들을 **가릴 수 있는 occluder**가 된다
3. 어떤 voxel이 여러 카메라에서

   * “최소 두 개 이상의 카메라에서 직접 보이는가”를 체크한다
   * 만약 그렇지 않고, 항상 앞의 voxel에 가려진다면
     그 voxel은 실제 표면이 될 수 없으므로 제거한다

이렇게 하면 “표면에 가장 가까운 voxel만 남도록” 정리할 수 있고
모호성이 크게 줄어든다

결론적으로 Voxel Coloring은

* **모양 + 텍스처를 동시에** 추정할 수 있는 매우 매력적인 방법이고
* Lambertian 가정, 카메라 배치 제한(가시성 순서를 정의해야 함) 같은 제약이 있다

---

## 7. 간단한 Space Carving 실습 코드이다 (OpenCV + Open3D)

이제 Lecture 5의 개념 중 **Space Carving**을
직접 실험해 볼 수 있는 Python 코드를 하나 만들어 보겠다

가정이다

* 여러 각도에서 촬영한 **실루엣 마스크 이미지**를 가지고 있다

  * 각 이미지는 흰색(255)이 물체, 검은색(0)이 배경인 이진 이미지이다
  * 예: `sil_0.png`, `sil_1.png`, …
* 각 뷰에 대한 **projection matrix** $P_i = K_i [R_i | t_i]$를 알고 있다

아래 코드는

1. 일정 범위의 3D 공간을 voxel grid로 나누고
2. 각 voxel을 모든 카메라에 투영하여

   * 어떤 뷰에서라도 실루엣 바깥이면 제거하고
3. 남은 voxel들을 Open3D로 점군으로 시각화한다

```python
import numpy as np
import cv2
import open3d as o3d

# -----------------------------------------------
# 1. 입력 준비 부분이다
# -----------------------------------------------

# 실루엣 이미지 경로들이다 (흰색=물체, 검정=배경이라고 가정한다)
silhouette_paths = [
    "sil_0.png",
    "sil_1.png",
    "sil_2.png",
]

# 각 뷰의 3x4 projection matrix $P = K [R | t]$ 리스트이다
# 실제 값으로 채워야 한다
# 예시로 자리에 맞는 형태만 만들어 둔다
P_mats = []
for i in range(len(silhouette_paths)):
    # 여기에는 실제 카메라 캘리브레이션에서 얻은 3x4 P를 넣어야 한다
    P = np.array([
        [1000, 0, 320, 0],
        [0, 1000, 240, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    P_mats.append(P)

# 실루엣 이미지를 읽어서 0/1 마스크로 만든다
silhouettes = []
for path in silhouette_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"{path} 이미지를 찾을 수 없다")
    # threshold로 0 또는 1 값으로 만든다
    _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    silhouettes.append(mask)
    print(f"{path}: shape={mask.shape}")

H, W = silhouettes[0].shape

# -----------------------------------------------
# 2. voxel grid 정의이다
# -----------------------------------------------

# 작업 부피 범위 (단위: 예를 들어 미터)이다
# 실제 세팅에 맞게 조정해야 한다
xmin, xmax = -0.5, 0.5
ymin, ymax = -0.5, 0.5
zmin, zmax = 0.2, 1.0  # 카메라 앞쪽만 본다는 가정이다

nx, ny, nz = 50, 50, 50  # 해상도이다 (너무 크면 매우 느려진다)

xs = np.linspace(xmin, xmax, nx)
ys = np.linspace(ymin, ymax, ny)
zs = np.linspace(zmin, zmax, nz)

# -----------------------------------------------
# 3. Space Carving 메인 루프이다
# -----------------------------------------------

kept_points = []

for ix, x in enumerate(xs):
    print(f"{ix+1}/{nx} x-슬라이스 처리 중이다")
    for iy, y in enumerate(ys):
        for iz, z in enumerate(zs):
            P_world = np.array([x, y, z, 1.0])

            inside_all = True
            for P, sil in zip(P_mats, silhouettes):
                proj = P @ P_world
                if proj[2] <= 0:
                    inside_all = False
                    break

                u = proj[0] / proj[2]
                v = proj[1] / proj[2]
                u_int = int(round(u))
                v_int = int(round(v))

                # 이미지 범위 밖이면 실루엣 밖으로 간주한다
                if not (0 <= u_int < W and 0 <= v_int < H):
                    inside_all = False
                    break

                # 실루엣 값이 0이면 물체가 아니므로 해당 voxel 제거한다
                if sil[v_int, u_int] == 0:
                    inside_all = False
                    break

            if inside_all:
                kept_points.append([x, y, z])

kept_points = np.array(kept_points, dtype=np.float64)
print("남은 voxel 개수:", kept_points.shape[0])

# -----------------------------------------------
# 4. Open3D로 결과 시각화이다
# -----------------------------------------------

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(kept_points)

# 색상을 임의로 파란색으로 설정한다
colors = np.zeros_like(kept_points)
colors[:, 2] = 1.0
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])
```

이 코드는

* Lecture 5에서 설명한 **Space Carving의 voxel 기반 구현**을 그대로 따라간다
* 해상도(nx,ny,nz)를 줄이면 금방 끝나고, 키우면 매우 느려지는 걸 직접 느낄 수 있다
* 실제로는

  * voxel grid를 옥트리(octree)로 바꾸거나
  * GPU를 사용하거나
  * working volume을 더 작게 잡는 식으로 최적화한다

---

## 8. 간단한 Structured Light 패턴 생성 코드 예시이다 (Active Stereo용)

Active Stereo 실험을 해보고 싶다면
우선 프로젝터로 쏠 패턴 이미지를 만들어야 한다

아래 코드는

* 다양한 색의 세로 stripe를 가진 structured-light 패턴 이미지를 만들어
* `pattern.png`로 저장한다

```python
import numpy as np
import cv2

W, H = 1920, 1080  # 프로젝터 해상도에 맞게 설정해야 한다
num_stripes = 32

pattern = np.zeros((H, W, 3), dtype=np.uint8)

stripe_width = W // num_stripes

for i in range(num_stripes):
    x0 = i * stripe_width
    x1 = (i + 1) * stripe_width if i < num_stripes - 1 else W

    # 각 stripe마다 서로 다른 색을 부여한다
    hue = int(180.0 * i / num_stripes)
    color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]

    pattern[:, x0:x1, :] = color_bgr

cv2.imwrite("pattern.png", pattern)
print("pattern.png 저장 완료다")
```

* 이 이미지를 프로젝터로 쏘고
* 카메라로 촬영한 뒤
* 각 픽셀이 어느 stripe에서 왔는지 색으로 디코딩하면
* **Active Structured Light** 방식의 깊이 추출을 시작해 볼 수 있다

실제 깊이 계산은 강의보다 훨씬 많은 구현 디테일이 필요하지만,
Lecture 5에서 이야기한 아이디어를 실험해 볼 수 있는 좋은 출발점이 된다

---

여기까지가 Lecture 5의 전체 스토리와 핵심 개념, 그리고 실제로 손으로 만져볼 수 있는 코드 예시까지 정리한 내용이다

* Active Stereo는 **문제를 쉽게 만들기 위해 빛을 설계해서 쏘는 방법**이다
* Volumetric Stereo는 **공간 전체를 voxel로 쪼개서 일관성으로 깎아나가는 방법**이다
* Space/Shadow Carving, Voxel Coloring이 각각
  실루엣, self-shadow, 색 일관성을 사용해
  서로 다른 장단점을 가진 3D 복원법을 제공한다

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [05-active-volumetric-stereo.pdf](https://web.stanford.edu/class/cs231a/course_notes/05-active-volumetric-stereo.pdf)
