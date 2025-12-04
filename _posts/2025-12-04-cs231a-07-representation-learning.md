---
title: "[CS231A] Lecture 07: Representation Learning (표현 학습)"
categories: [3D Geometry]
tags: [3D Vision, CS231A, Representation Learning, Deep Learning]
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

이 포스트는 Stanford CS231A 강의의 일곱 번째 강의 노트인 "Representation Learning"를 정리한 것입니다.

**원본 강의 노트**: [07-representation-learning.pdf](https://web.stanford.edu/class/cs231a/course_notes/07-representation-learning.pdf)

<!--more-->

이 강의는 **"Representation(표현)"이란 도대체 뭔지, 그리고 그걸 어떻게 잘 배우게 할 건지**에 대한 이야기다

지금까지는 카메라 기하, 에피폴라, SfM처럼 "세상은 이렇게 돌아간다" 쪽을 봤다면
이 강의부터는 "이미지 → 벡터로 바꿔서 뇌(=모델)가 이해하기 좋게 만드는 법" 쪽으로 넘어가는 거다.

---

## 1. 상태(state)와 표현(representation) 이야기다

### 1.1 상태란 뭔가이다

PDF 첫 부분에 2D 평면 위를 걷는 로봇 그림이 있다

* 이 시스템의 **상태(state)**는 지금 로봇을 완전히 설명하는 데 필요한 정보 묶음이다

  * 시간에 따라 변하는 것: 몸 위치, 방향, 관절 각도, 각속도 등
  * 고정된 것: 다리 길이, 각 관절의 질량, 마찰 계수 등

이걸 어떤 좌표계·표현으로 적을지는 **내가 풀고 싶은 문제**에 따라 달라진다

* 회전은 Euler angle로 써도 되고, quaternion, angle-axis로 써도 된다
* 위치는 $(x, y, z)$ 카테시안이든, 거리+각도 polar든 가능하다

즉, **상태 = 정보 자체**,
**표현 = 그 정보를 어떻게 표현해서 쓸지에 대한 선택**이다.

그리고 상태는 시간에 따라 바뀐다.
시간 $t$의 상태를 $x_t$라고 쓴다.

---

### 1.2 Markov chain: “현재가 전부다” 가정이다

많은 동적 시스템에서 **Markov property**를 가정한다

$$P(x_{t+1} \mid x_t, x_{t-1}, \dots, x_1) = P(x_{t+1} \mid x_t)$$

즉,

> 미래는 과거 전체를 볼 필요 없고,
> "지금 상태"만 알면 된다

라는 가정이다.

이때 상태 전이 함수 $f$를 써서

$$x_{t+1} = f(x_t)$$

형태로 보기도 한다.
로봇 관절 각속도가 state 안에 들어있으면,
그걸 이용해 다음 스텝의 관절 각도를 예측할 수 있다는 느낌이다.

이렇게 **상태가 다 보이는 완전 관측** 상황을 Markov chain으로 모델링한다
페이지 2의 그래픽 모델 그림이 바로 이 구조다 

---

### 1.3 HMM: 진짜 상태는 숨겨져 있고, 센서는 noisy하다

현실은 state를 직접 볼 수 없다.
우리는 센서에서 나오는 **관측 값 $z_t$**만 본다.

예:

* 자율주행 차 입장에서 **진짜 state**

  * 주변 차/사람/도로의 3D 위치, 속도, 크기, 방향 등
* 우리가 실제로 받는 관측

  * RGB 이미지, depth, LiDAR 포인트, 레이더 등

그래서 **Hidden Markov Model(HMM)**을 쓴다

가정은 두 개다:

1. 상태는 여전히 Markov
$$P(x_{t+1} | x_t, \dots) = P(x_{t+1} | x_t)$$
2. 관측은 현재 상태에만 의존
$$P(z_t | x_t, x_{t-1}, \dots, z_{t-1}, \dots) = P(z_t | x_t)$$

그래서

* **상태 전이 모델**: $f(x_t) = x_{t+1}$
* **관측 모델**: $h(x_t) = z_t$

이라는 두 개 함수를 가지는 구조가 된다
Figure 3의 PGM이 바로 이 HMM 구조다

자율주행 예시(페이지 3의 NuScenes 그림)에서는 

* $x_t$: 각 객체의 3D 위치, 크기, 속도
* $z_t$: 센서에서 본 RGB 또는 LiDAR
* 우리가 하고 싶은 일: $z_{1:t}$를 보고 $x_t$를 최대한 잘 추정하는 것 (=state estimation)

뒤 강의에서 나오는 Kalman Filter, Particle Filter 등이 이걸 푸는 알고리즘이다.

---

### 1.4 Generative vs Discriminative 접근이다

"이미지 $z$에서 object pose $x$를 추정하자"라고 했을 때,
크게 두 방향이 있다

#### (1) Generative

* 전체 joint $p(z, x)$ (또는 likelihood $p(z|x)$, prior $p(x)$)를 모델링한다
* 예: pose $x$를 하나 샘플하고 → 그걸로 **이미지를 렌더링**해서 $z$와 비교하는 방식

Pose estimation 예시에서:

1. prior $p(x)$에서 pose sample $x^{(i)}$를 뽑는다
2. observation model $h(x^{(i)})$로 해당 pose일 때 이미지를 렌더 $z^{(i)}$한다
3. 실제 관측 $z$와 비교해 $p(z | x^{(i)})$를 계산한다

이게 generative한 사고방식이다.

#### (2) Discriminative

* 관측 → 상태로 바로 점프하는 $p(x|z)$ 또는 직접 mapping $x = f_\theta(z)$을 학습한다
* PoseCNN 같은 네트워크가 그 예다 (Figure 5) 

즉,

> Generative: “이 pose면 이런 이미지가 나올 텐데?”
> Discriminative: “이미지 봐서 바로 pose 찍자”

둘 다 장단점이 있고, 실제론 많이 섞어서 쓴다.

---

## 2. Representation이란 사실 어디에나 끼어 있는 개념이다

### 2.1 입력 / 중간 / 출력 표현이다

Figure 6을 보면 전체 vision 파이프라인이 세 단계로 나뉜다

1. **Input representation**

   * 원시 센서 데이터 포맷
   * RGB 이미지, depth, point cloud, stereo pair, LiDAR 등
   * RGB냐 HSV냐, 컬러냐 흑백이냐도 다 input representation 선택이다

2. **Intermediate representation**

   * 고차원 입력을 **낮은 차원 벡터**로 압축한 것
   * 이미지 → 512-d feature vector 같은 것
   * semantic segmentation 예시에서는 $X$ → 몇 개 숫자(피처) → 최종 label로 이어진다

3. **Output representation**

   * 최종적으로 decision에 필요한 high-level 요약이다
   * 예:

     * "Fish / Coral / Background" 같은 label
     * 물고기 species
     * 3D bounding box, 6D pose

중간 표현은 “입력을 압축해서, 마지막 task를 쉽게 만들어 주는 정보 묶음”이라고 보면 된다.

---

### 2.2 좋은 Representation의 조건이다 (Bengio et al.)

논문 [Bengio et al., 2013] 기준으로 좋은 representation은 다음 성질을 갖는다 

1. **Compact(압축)**

   * 불필요한 중복 없이 최소한의 차원으로 정보를 담되,
   * 중요한 건 잃지 않아야 한다

2. **Explanatory(설명력)**

   * 다양한 입력을 “설명”할 수 있을 만큼 expressive해야 한다

3. **Disentangled(요인 분리)**

   * 조명, 포즈, 아이덴티티, 배경 같은 **변화 요인들이 서로 분리되어 표현**되면 좋다
   * 예: 한 차원은 회전, 한 차원은 밝기… 이런 식

4. **Hierarchical(계층적)**

   * 낮은 수준 개념(에지, 코너)으로부터 높은 수준 개념(객체, 파트)을 쌓아올릴 수 있어야 한다
   * 재사용이 쉬워져서 계산 효율이 좋아진다

5. **Downstream task를 쉽게 만든다**

   * 진짜 평가는 “이 representation 위에 얹은 classifier/ planner가 얼마나 잘 작동하냐”이다
   * 결국 downstream performance로 평가할 수 있다

---

## 3. 전통적인 CV: 사람이 만든 feature로 표현을 설계하던 시대다

Figure 7이 pre-deep-learning 시절 파이프라인이다

* Input image →
* Edge / Texture / Color 같은 hand-crafted feature extractor →
* segment / parts →
* classifier → label(예: clown fish)

핵심은

> **중간 표현을 우리가 직접 설계한다**는 점이다

예:

* SIFT, HOG, Harris corner 같은 로컬 descriptor
* Gabor filter, color histogram, edge detector 등

장점이다

* **Interpretable** 하다

  * 왜 이런 결과가 나왔는지 설명하기 쉽다
  * “이 이미지는 오렌지색+흰색 줄무늬가 많아서 clownfish로 판단했다” 같은 스토리가 가능하다
  * 의료·법률 같이 설명 책임이 중요한 도메인에서 유리하다

단점이다

* feature 설계가 사람 손으로 하나하나 해야 해서 **노가다 + 도메인 지식**이 많이 필요하다
* 복잡한 문제에서는 이런 핸드메이드 feature로는 한계가 빨리 온다

---

## 4. 현대 CV: 중간 표현까지 전부 모델이 배우는 시대다

Figure 8이 딱 이 구조다

* Input image → CNN → 중간 feature (필터 출력들) → classifier → label

여기서

* filter bank(합성곱 커널) 자체를 **gradient descent로 학습**한다
* 즉, 중간 표현(필터 출력)은 사람이 설계하는 게 아니라
  모델이 **task loss를 최소화하는 방향으로 자동으로 찾아낸 표현**이다

이게 왜 강력하냐면

* 사람이 상상도 못한 복잡한 패턴 조합을 학습할 수 있고
* 데이터만 많다면, 상당히 general하고 transferable한 feature가 나온다

### 4.1 Learned representation은 실제로 뭘 배우는가이다

Zeiler & Fergus의 유명한 visualization이 Figure 9에 나와 있다

* 각 레이어의 필터를 "가장 세게 활성화시키는 이미지 patch"들을 모아보면

  * 첫 레이어: edge, 색깔 blob
  * 중간 레이어: 텍스처, 패턴
  * 마지막 레이어: 개의 얼굴, 물체 파트 같은 high-level concept

즉,

> CNN이 스스로 배운 representation이
> 우리가 classical pipeline에서 힘들게 만들던 edge/texture/part랑 굉장히 닮아 있다

라는 걸 보여준다.

### 4.2 t-SNE로 representation을 2D로 들여다보기다

중간 레이어 feature는 보통 128~2048차원이라 눈으로 보기 어렵다.
그래서 **t-SNE** 같은 차원 축소 알고리즘으로 2D, 3D로 줄여서 시각화한다

Figure 10은 MNIST에 t-SNE를 적용한 예시다

* 각 점: 한 이미지의 중간 표현 벡터
* 색: digit class (0~9)

잘 학습된 representation이라면

> 같은 class끼리는 2D 상에서 한 덩어리로 잘 모이고,
> 다른 class와는 떨어져 있어야 한다

t-SNE plot이 깨끗하면
“아, 이 네트워크가 진짜로 의미 있는 feature space를 만들었구나” 정도를 sanity check 할 수 있다.

---

## 5. Label 없이 representation을 배우는 방법이다

마지막 부분은 **Unsupervised / Self-supervised representation learning** 이야기다다 

### 5.1 Autoencoder: 입력을 스스로 복원하는 자기지도 학습이다

Figure 11의 구조가 autoencoder다다 

* Encoder: 이미지 X → latent vector z (low-dim)
* Decoder $(F)$: $z$ → 재구성된 이미지 $\hat{X} = F(X)$
* Loss: reconstruction loss
$$\mathbb{E}_X[|F(X) - X|]$$

**중간의 z가 bottleneck**이다

* 차원을 강제로 줄여 놓고
* 그 상태에서 다시 이미지를 복원하라고 했는데
* 그게 잘 된다면
  → z에는 입력 이미지의 “핵심 정보”가 담겨 있다고 보는 거다

이때 label y는 필요 없다

* 입력 X 자체가 target이기 때문에
* 순수 unsupervised로 representation을 배운다

### 5.2 Self-Supervised: 입력 일부를 가리고 그걸 맞추게 하는 트릭이다

Figure 12는 이미지를 대각선 기준으로 아래/위로 나누는 예다

* 아래 삼각형 = 입력 $X_{in}$
* 위 삼각형 = target $X_{out}$

모델에게

> "아래만 보고 위를 복원해라"

라고 시키면,
모델은 **이미지의 구조, object, context에 대한 representation**을 배우지 않으면 이 task를 잘 못 풀게 된다.

이런 식으로

* patch 예측
* rotation angle 맞추기
* 색 복원(colorization)
* 미래 frame 예측

등등 수많은 self-supervised task가 representation learning에 쓰인다.

### 5.3 Pretrain → Fine-tuning 패턴이다

실전에서는 보통 이렇게 쓴다

1. **라벨 없는 대규모 데이터**로

   * autoencoder나 self-supervised task를 학습해서
   * encoder를 pretrain 한다 (좋은 $z$ 공간을 만든다)

2. 그다음 **적은 라벨 데이터**만 가지고

   * encoder 위에 작은 classifier head를 붙여서
   * label task를 fine-tuning 한다

이러면

* scratch에서 supervised만으로 학습할 때보다
* 훨씬 적은 라벨로도 좋은 성능을 낼 수 있다

---

## 6. 실습용 코드 예시다 (PyTorch + t-SNE + 약간의 OpenCV)

Lecture 7 내용을 직접 손으로 만져볼 수 있는 예제로
**(1) 간단 CNN Autoencoder** + **(2) t-SNE로 latent 시각화**를 보여주겠다.

여기서는 PyTorch와 sklearn을 사용한다.

```bash
pip install torch torchvision matplotlib scikit-learn opencv-python
```

---

### 6.1 간단 CNN Autoencoder로 representation 학습하기다

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 데이터셋: MNIST (28x28, 흑백)
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2. 간단 CNN Autoencoder 정의
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(True),
            nn.Flatten(),                              # 32*7*7 = 1568
            nn.Linear(1568, latent_dim),
        )

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 1568)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 7->14
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid(),  # 0~1
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_fc(z)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(latent_dim=32).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. 학습 루프 (재구성 loss 최소화 = unsupervised)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, _ in train_loader:  # 라벨 안 씀
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon = model(imgs)
        loss = criterion(recon, imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
```

이 코드는 Lecture 7의 **autoencoder 구조**를 그대로 구현한 거다:

* encoder가 중간 representation $z$를 만든다
* decoder가 $z$로부터 입력 이미지를 복원한다
* loss는 $|X-\hat{X}|$이다

---

### 6.2 t-SNE로 latent representation 시각화하기다

학습이 끝난 뒤, test set에 대해 encoder가 만든 **latent 벡터 z**를 뽑아서
t-SNE로 2D에 뿌려 보자.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

model.eval()
latents = []
labels = []

with torch.no_grad():
    for imgs, ys in test_loader:
        imgs = imgs.to(device)
        z = model.encode(imgs)  # (B, latent_dim)
        latents.append(z.cpu().numpy())
        labels.append(ys.numpy())

latents = np.concatenate(latents, axis=0)
labels  = np.concatenate(labels, axis=0)

print("Latent shape:", latents.shape)

# t-SNE로 2D 임베딩
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='random', random_state=0)
embeds = tsne.fit_transform(latents)

# Plot
plt.figure(figsize=(8, 6))
num_classes = 10
colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

for c in range(num_classes):
    idx = labels == c
    plt.scatter(embeds[idx, 0], embeds[idx, 1], s=5, color=colors[c], label=str(c), alpha=0.7)

plt.legend()
plt.title("t-SNE of autoencoder latent space (MNIST)")
plt.tight_layout()
plt.show()
```

이 플롯이 Figure 10(MNIST t-SNE)와 같은 역할을 한다

* latent representation이 잘 학습되었다면

  * 숫자 0, 1, 2, … 각각이 서로 다른 군집으로 뭉쳐서 나타나는 걸 볼 수 있다
* label을 전혀 사용하지 않았는데도

  * autoencoder가 "형태가 비슷한 이미지끼리 비슷한 $z$로 보내는" 표현을 배웠다는 뜻이다

여기서 encoder를 고정하고
그 위에 작은 linear classifier만 얹어서 supervised fine-tuning을 하면,
적은 라벨로도 꽤 좋은 분류 모습을 볼 수 있다
→ Lecture 7 마지막에 설명한 unsupervised pretraining + supervised head 구조를 직접 확인하는 셈이다.

---

이렇게 Lecture 7에서는

* 동적 시스템의 상태/표현, Markov chain/HMM
* generative vs discriminative
* classical hand-crafted representation와 modern learned representation
* t-SNE로 representation 들여다보기
* label 없이 representation을 학습하는 autoencoder / self-supervised

까지 representation learning의 기본 철학과 도구들을 정리했다.

## 참고 자료

- [Stanford CS231A Course Notes](https://web.stanford.edu/class/cs231a/course_notes.html)
- [07-representation-learning.pdf](https://web.stanford.edu/class/cs231a/course_notes/07-representation-learning.pdf)
