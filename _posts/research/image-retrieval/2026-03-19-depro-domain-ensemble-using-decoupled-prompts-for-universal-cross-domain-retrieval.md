---
title: "[PaperReview] DePro: Domain Ensemble using Decoupled Prompts for Universal Cross-Domain Retrieval"
categories: [Image Retrieval]
tags: [Image Retrieval, UCDR, Prompt Learning, CLIP, DePro]
article_header:
  type: overlay
  theme: dark
  background_color: '#1b263b'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(27, 38, 59, .86), rgba(105, 89, 122, .60))'
    src: /assets/images/papers/depro-review/figure2_depro_overview.png
mathjax: true
mathjax_autoNumber: true
---

**Paper:** SIGIR 2025, *DePro: Domain Ensemble using Decoupled Prompts for Universal Cross-Domain Retrieval*  
**PDF:** [paper link](https://palm.seu.edu.cn/hxue/publications/DePro%20Domain%20Ensemble%20using%20Decoupled%20Prompts%20for%20Universal%20Cross-Domain%20Retrieval.pdf)

<!--more-->

## Intro

Universal Cross-Domain Retrieval(UCDR)은 학습할 때 본 도메인과 클래스만 잘 맞추는 문제가 아니다. 테스트에서는 **처음 보는 도메인**, **처음 보는 클래스**, 혹은 둘 다 동시에 등장할 수 있고, 그 상황에서도 query와 같은 의미의 이미지를 gallery에서 찾아야 한다. 그래서 이 문제는 단순 retrieval이 아니라, 사실상 **도메인 일반화(domain generalization)** 와 **카테고리 일반화(category generalization)** 를 동시에 요구한다.

논문은 이 설정을 세 가지 practical scenario로 설명한다.

- **U\_DCDR**: unseen domain, seen class
- **U\_CCDR**: seen domain, unseen class
- **UCDR**: unseen domain, unseen class

즉 모델은 "보지 못한 분포 변화"와 "보지 못한 의미 변화"를 동시에 감당해야 한다.

![UCDR scenarios](/assets/images/papers/depro-review/figure1_ucdr_scenarios.png)

기존 CLIP 기반 UCDR 연구들, 특히 ProS나 UCDR-Adapter 계열은 prompt tuning을 통해 일반화를 끌어올리려 했다. 하지만 DePro는 여기서 한 가지 중요한 문제를 지적한다. **domain prompt와 class prompt를 나눠 쓴다고 해도, 실제 학습 결과에서는 class prompt 안에 domain 정보가 섞여 들어갈 수 있다**는 점이다. 그렇게 되면

- domain prompt는 domain 역할,
- class prompt는 semantic 역할

을 맡아야 하는데, 실제로는 둘의 역할이 겹치거나 충돌한다. 특히 domain-specific prompt를 여러 개 두는 방식은 source domain에는 잘 맞지만 unseen domain에는 과적합될 위험이 있다.

DePro의 문제의식은 명확하다.

> 도메인 정보는 여러 source domain에 공통적인 방향으로 묶고, 클래스 정보는 domain 오염 없이 semantic discrimination에 집중시켜야 한다.

이를 위해 논문은 두 종류의 prompt를 설계한다.

- **Universal Domain Prompts (UDPs)**: 도메인 공통성을 학습하는 prompt
- **Class Prompts (CPs)**: 입력 이미지에 따라 적응적으로 생성되는 semantic prompt

그리고 여기에 decoupling loss, regulation loss, domain-aware triplet learning을 결합해 UCDR 문제를 푼다.

## Method

DePro를 이해할 때 가장 중요한 건, 이 논문이 UCDR을 다음 두 하위 목표로 분해한다는 점이다.

1. 서로 다른 domain의 이미지를 하나의 더 강건한 **universal domain** 으로 매핑하는 것  
2. 그 universal domain 안에서 클래스 간 구별력을 유지하는 것

이 두 목표를 각각 UDP와 CP가 담당한다.

![DePro overview](/assets/images/papers/depro-review/figure2_depro_overview.png)

### 1. Universal Domain Prompts

기존 방법은 종종 `real`, `sketch`, `painting` 같은 domain마다 별도 prompt를 둔다. 직관적으로는 자연스럽지만, 저자들은 이런 방식이 source domain에 과적합될 수 있다고 본다. 그래서 DePro는 domain-specific prompt 대신, 여러 도메인의 공통 특성을 담는 **universal domain prompt**를 사용한다.

텍스트 쪽에서는 class token과 함께 universal domain prompt를 넣는다. 논문이 의도하는 바는 명확하다. 텍스트 자체를 단순 class label이 아니라, **domain-invariant semantic anchor** 로 쓰겠다는 것이다.

텍스트 인코더 출력은 다음처럼 쓸 수 있다.

$$
T(y, P_0)=\mathrm{TextProj}(w^{y}_{L})
$$

여기서 \(P_0\) 는 learnable textual UDP이고, \(w^{y}_{L}\) 는 최종 transformer layer의 class-conditioned token이다.

이미지 쪽에서도 UDP를 visual prompt처럼 붙여, 입력이 어느 source domain에서 왔든 더 공통적인 domain space로 들어가도록 유도한다.

### 2. Adaptive Class Prompts

도메인 역할을 UDP가 맡는다면, 클래스 의미는 **Class Prompt(CP)** 가 맡는다. 하지만 DePro는 CP를 고정된 prompt table로 두지 않는다. 대신 **Adaptive Semantic-Prompter (ASP)** 라는 transformer 기반 모듈을 통해, 입력 이미지에 따라 적응적인 class prompt를 생성한다.

초기 meta class prompt \(\bar{P}_0\) 는 ASP를 거치며 점진적으로 업데이트된다.

$$
[\bar{P}_{j}, E^{I}_{j}] = F_j(\bar{P}_{j-1}, E^{I}_{j-1}), \quad j=1,\dots,\ell
$$

최종적으로 얻는 \(\bar{P}_{\ell}\) 이 adaptive class prompt다. 직관적으로는 같은 `dog` 클래스라도 `real dog`, `sketch dog`, `clipart dog`가 서로 다르게 보이기 때문에, 이미지 내용에 맞는 semantic prompt를 생성해 더 세밀한 class 정보를 뽑아내려는 것이다.

이미지-텍스트 정렬의 기본 목표는 결국 UDP와 CP가 들어간 이미지 feature를 텍스트 feature에 맞추는 것이다.

### 3. Prompt Decoupling

DePro의 핵심은 여기 있다. 저자들은 단순히 UDP와 CP를 따로 두는 것만으로는 충분하지 않다고 본다. CP가 학습 과정에서 domain 정보를 다시 품어버리면, 원래 의도한 역할 분리가 무너진다.

그래서 논문은 **decoupling loss** 를 도입한다. 아이디어는 단순하다. UDP를 제거한 상태에서도 CP만으로 class semantics를 잘 설명하도록 별도의 정렬 loss를 건다.

$$
g(y)=T(y), \qquad f(I)=V(I,\bar{P}_{\ell})
$$

$$
\mathcal{L}^{itm}_{dec}(I)
=-\sum_{y=1}^{C} p_y
\log
\frac{\exp(\mathrm{sim}(g(y),f(I)))}
{\sum_{\hat{y}=1}^{C}\exp(\mathrm{sim}(g(\hat{y}),f(I)))}
$$

이 loss는 CP가 domain-specific shortcut을 배우지 않고, **domain-agnostic class representation** 으로 남도록 강제한다.

### 4. Domain Ensemble과 DaTri

저자들은 여기서 한 발 더 나간다. 이미지 feature를 두 가지 관점으로 본다.

- **learned universal domain feature**
- **frozen CLIP domain feature**

DePro는 둘 중 하나만 고집하지 않고, regulation loss를 통해 두 feature가 너무 멀어지지 않도록 만든다. 쉽게 말해 학습된 universal domain 쪽으로 적응하되, CLIP이 이미 갖고 있던 안정적인 일반 지식도 버리지 않겠다는 전략이다.

또 metric learning도 강화한다. 일반 triplet-hard loss는 hardest positive / hardest negative를 이용하지만, UCDR에서는 **positive가 다른 domain에서 오도록 만드는 것** 이 더 중요하다. 그래서 DePro는

- **DPK domain sampler**
- **Domain-aware Triplet-Hard (DaTri) loss**

를 제안한다. 핵심은 positive pair를 cross-domain으로 더 강하게 만들고, domain 차이를 견디는 retrieval space를 학습하는 것이다.

![DaTri and DPK sampler](/assets/images/papers/depro-review/figure3_datri_sampler.png)

최종 loss는 다음처럼 구성된다.

$$
\mathcal{L}
= \mathcal{L}_{itm}
 + \mathcal{L}^{itm}_{dec}
 + \mathcal{L}_{dtr}
 + \mathcal{L}^{dtr}_{dec}
 + \mathcal{L}_{reg}
$$

즉 DePro는 단순 prompt 논문이 아니라,

- prompt decoupling
- domain ensemble
- cross-domain metric learning

을 함께 묶은 구조라고 보는 편이 정확하다.

## Result

결과 메시지는 꽤 분명하다. DePro는 ProS를 포함한 기존 CLIP 기반 UCDR 방법들보다 더 강한 성능을 보인다.

### 1. DomainNet UCDR / U_DCDR 성능

가장 큰 benchmark인 DomainNet에서 DePro는 unseen gallery와 mixed gallery 모두에서 ProS를 일관되게 넘는다. 예를 들어 `Sketch` query domain 설정에서:

- **ProS**: `mAP@200 0.6457`, `Prec@200 0.6001`
- **DePro**: `mAP@200 0.6936`, `Prec@200 0.6521`

처럼 꽤 의미 있는 폭으로 향상된다.

![DomainNet results](/assets/images/papers/depro-review/table1_domainnet_results.png)

이 결과는 단순히 backbone이 좋아서가 아니라,

- domain-specific prompt 대신 UDP를 사용해 domain invariance를 더 잘 만들고
- CP를 decoupling하여 semantic 역할을 분명히 하고
- DaTri로 cross-domain metric structure를 강화한 것

의 효과로 해석할 수 있다.

### 2. U_CCDR에서도 개선

Sketchy와 TU-Berlin에서도 DePro는 ProS보다 더 높다.

- **Sketchy**
  - ProS: `mAP@200 0.6991`
  - DePro: `mAP@200 0.7470`
- **TU-Berlin**
  - ProS: `mAP@all 0.6675`
  - DePro: `mAP@all 0.7014`

즉 DePro는 unseen domain만이 아니라 unseen class generalization에서도 이득을 준다.

### 3. Ablation이 말해주는 핵심

ablation도 논문의 주장과 잘 맞는다.

- 단순 universal prompt baseline만으로도 domain-specific prompt보다 낫다.
- ASP를 더하면 성능이 소폭 오른다.
- `L^{itm}_{dec}` 를 넣으면 class prompt decoupling 효과가 확실히 난다.
- `L_reg` 와 LN-trick, DaTri를 더하면 단계적으로 계속 오른다.

즉 DePro의 성능 향상은 한 가지 트릭 때문이 아니라, **prompt decoupling + domain ensemble + domain-aware metric learning** 이 서로 맞물리며 생긴 결과다.

![Ablation results](/assets/images/papers/depro-review/table2_ablation_results.png)

### 4. Feature visualization

t-SNE 시각화에서도 frozen CLIP이나 ProS보다 DePro가 unseen class cluster를 더 잘 분리한다. 특히 query domain과 gallery domain의 샘플이 더 안정적으로 class 중심으로 묶이는 모습을 보이며, 이는 retrieval에 중요한 임베딩 구조가 더 domain-robust하게 형성되었다는 뜻이다.

![t-SNE visualization](/assets/images/papers/depro-review/figure4_tsne_results.png)

## 정리

DePro는 UCDR 문제를 단순히 "prompt를 더 잘 만들자"로 보지 않는다. 대신 다음처럼 역할을 분해한다.

- 도메인 공통성은 **Universal Domain Prompt**
- 클래스 의미는 **Adaptive Class Prompt**
- 둘의 충돌은 **Decoupling Loss**
- cross-domain retrieval 구조는 **DaTri + DPK sampler**

로 풀어낸다.

결국 이 논문의 핵심은 다음 한 문장으로 요약할 수 있다.

> **DePro는 universal domain prompt와 adaptive class prompt를 분리하고, decoupling loss로 class prompt를 domain-agnostic하게 유지하며, domain-aware metric learning으로 UCDR 일반화를 끌어올린다.**

개인적으로 이 논문의 가장 좋은 점은, 기존 ProS/UCDR-Adapter 계열이 암묵적으로 안고 있던 문제인 "class prompt가 정말 class만 보고 있는가?"를 정면으로 다뤘다는 데 있다. 그래서 DePro는 단순한 후속작이 아니라, prompt-based UCDR의 설계 원칙을 한 단계 더 분명하게 만든 논문이라고 볼 수 있다.
