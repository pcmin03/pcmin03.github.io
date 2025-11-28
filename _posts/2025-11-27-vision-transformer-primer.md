---
title: Vision Transformer Primer
categories: [Deep Learning]
tags: [Deep Learning, Vision Transformers, Attention]
article_header:
  type: overlay
  theme: dark
  background_color: '#0d1b2a'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(13, 27, 42, .85), rgba(176, 125, 103, .65))'
    src: /assets/images/study/deep-learning.jpg
---

ViT는 CNN을 대체하는 기본 백본으로 자리 잡았다. 이 글에서는 **Patch tokenization → Multi-head Self-Attention → MLP Head**로 이어지는 파이프라인을 짧게 정리한다.

## 1. Patch Embedding
이미지를 $16\times16$ 혹은 $14\times14$ 패치로 자른 뒤 linear projection을 통과시키면 $N$개의 토큰이 된다. CLS 토큰을 prepend해 전체 이미지 표현을 학습한다.

## 2. Positional Encoding
사각 격자 위치를 학습 가능한 `pos_embed`로 더해주면 attention이 공간 정보를 인지한다. 최근에는 2D Rotary Embedding이나 ALiBi가 자주 사용된다.

## 3. Transformer Encoder
각 블록은
1. LayerNorm
2. Multi-Head Attention
3. Residual connection
4. MLP (GELU + Dropout)
으로 구성된다. Stochastic Depth를 적용하면 학습 안정성이 크게 올라간다.

## 4. Fine-tuning Tips
- **Token pruning**: 고해상도에서 latency를 줄이려면 early layer attention map 기준으로 토큰을 줄인다.
- **Prompting & Adapters**: 소규모 데이터 셋에서는 lightweight adapter/prompt를 삽입해 전체 파라미터 업데이트를 피한다.
- **Hybrid heads**: CLS 토큰과 Global Average Pooling을 함께 사용하면 안정적으로 수렴한다.

> ViT는 구조가 단순하지만, positional encoding, regularization, adapter 설계에 따라 성능 편차가 크다. 위 원칙을 기본으로 두면 어떤 도메인에도 쉽게 확장할 수 있다.
