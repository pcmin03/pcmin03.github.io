---
title: Deep Learning Systems for Reliable Vision
categories: [Deep Learning]
tags: [Deep Learning, Computer Vision, MLOps]
article_header:
  type: overlay
  theme: dark
  background_color: '#0d1b2a'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(13, 27, 42, .85), rgba(176, 125, 103, .65))'
    src: /assets/images/study/deep-learning.jpg
---

연구용 prototype을 실제 서비스로 전환할 때 고려해야 할 데이터 파이프라인, 학습 인프라, 배포 전략을 정리합니다.

<!--more-->

## 1. Data Engine
- **Continuous labeling**: 모델 피드백을 바탕으로 hard example을 우선 수집해 annotation 비용을 줄인다.
- **Auto-validations**: 입력 형식, class imbalance, annotation drift를 자동 감시해 학습 직전에 문제를 발견한다.
- **Augmentation recipes**: RandAugment, CutMix, style transfer 등을 dataset 프로필별로 관리해 재현 가능한 실험 환경을 만든다.

## 2. Training Stack
- **Config-first experiment**: Hydra/Weights & Biases 같은 도구로 실험 구성을 코드에서 분리하고, 각 실험을 Git hash + config로 추적한다.
- **Mixed precision & gradient checkpointing**: GPU 메모리를 30~50% 절약하면서 batch size를 크게 가져갈 수 있어 수렴 속도가 빨라진다.
- **EMA & SWA**: production 배포 전 성능 안정화를 위해 moving average weight를 기본으로 쓴다.

## 3. Deployment & Monitoring
- **TorchScript/ONNX export**로 모델을 일관된 형태로 패키징하고, TensorRT 혹은 OpenVINO를 통해 지연 시간을 줄인다.
- **Shadow deployment**: 새로운 모델을 실제 트래픽에 숨겨서 돌리며 online metric을 비교한다.
- **Drift dashboard**: 특성 분포, false positive 비율, latency 등을 주기적으로 기록해 운영 이슈를 조기에 발견한다.

> 연구 성능 못지않게, **데이터 → 학습 → 배포** 전 과정을 자동화하는 것이 신뢰할 수 있는 딥러닝 시스템의 핵심이다.
