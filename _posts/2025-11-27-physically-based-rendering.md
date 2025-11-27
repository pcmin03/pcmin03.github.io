---
title: Physically Based Rendering Cheat Sheet
categories: [3D Geometry]
tags: [3D Geometry, Rendering, PBR]
article_header:
  type: overlay
  theme: dark
  background_color: '#1e1b4b'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(30, 27, 75, .8), rgba(52, 211, 153, .45))'
    src: /assets/images/study/3d-geometry.jpg
---

PBR(Physically Based Rendering)는 현대 실시간/오프라인 렌더러의 기본 규칙이다. 여기에서는 핵심 수식과 파라미터가 어떤 의미를 가지는지 요약한다.

## 1. BRDF 구성 요소
- **Diffuse**: Lambertian, $f_d = \frac{c}{\pi}$
- **Specular**: Microfacet 모델, $f_s = \frac{D(h)F(v,h)G(n,v,l)}{4(n\cdot v)(n\cdot l)}$
- **Energy conservation**을 위해 $f_d + f_s \le 1$을 만족시키도록 mixing한다.

## 2. GGX Microfacet
- Normal Distribution Function: $D(h) = \frac{\alpha^2}{\pi((n\cdot h)^2(\alpha^2-1)+1)^2}$
- Geometry term( Smith )은 view, light 방향별 masking·shadow를 보정한다.
- **Roughness** $r$는 $\alpha = r^2$로 매핑해 perceptual linearity를 유지한다.

## 3. Fresnel-Schlick Approximation
$F(v,h) = F_0 + (1 - F_0)(1 - (v\cdot h))^5$
- $F_0$는 금속 여부에 따라 다르게 세팅한다 (금속은 베이스 컬러, 비금속은 0.04 근처).

## 4. Texture Workflow
- **Metallic-Roughness**: baseColor, metallic, roughness, normal, AO, emissive 텍스처를 표준으로 사용.
- **Linear space**로 샘플링한 뒤 필요한 경우 gamma correction을 적용한다.
- Normal map은 `TBN` 행렬로 tangent 공간에서 월드 공간으로 변환한다.

> 최소한의 공식을 이해하면 엔진/렌더러가 무엇을 가정하는지 명확해지고, 아티스트와 엔지니어 사이 커뮤니케이션이 쉬워진다.
