---
date: 2026-04-07
project: reproseg_v1
status: completed
tags: [ablation, grl, csg, reproducibility, icc]
---

# Research Note #003: Ablation G/H 결과 및 A~H 종합 분석

## 1. 배경

G (GRL-only) 및 H (CSG-only) ablation은 C (InvStream+CSG+GRL) 구성에서 각 컴포넌트의 개별 기여를 분리하기 위해 설계됐다.
- **G**: InvStream + GRL만 활성. CSG 없음 (a_pure = backbone feature 그대로).
- **H**: CSG만 활성. GRL/InvStream 없음. CSG gate는 inv_stream 없이 backbone feature 자체에서 학습.

두 실험 모두 ep30 완료 후 deterministic eval (SEED=42, n=270) 수행.

**주의**: G에 대해 두 개의 ICC 수치가 존재함.
- `icc_G.json` (2026-04-05 생성): mean_icc=0.7625 — 훈련 중간 체크포인트 기반.
- `icc_G_final.json` (2026-04-07 생성): mean_icc=0.9319 — ep30 완료 후 최종 best.pt 재평가.
두 평가 모두 동일한 `best.pt` 경로를 참조하지만 실제 파일이 다른 시점에 업데이트됨.
**확정값으로 icc_G_final.json (0.9319) 사용.**

---

## 2. 실험 설정

| 항목 | G (GRL-only) | H (CSG-only) |
|------|:------------:|:------------:|
| 실험명 | `reproseg_v1_G_grl_only` | `reproseg_v1_H_inv_only` |
| Mode | `inv` (CSG bypass) | `baseline` + CSG active |
| InvStream | 활성 | 비활성 |
| GRL | 활성 (alpha schedule) | 비활성 |
| CSG × 4 | 비활성 (gate=0 고정) | 활성 |
| GPU | B200 × 2 | B200 × 2 |
| Epochs | 30 | 30 |
| Dataset | n=1,917 (cross-sect.) | n=1,917 (cross-sect.) |
| Eval | n=270 pairs, SEED=42 | n=270 pairs, SEED=42 |

### G의 Gradient 흐름

```
loss_seg  → backbone encoder/decoder
loss_inv  → site_classifier ← GRL ← inv_stream ← backbone
             (backbone이 scanner-invariant 표현 학습 강제)
CSG       → 없음 (gate=0 bypass)
```

### H의 Gradient 흐름

```
loss_seg  → backbone encoder/decoder, CSG (a_pure 경유)
             (CSG gate가 순수 segmentation loss로만 학습)
GRL/inv   → 없음
```

---

## 3. 결과

### 3.1 G (GRL-only) — ep30 최종 ICC

출처: `/home/vlm/minyoung2/experiments/ReproSeg/results/icc_G_final.json`

| Structure | ICC | CV% |
|-----------|:---:|:---:|
| hippocampus_L | 0.9197 | 3.75% |
| hippocampus_R | 0.9375 | 3.68% |
| amygdala_L | 0.9256 | 4.98% |
| amygdala_R | 0.9377 | 4.36% |
| entorhinal_L | 0.8827 | 7.22% |
| entorhinal_R | 0.8825 | 7.40% |
| ventricle | 0.9462 | 12.08% |
| white_matter | 0.9618 | 3.00% |
| cortical_L | 0.9406 | 2.83% |
| cortical_R | 0.9452 | 2.80% |
| subcortical | 0.9710 | 1.66% |
| **MEAN** | **0.9319** | **4.89%** |

**Best val_dice**: 0.9196 (ep23), n_pairs=270

### 3.2 H (CSG-only) — ep30 최종 ICC

출처: `/home/vlm/minyoung2/experiments/ReproSeg/results/icc_H.json`

| Structure | ICC | CV% |
|-----------|:---:|:---:|
| hippocampus_L | 0.9140 | 3.33% |
| hippocampus_R | 0.9350 | 3.30% |
| amygdala_L | 0.8902 | 4.39% |
| amygdala_R | 0.7636 | 16.36% |
| entorhinal_L | 1.0000 | 0.00% |
| entorhinal_R | 0.8658 | 5.66% |
| ventricle | 0.9472 | 11.41% |
| white_matter | 0.9738 | 1.97% |
| cortical_L | 0.9484 | 2.49% |
| cortical_R | 0.9522 | 2.41% |
| subcortical | 0.9715 | 1.59% |
| **MEAN** | **0.9238** | **4.81%** |

**Best val_dice**: 0.9211 (ep27), n_pairs=270

---

## 4. A~H 전체 비교 테이블

출처: `icc_A.json`, `icc_B.json`, `icc_C.json`, `icc_D.json`, `icc_E.json`, `icc_F.json`, `icc_G_final.json`, `icc_H.json`
평가: SEED=42, n=270 scan-rescan pairs

| EXP | 구성 | ICC | CV% | Best Dice | ΔICC vs A |
|-----|------|:---:|:---:|:---------:|:---------:|
| A | Baseline (SwinUNETR) | 0.8366 | 8.39% | 0.9178 | — |
| B | +TCL+FeatCons | 0.7996 | 8.85% | — | −0.037 |
| C | +InvStream+CSG+GRL | 0.8906 | 6.60% | 0.9208 | +0.054 |
| D | +VolumeHead | 0.8355 | 8.40% | — | −0.001 |
| E | +InvStream+GRL+PCGrad | 0.8640 | 7.21% | 0.9206 | +0.027 |
| F | Full+PCGrad | 0.7507 | 10.89% | — | −0.086 |
| **G** | **GRL-only** | **0.9319** | **4.89%** | 0.9196 | **+0.095** |
| **H** | **CSG-only** | **0.9238** | **4.81%** | 0.9211 | **+0.087** |

**순위 (ICC)**: G(0.9319) > H(0.9238) > C(0.8906) > E(0.8640) > A(0.8366) > D(0.8355) > B(0.7996) > F(0.7507)

**순위 (CV%)**: H(4.81%) < G(4.89%) < E(7.21%) < C(6.60%) < A(8.39%) < B(8.85%) < D(8.40%) < F(10.89%)

### SOTA 비교 (ICC)

| 방법 | ICC | CV% | n_pairs | 비고 |
|------|:---:|:---:|:-------:|------|
| **G (GRL-only, 제안)** | **0.9319** | 4.89% | 270 | 최종 확정값 |
| SynthSeg (Billot 2023) | 0.9263 | 3.89% | 60 | n이 우리보다 작음 |
| **H (CSG-only)** | **0.9238** | **4.81%** | 270 | — |
| FastSurfer | 0.9030 | 5.91% | 60 | — |
| A (SwinUNETR baseline) | 0.8366 | 8.39% | 270 | — |

SynthSeg와의 평가 조건 차이 주의: SynthSeg는 n=60 pairs (우리 n=270의 22%). 소표본에서의 ICC는 일반적으로 과대추정 경향 [VERIFY].

---

## 5. Seed 안정성 결과 (3-seed)

### A (Baseline)

출처: `icc_A.json`, `icc_A_seed0.json`, `icc_A_seed1.json`

| Seed | ICC | CV% |
|------|:---:|:---:|
| seed=42 | 0.8366 | 8.39% |
| seed=0 | 0.9367 | 4.69% |
| seed=1 | 0.9347 | 4.74% |
| **mean ± std** | **0.9027 ± 0.0467** | 5.94 ± 1.73 |

seed=42의 이상치(0.8366)가 전체 std를 크게 끌어올림. seed=0,1은 일관되게 0.93대.
**해석**: A 자체도 seed에 따라 0.8~0.94 범위 변동 — seed=42가 특이 초기화이거나 학습이 다른 local minima에 수렴한 것으로 보임.

### C (InvStream+CSG+GRL)

출처: `icc_C.json`, `icc_C_seed0_final.json`, `icc_C_seed1_final.json`

**주의**: `icc_seed_stats.json`의 C 수치 [0.8906, 0.6688, 0.6554]는 훈련 중간 체크포인트 기반 (2026-04-05). 최종값은 아래 사용.

| Seed | ICC | CV% |
|------|:---:|:---:|
| seed=42 | 0.8906 | 6.60% |
| seed=0 (final) | 0.9323 | 4.88% |
| seed=1 (final) | 0.9350 | 4.77% |
| **mean ± std** | **0.9193 ± 0.0241** | 5.42 ± 0.98 |

최종값 기준 C의 seed 안정성은 std=0.024로 A(std=0.047)보다 오히려 안정적.
seed=42의 낮은 값(0.8906)이 C 특유의 불안정성이 아닌 공통 현상임을 A seed 결과가 지지.

### H (CSG-only)

H seed=0, seed=1 현재 훈련 중 (2026-04-07 07:05 기준 ep5/30). ICC 결과 미확보. [VERIFY — H seed 완료 후 업데이트 필요]

---

## 6. 핵심 발견

### 발견 1: G(GRL-only)가 수치상 최고 ICC — 주의 해석 필요

G (ICC=0.9319) > H (ICC=0.9238) > SynthSeg (ICC=0.9263)

이는 GRL 자체가 ICC를 향상시킨다는 의미로 보일 수 있으나, **해석 주의**가 필요하다:

1. **C(CSG+GRL=0.8906) vs G(GRL-only=0.9319)**: GRL이 CSG와 함께 쓰일 때 오히려 ICC가 낮아진다.
   - 원인 가설: CSG가 GRL과 함께 쓰일 때, adversarial gradient가 CSG gate 학습을 불안정하게 만듦.
   - GRL 단독으로는 backbone이 site-invariant 표현을 학습하면서 ICC 향상.

2. **G의 per-structure 분석**: G는 amygdala, entorhinal 전반에서 고른 향상. H의 amygdala_R(0.7636)이 유독 낮음.

3. **G vs H 차이**: ICC 기준 G>H이지만 CV% 기준 H(4.81%) < G(4.89%) — 근소한 차이.

### 발견 2: CSG+GRL 조합은 단순 합산보다 못함

| 구성 | ICC |
|------|:---:|
| GRL alone (G) | 0.9319 |
| CSG alone (H) | 0.9238 |
| CSG + GRL (C) | 0.8906 |

C < G 이고 C < H. 조합이 개별보다 성능이 낮다는 것은 두 컴포넌트 간 negative interference가 존재함을 시사.
메커니즘 가설: CSG gate가 GRL의 adversarial gradient로 인해 불안정하게 학습되고, 이것이 backbone feature purification을 오히려 방해.

### 발견 3: "more losses ≠ better ICC" 재확인

F(Full, ICC=0.7507) < A(0.8366): 전체 loss 조합이 baseline보다 크게 저하.
B(TCL, 0.7996) < A: Temporal consistency loss 단독도 역효과.
G, H는 각각 단일 메커니즘 집중으로 최고 성능.

### 발견 4: Dice cost 없음 확인

| EXP | Best Val Dice |
|-----|:-------------:|
| A (Baseline) | 0.9178 |
| G (GRL-only) | 0.9196 (+0.0018) |
| H (CSG-only) | 0.9211 (+0.0033) |

재현성 향상이 segmentation 정확도를 희생하지 않음. H는 Dice도 A보다 높음.

---

## 7. 해석의 불확실성 및 리스크

1. **G_final vs icc_G 불일치**: icc_G.json(0.7625)과 icc_G_final.json(0.9319)의 괴리가 큼. 두 파일 모두 동일 best.pt를 참조하지만 체크포인트 저장 시점이 달랐을 가능성 높음. icc_G.json은 훈련 완료 전 (ep20 이전) 체크포인트 평가로 판단.

2. **G vs H 최종 순위**: G(0.9319)가 H(0.9238)보다 높지만 차이는 0.0081. seed 변동성(A std=0.047) 대비 유의미하지 않을 수 있음. H seed 완료 후 통계 검정 필요.

3. **H amygdala_R ICC 0.7636**: 다른 구조 대비 이상치. CSG gate가 amygdala_R의 scan-rescan 변동을 충분히 제거하지 못하는 것으로 보임. G에서 amygdala_R=0.9377인 것과 대조적.

4. **n=60 vs n=270**: SynthSeg/FastSurfer 비교는 n이 다름. 직접 통계 비교 불가.

---

## 8. 다음 방향

1. **[즉시 대기]** H seed=0, seed=1 ep30 완료 (현재 ep5/30, ~25h 후)
   - H 3-seed mean±std 계산 → G vs H 통계 비교
   - H가 A보다 안정적임을 보이면 논문 방어력 향상

2. **[중기]** G vs H 어느 것을 "proposed method"로 제시할지 결정
   - G가 ICC 수치는 높지만 adversarial training의 불안정성 리스크 존재
   - H는 CSG만으로 단순하고 안정적 → 논문 contribution이 더 명확
   - **현재 권장**: H를 proposed, G를 ablation으로 유지. 단 G_final 수치를 honest하게 보고.

3. **[결과 확정 후]** Table 1 완성 및 논문 contribution 재정의

---

## 관련 파일

- `icc_G_final.json` — G 최종 ICC (ep30 best.pt)
- `icc_G.json` — G 중간 체크포인트 ICC (비교 참고용)
- `icc_H.json` — H 최종 ICC (ep30 best.pt)
- `icc_C_seed0_final.json`, `icc_C_seed1_final.json` — C 최종 seed ICC
- `icc_A_seed0.json`, `icc_A_seed1.json` — A seed ICC
- `reproseg_v1_G_grl_only/logs/training_log.json` — G 학습 커브
- `reproseg_v1_H_inv_only/logs/training_log.json` — H 학습 커브
- `h_seeds_watcher.log` — H seed 훈련 상태 모니터
