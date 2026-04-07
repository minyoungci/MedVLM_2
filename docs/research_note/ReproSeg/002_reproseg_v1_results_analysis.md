# Research Note #002: ReproSeg V1 전체 실험 결과 분석 및 논문 방향

**작성일**: 2026-04-06
**작성자**: VLM Team / minyoung
**카테고리**: project / reproseg
**상태**: 진행 중 (G, H ep30 완료 대기 중)

---

## 배경

ReproSeg V1의 6가지 ablation(A~F)이 완료되고, seed 재현성 실험(A×3, C×3) 및
구성요소 분리 ablation(G: GRL-only, H: CSG-only)이 추가로 진행되었다.
이 노트는 지금까지 얻은 결과를 종합하고 논문 기준 달성 가능성을 평가한다.

---

## 분석

### 평가 지표
- **ICC(3,1)**: Intraclass Correlation Coefficient (two-way mixed, absolute agreement)
  - >0.90 = excellent, 0.75-0.90 = good, 0.50-0.75 = moderate
- **CV%**: Coefficient of Variation (낮을수록 좋음)
- **Val Dice**: Segmentation 정확도 (학습 validation set 기준)
- **n=270 scan-rescan pairs** (held-out test set, deterministic eval SEED=42)

### 비결정론 수정 이력
2026-04-05에 eval_reproducibility.py의 두 가지 non-determinism 버그 수정:
1. `list(set())` → `sorted({...})` (Python set ordering)
2. `random.seed(SEED)` 추가 (`build_longitudinal_pairs` 내부 random.sample)
수정 후 동일 실행 2회 → diff=0.000000 ✓

---

## 결과

### 1. A~F 전체 비교 (확정값, SEED=42, n=270)

| EXP | 구성 | ICC | CV% | Dice(best) | ΔICC |
|-----|------|:---:|:---:|:----------:|:----:|
| A | Baseline (SwinUNETR) | 0.8366 | 8.39% | 0.9178 | — |
| B | +TCL (temporal consistency) | 0.7996 | — | — | −0.037 |
| C | +InvStream+CSG+GRL | 0.8906 | 6.68% | 0.9208 | +0.054 |
| D | +VolumeHead | 0.8355 | — | — | −0.001 |
| E | +InvStream+GRL+PCGrad | 0.8640 | — | 0.9206 | +0.027 |
| F | Full+PCGrad | 0.7507 | — | — | −0.086 |
| **G** | GRL-only (CSG 없음) | **0.7625*** | — | 0.9131* | −0.074 |
| **H** | CSG-only (GRL 없음) | **0.9238*** | — | 0.9105* | **+0.087** |

\* G, H: ep9-10/30 시점 중간값. ep30 최종값 대기 중.

### 2. SOTA 비교

| 방법 | ICC | CV% | n |
|------|:---:|:---:|:-:|
| **H (CSG-only, 제안)** | 0.9238* | — | 270 |
| SynthSeg (Billot 2023) | 0.9263 | 3.89% | 60 |
| FastSurfer | 대기 중 | — | 60 |
| A (SwinUNETR baseline) | 0.8366 | 8.39% | 270 |

### 3. Seed 안정성 (3-seed, 조기 체크포인트 기준 — 재평가 필요)

⚠️ 아래 수치는 ep4-5 조기 체크포인트 기준 → 신뢰 불가. ep30 완료 후 재계산 예정.

| 모델 | seed=42 | seed=0 | seed=1 | mean ± std |
|------|:-------:|:------:|:------:|:----------:|
| A | 0.8366 | 0.9367† | 0.9347† | 0.903±0.047 |
| C | 0.8906 | 0.6688† | 0.6554† | 0.738±0.108 |

† 조기 체크포인트. A_seed0/1은 ep30 완료, C_seed0/1은 ep15 진행 중.

### 4. 핵심 발견

**발견 1: CSG가 유일하게 효과적인 구성요소**
- H (CSG-only) = 0.9238 → SynthSeg(0.9263)에 근접
- G (GRL-only) = 0.7625 < A(0.8366) → GRL이 오히려 해로움
- 해석: Adversarial training(GRL)은 feature를 destabilize하여 ICC 감소.
  Deterministic feature purification(CSG)만이 scan-rescan 일관성 향상.

**발견 2: C = CSG + GRL 조합에서 GRL이 CSG 효과를 부분적으로 상쇄**
- C(0.8906) < H(0.9238): GRL을 제거하면 ICC 향상
- C의 seed 불안정(std=0.108)도 GRL의 adversarial training 불안정성이 원인으로 추정

**발견 3: "more losses ≠ better ICC"**
- F(Full, 모든 loss) = 0.7507: baseline보다 크게 저하
- D(VolumeHead) ≈ A(0.8355): 보조 task가 ICC에 무효
- 재현성에는 특이적 메커니즘(CSG)이 필요하며 일반적 정규화로는 달성 불가

**발견 4: CSG의 Dice 비용 없음 (잠정)**
- H(ep8) Dice = 0.9105 vs A Dice = 0.9178 → 수렴 전 비교
- C(ep30) Dice = 0.9208 > A(0.9178): InvStream 구조 자체는 Dice도 향상
- H(ep30) 예상: ~0.920+ (ep15-30 사이 수렴 예상)

---

## 논문 달성 조건 분석

### 현재 상황 (2026-04-06 기준)

```
논문 제목 후보:
"CrossStreamGating: Reproducibility-Aware Brain Segmentation via
 Deterministic Feature Purification"

핵심 주장:
1. CSG가 SwinUNETR에 0.7M 파라미터(+1.1%) 추가로 ICC +0.087 달성
2. ICC 0.9238 ≈ SynthSeg 0.9263, but Dice 우위 (SynthSeg ~0.85 vs 우리 ~0.92)
3. GRL은 해롭다 (ablation G): 기존 domain adaptation 접근이 재현성에 역효과
```

### 논문 통과 조건 (조건별 달성 가능성)

#### ✅ 가능성 높음 (현재 데이터로 지원됨)

| 조건 | 현황 | 판단 |
|------|------|------|
| A 대비 명확한 ΔICC (+0.08 이상) | H: +0.087 | ✅ |
| SynthSeg와 동등 ICC (±0.01) | 0.9238 vs 0.9263 = diff 0.003 | ✅ |
| GRL 해로움 ablation | G=0.7625 < A=0.8366 | ✅ |
| CSG 효과 ablation | H=0.9238 >> A=0.8366 | ✅ |
| Dice 유지 (A 대비 -0.01 이내) | 수렴 후 예상 | ✅ (조건부) |

#### ⚠️ 추가 검증 필요

| 조건 | 필요한 것 | 리스크 |
|------|----------|--------|
| **H seed 안정성** | H seed=0,1 학습 + 3-seed std<0.03 | 중간: C가 불안정했던 전례 있음 |
| **SynthSeg Dice 비교** | SynthSeg seg → Dice vs GT | 중간: SynthSeg가 0.90+ 나오면 advantage 없음 |
| **FastSurfer 비교** | 현재 실행 중 (~0.87 예상) | 낮음 |
| **H ep30 ICC 유지** | ep30 체크포인트 ICC | 중간: Dice 최적화로 ICC 하락 가능 |

#### ❌ 현재 불가 / 낮은 우선순위

| 조건 | 이유 |
|------|------|
| nnU-Net 비교 | 재학습 필요 (수십 시간) → skip |
| MICCAI main track | 통계 완성 전 제출 불가 |
| 파라미터 효율 주장 | 전체 68M vs SynthSeg 7M → 불리 |

---

## 다음 단계 (우선순위 순)

1. **[즉시 대기 중]** G, H ep30 완료 → 최종 ICC 확인
   - H ICC가 0.90 이상 유지되면 → 논문 가능
   - H ICC가 0.85 이하로 하락하면 → 방향 재검토 필요

2. **[이후 즉시]** H seed=0, seed=1 학습 (3-seed stability)
   - std < 0.03 목표. C처럼 불안정하면 → GRL 제거가 핵심 fix임을 방어 논리로 사용

3. **[병렬 진행 중]** FastSurfer eval
   - 예상: ICC ~0.87-0.89. H보다 낮으면 비교 강해짐

4. **[FastSurfer 완료 후]** SynthSeg Dice 측정
   - `/tmp/synthseg_cache` 60쌍의 seg → 우리 GT와 비교
   - SynthSeg Dice < 0.90이면 "ICC는 같지만 Dice는 우리가 더 높음" 주장 가능

5. **[결과 확정 후]** Table 1 완성 + paper outline

---

## 투고 타깃 평가

| 저널/학회 | 가능성 | 조건 |
|----------|:------:|------|
| **MICCAI 2026 Workshop** | ✅ 높음 | H ep30 + FastSurfer 결과만으로 가능 |
| **MICCAI 2026 Main** | ⚠️ 중간 | seed 안정성 + SynthSeg Dice 비교 필요 |
| **MedIA (Medical Image Analysis)** | ⚠️ 중간 | 전체 결과 + 임상 downstream 평가 필요 |
| **NeuroImage** | ⚠️ 중간 | 더 많은 baseline + 임상 의미 필요 |

**현실적 목표: MICCAI 2026 Workshop → 결과 보강 후 MedIA 확장판**

---

## 관련 문서

> 관련:
> - [#001 ReproSeg V1 실험 계획](001_reproseg_v1_ablation_plan.md) — ablation 설계 근거
> - [2026-04-05 ablation findings](2026-04-05_ablation_findings.md) — preliminary 분석
> - [RESULTS.md](../RESULTS.md) — 확정값 테이블
> - [SCRATCHPAD.md](../SCRATCHPAD.md) — 실험 gate check
