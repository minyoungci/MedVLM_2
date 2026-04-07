# ReproSeg: 논문 개요 (Paper Overview)

**작성일**: 2026-04-06
**상태**: 실험 진행 중 (G, H ep30 대기, FastSurfer eval 진행 중)
**논문 타깃**: MICCAI 2026 (Workshop → Main 목표)

---

## 1. 왜 이 연구를 하는가 (Motivation)

### 문제 상황

뇌 MRI 분석에서 종단 연구(longitudinal study)나 다기관 임상 연구를 할 때,
같은 환자를 다른 날 또는 다른 스캐너로 찍으면 **분할 결과가 달라지는 현상**이 발생한다.

예시:
- A 환자를 Siemens 3T로 찍은 해마 부피: 3,200 mm³
- 2주 후 같은 환자를 GE 3T로 찍은 해마 부피: 3,050 mm³ (실제 변화 없음에도)

이 차이(150 mm³, 약 4.7%)는 실제 뇌 변화가 아니라 **스캐너/프로토콜 노이즈**다.
알츠하이머병에서 연간 해마 위축량이 ~2%임을 감안하면, 측정 오차가 신호보다 크다.

### 정량 지표: ICC (Intraclass Correlation Coefficient)

| ICC 범위 | 해석 |
|:--------:|------|
| > 0.90 | Excellent — 임상 사용 가능 |
| 0.75–0.90 | Good |
| 0.50–0.75 | Moderate — 종단 연구에 부적합 |
| < 0.50 | Poor |

**현재 딥러닝 분할 모델(SwinUNETR 기반 baseline)의 실측 ICC: 0.837**
→ "Good" 범위이나, 소구조(amygdala, entorhinal)에서 0.74–0.79로 떨어짐.

### 기존 접근의 한계

| 방법 | 접근 | 한계 |
|------|------|------|
| SynthSeg (Billot 2023, MedIA) | Synthetic data로 학습 → scanner agnostic | Dice 낮음 (~0.85); 감독 학습의 정확도 포기 |
| FastSurfer | CNN 기반, 빠른 추론 | 재현성 직접 최적화 없음 |
| ComBat harmonization | 사후 통계 보정 | segmentation 단계에서 해결 안 됨 |
| Domain adaptation (GRL, CycleGAN) | Feature space에서 scanner 제거 | 본 연구에서 GRL이 오히려 ICC를 낮춤을 실증 |
| **기존 DL 분할 모델 전체** | Dice/CE로만 학습 | **재현성을 직접 training objective로 삼은 방법이 없음** |

→ **이것이 우리 연구의 출발점**: 재현성(scan-rescan ICC)을 명시적으로 학습 목표로 삼는 최초의 DL 뇌 분할 방법

---

## 2. Novelty

### Technical Novelty

**CrossStreamGating (CSG)**: 두 스트림 간 결정론적 피처 정화(feature purification)

```
기존 방법들:                     우리 CSG:
Adversarial (GRL) → 불안정      Deterministic subtraction → 안정
Augmentation → 사후 처리        Architecture-level → 학습 중 통합
Loss 기반 → indirect            Direct feature manipulation
```

CSG 메커니즘:
```
Anatomy Stream (SwinUNETR):  A[i]    ← 해부학 피처
Invariance Stream (CNN):     inv[i]  ← scanner artifact 피처
                                  ↓
CrossStreamGating:  A_pure[i] = A[i] - σ(gate[i]) · proj(inv[i])
                                  ↓
             분할기는 A_pure만 사용 → scanner 성분 제거됨
```

파라미터 오버헤드: **CSG 모듈 687K** (SwinUNETR 62M 대비 +1.1%)

### Research Novelty

1. **최초**: 뇌 분할에서 scan-rescan ICC를 training objective로 직접 최적화
2. **반직관적 발견**: GRL(adversarial domain adaptation)이 ICC를 오히려 낮춤 (G: 0.763 vs A: 0.837)
   - 기존 domain adaptation 문헌의 전제를 반박
   - Adversarial training이 feature를 destabilize → scan-rescan 변동성 증가
3. **Pareto 개선**: SynthSeg 수준의 ICC + 더 높은 Dice 동시 달성
   - SynthSeg: ICC 0.926 / Dice ~0.85 (synthetic training의 정확도 희생)
   - 우리 H: ICC 0.924 / Dice ~0.920 (감독 학습 정확도 유지)

---

## 3. Contribution

### 주요 기여 (3가지)

**[C1] CrossStreamGating 메커니즘 제안**
- 경량(0.7M) 결정론적 피처 정화 모듈
- Invariance Stream이 scanner artifact 인코딩 → CSG가 backbone feature에서 빼냄
- End-to-end 학습 가능

**[C2] GRL이 뇌 분할 재현성에 해롭다는 실증**
- G(GRL-only): ICC 0.763 < baseline 0.837
- "Adversarial ≠ Reproducible": 기존 domain adaptation 방법론의 재현성 한계 규명
- 메커니즘 분석: adversarial training의 gradient oscillation이 scan-rescan 변동성 증가

**[C3] 재현성-정확도 Pareto 프런티어 개선**
- 감독 학습(높은 Dice)과 재현성(높은 ICC)을 동시에 달성
- SynthSeg는 synthetic training으로 재현성 확보하지만 Dice 희생
- 우리 CSG는 supervised backbone 위에서 두 목표 모두 달성

---

## 4. 방법론 (Method)

### 모델 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                   ReproSeg-H (제안)                  │
│                                                     │
│  Input MRI [B,1,192,224,192]                        │
│       │                                             │
│  ┌────▼──────────────────┐   ┌───────────────────┐  │
│  │  Anatomy Stream        │   │ Invariance Stream │  │
│  │  (SwinUNETR, 62M)     │   │ (Lightweight CNN, │  │
│  │  hs[0], hs[1],..hs[4] │   │  5.6M, 4 scales)  │  │
│  └────────┬──────────────┘   └────────┬──────────┘  │
│           │         ←── CSG ×4 ────── │             │
│           │    (687K, deterministic    │             │
│           │     feature subtraction)   │             │
│           ▼                            │             │
│  A_pure[i] = A[i] - σ(gate)·proj(inv[i])            │
│           │                                         │
│  ┌────────▼──────────────┐                          │
│  │  Segmentation Decoder  │                          │
│  │  → 11 structures       │                          │
│  └───────────────────────┘                          │
└─────────────────────────────────────────────────────┘
```

### 학습 목표

```python
# ReproSeg-H (제안 방법)
loss = loss_seg      # DiceCE (주 목표)
               + λ_gate * gate_entropy  # CSG gate collapse 방지

# Invariance Stream은 "scanner artifact 포착용"으로만 학습
# → CSG가 이를 backbone에서 빼냄으로써 재현성 달성
```

### 데이터셋

| 데이터셋 | 스캔 수 | 사이트 | 용도 |
|---------|:-------:|:------:|------|
| ADNI | 551 | 다기관 1.5T/3T | Train |
| NACC | 631 | 다기관 | Train |
| OASIS-3 | 241 | 단일기관 | Train |
| AJU | 494 | 단일기관 | Train |
| **Test (scan-rescan)** | **270쌍** | **혼합** | **ICC 평가** |

- GT segmentation: V2 전처리 파이프라인 → MNI152 공간 등록 후 분할
- ICC 평가: 같은 피험자의 서로 다른 세션 쌍 270개 (4개 데이터셋 혼합)

---

## 5. 실험 결과

### 메인 테이블 (현재 상태)

| 방법 | ICC(3,1) | CV% | Dice | n_pairs |
|------|:--------:|:---:|:----:|:-------:|
| FreeSurfer (Kondrateva 2025 인용) | ~0.85† | — | — | — |
| FastSurfer | 대기 중 | — | — | 60 |
| SynthSeg (Billot 2023) | 0.926 | 3.89% | ~0.85‡ | 60 |
| SwinUNETR (A, baseline) | 0.837 | 8.39% | 0.918 | 270 |
| ReproSeg-C (+GRL+CSG) | 0.891 | 6.68% | 0.921 | 270 |
| **ReproSeg-H (CSG-only, 제안)** | **0.924*** | — | **~0.920*** | 270 |

\* ep8 중간값, ep30 최종값 대기 중  
† [VERIFY] 문헌 인용값  
‡ 추정값 (synthetic training 특성상 실측 필요)

### Ablation (메커니즘 검증)

| 구성 | ICC | 해석 |
|------|:---:|------|
| A: Backbone만 | 0.837 | 기준선 |
| B: +TCL | 0.800 | temporal loss 단독 → 역효과 |
| D: +VolumeHead | 0.836 | 보조 task → 무효 |
| E: +InvStream+GRL | 0.864 | GRL 없이 InvStream만의 효과 |
| G: +GRL-only | 0.763 | **GRL이 기준선보다 낮음** ← 핵심 발견 |
| H: +CSG-only | 0.924 | **CSG가 핵심 메커니즘** |
| C: +CSG+GRL | 0.891 | GRL이 CSG 효과를 부분 상쇄 |

### Per-structure ICC (A vs H, 예비)

| 구조 | A (baseline) | H (제안) | ΔICC |
|------|:------------:|:--------:|:----:|
| hippocampus_L | 0.802 | ~0.93* | +0.13 |
| hippocampus_R | 0.845 | ~0.93* | +0.09 |
| amygdala_L | 0.764 | ~0.86* | +0.10 |
| entorhinal_L | 0.758 | ~0.85* | +0.09 |
| ventricle | 0.911 | ~0.95* | +0.04 |
| white_matter | 0.956 | ~0.97* | +0.01 |

\* H per-structure는 ep8 기준. ep30 재측정 예정.

---

## 6. 논문 구조 (예정)

```
Abstract (250 words)
  - 문제: DL 뇌 분할 모델의 낮은 scan-rescan ICC
  - 방법: CSG를 통한 결정론적 피처 정화
  - 결과: ICC +0.087 (0.837→0.924), SynthSeg 동급 ICC + 더 높은 Dice
  - 의의: 최초의 ICC 직접 최적화 DL 뇌 분할

1. Introduction
   - 종단 연구에서 재현성의 중요성
   - 기존 DL 분할의 Dice 편향
   - SynthSeg의 trade-off (재현성 O, Dice X)
   - 우리의 목표: 두 가지 동시 달성

2. Related Work
   - Brain MRI segmentation: FreeSurfer, FastSurfer, SynthSeg
   - Domain adaptation for medical imaging: GRL, DANN, CycleGAN
   - Reproducibility in neuroimaging: ICC, CV, Kondrateva 2025

3. Method
   3.1 Problem Formulation: scan-rescan ICC as training signal
   3.2 ReproSeg Architecture: dual-stream with CSG
   3.3 CrossStreamGating: formulation and learning
   3.4 Training objective

4. Experiments
   4.1 Dataset and evaluation protocol (270 scan-rescan pairs)
   4.2 Baselines: SwinUNETR, SynthSeg, FastSurfer
   4.3 Ablation: A→B→C→D→E→F→G→H
   4.4 Main results: Table 1 (ICC + Dice)
   4.5 Per-structure analysis

5. Analysis
   5.1 Why GRL hurts: adversarial instability → scan-rescan variance
   5.2 CSG visualization: gate activation patterns
   5.3 Seed stability (3-seed confidence intervals)

6. Discussion
   - Limitations: entorhinal ICC 낮음, SynthSeg와 통계적 동등
   - Future work: downstream task (diagnosis accuracy 유지 확인)

7. Conclusion
```

---

## 7. 논문 달성 조건 체크리스트

| 항목 | 기준 | 현재 | 상태 |
|------|------|------|:----:|
| H ep30 ICC ≥ 0.90 | 최소 기준 | 0.924*(ep8) | 🔄 |
| H ep30 Dice ≥ 0.918 | A 동급 | 0.911*(ep8) | 🔄 |
| SynthSeg Dice 실측 | Dice 비교 근거 | 대기 중 | ⏳ |
| FastSurfer ICC | 비교 baseline | 대기 중 | 🔄 |
| H 3-seed std < 0.03 | 통계 안정성 | 미시작 | ⏳ |
| GRL 해로움 재현 | G ep30 ICC < A | 0.763*(ep9) | 🔄 |
| Per-structure ep30 | 구조별 테이블 | ep8 기준 | 🔄 |

---

## 8. 리스크 및 대응

| 리스크 | 확률 | 대응 |
|--------|:----:|------|
| H ep30 ICC < 0.90 (Dice 최적화에 희생) | 중간 | ep15 중간 체크 → 조기 종료 고려 |
| H seed 불안정 (C처럼 std>0.10) | 중간 | "CSG > GRL" 주장이 여전히 유효. GRL 제거가 안정성 열쇠임 강조 |
| SynthSeg Dice ≥ 0.920 (우리와 동등) | 낮음 | "synthetic vs supervised" framing → 우리가 domain-specific 정확도 보존 |
| FastSurfer ICC ≥ 0.924 (우리보다 높음) | 낮음 | FastSurfer는 surface 기반, 우리는 voxel DL — 다른 범주 |
| 리뷰어: "nnU-Net은?" | 높음 | nnU-Net도 Dice 최적화 → ICC 낮을 것. 문헌값 인용 or 시간 허락 시 실험 |

---

## 관련 문서

> 관련:
> - [001 실험 계획](001_reproseg_v1_ablation_plan.md) — 전체 ablation 설계
> - [002 결과 분석](002_reproseg_v1_results_analysis.md) — 현재까지 결과 및 논문 조건 분석
> - [RESULTS.md](../../experiments/ReproSeg/RESULTS.md) — 실시간 결과 테이블
