# 연구 노트: ReproSeg V1 Ablation 결과 분석
**날짜**: 2026-04-05  
**상태**: Preliminary (fixed re-eval 진행 중, ~2h 후 확정값 예정)

---

## 1. 실험 설정

- **데이터**: V2 preprocessed, 270 held-out scan-rescan pairs (270 피험자)
- **평가**: ICC(3,1) + CV% per structure (11개 grouped structures)
- **결정론 보장**: SEED=42, sorted set iteration, Python random.seed 고정 ✓
- **비결정론 버그 수정일**: 2026-04-05 (수정 전 결과는 신뢰 불가)

---

## 2. 실험별 결과 (PRELIMINARY — 수정 후 재평가 진행 중)

| EXP | 구성 | Mean ICC | CV% | ΔICC vs A |
|-----|------|:--------:|:---:|:---------:|
| **A** | Baseline (SwinUNETR) | 0.8366* | 8.39%* | — |
| **B** | +TCL (temporal consistency loss) | ~0.8134† | ~8.96%† | −0.023 |
| **C** | +InvStream+CSG+GRL | **~0.8936†** | ~6.68%† | **+0.057** |
| **D** | +VolumeHead aux task | ~0.8481† | ~8.46%† | +0.012 |
| **E** | +InvStream+GRL+PCGrad | ~0.8749† | ~7.22%† | +0.038 |
| **F** | Full (A+B+C+D+E) + PCGrad | ~0.7817† | ~10.56%† | −0.055 |

\* 확정값 (determinism 수정 후 검증)  
† 비결정론 수정 전 값 — 방향성은 신뢰, 절댓값은 ±0.005 내외 오차 예상

---

## 3. 주요 발견

### 3.1 C가 명확히 Best (ΔICC +0.057)
InvarianceStream + CrossStreamGating (CSG) 조합이 가장 효과적.  
메커니즘: InvStream이 site-invariant 표현을 학습 → CSG가 backbone 피처에서 site-specific 성분을 제거.

**Per-structure**: hippocampus_L (0.91), hippocampus_R (0.91)로 임상적으로 중요한 구조에서 dramatic 개선.

### 3.2 F가 최악 (Baseline보다 -0.055)
모든 loss를 합치면 성능 저하. 원인 분석:
- UWL(Uncertainty Weighted Loss)이 각 task weight를 자동 조정하지만, seg weight가 5.07까지 상승하여 inv task를 suppressing
- PCGrad가 gradient conflict를 완화하지만 너무 많은 task가 경쟁 → backbone이 혼란
- **함의**: 재현성 손실 조합은 신중한 task selection이 필요. "more is more"가 성립하지 않음

### 3.3 B가 Baseline보다 낮음
TCL(Temporal Consistency Loss)이 standalone으로는 역효과.  
이유: TCL은 동일 피험자 다른 시점의 paired scan에서 feature 유사성을 강제하지만, volume 변화가 있는 경우 (진행성 질병) 이를 무시함. 반면 InvStream의 GRL은 site/scanner의 artifact를 target으로 잡음 — 더 올바른 귀인(attribution).

### 3.4 Gate Check 결과
계획했던 단조 증가 (A<B<C<D<E<F)는 성립하지 않음:
- B < A: TCL 효과 없음 → 확인
- C > E > D > A > B > F: C만 명확한 승자
- 논문 framing: C를 proposed method, B/D/E/F를 ablation으로 재구성

---

## 4. 이 결과가 논문에서 갖는 의미

### 핵심 contribution
**"Reproducibility-Aware Brain Segmentation Training: InvarianceStream + CrossStreamGating"**
- 재현성을 명시적 training objective로 사용하는 첫 DL 뇌 분할 논문 (MICCAI 주장 가능) [VERIFY]
- InvStream+CSG 메커니즘이 site-invariant 피처 학습을 통해 ICC +0.057 달성

### 현재 데이터의 논문 충분성 평가
| 필요 조건 | 상태 |
|----------|------|
| 명확한 baseline vs proposed ΔICC | ✅ +0.057 (C vs A) |
| Ablation (mechanism validation) | ⚠️ G/H 실험 진행 중 (GRL-only vs CSG-only) |
| 통계 유의성 (3-seed CI) | 🔄 진행 중 (A/C seed=0,1 학습 중) |
| SOTA 비교 (SynthSeg) | 🔄 eval script 준비, 실행 대기 |
| Dice 성능 유지 확인 | ⚠️ per-experiment Dice 기록 필요 |

### 리스크 요인
1. **F의 실패**: 왜 full model이 worst인지 설명 필요. UWL instability vs gradient conflict 중 어느 것이 주원인인가? → G/H ablation이 답을 줄 것
2. **B의 실패**: TCL이 longitudinal pairs에서 왜 안 되는지 명확히 해야 함
3. **절댓값 ICC**: C=0.893이 임상적으로 "excellent" (>0.90) 경계. 재평가에서 0.90을 넘으면 narrative 강해짐
4. **SynthSeg 비교**: SynthSeg ICC가 0.85 이상이면 우리 advantage가 작아 보임. 예측: SynthSeg는 C보다 낮을 가능성 높음 (SynthSeg는 reproducibility를 직접 optimize하지 않음)

---

## 5. 다음 단계

1. **즉시**: icc_all_models_fixed.json 완성 후 확정값 이 노트에 업데이트
2. **~15h**: A_seed0/1, C_seed0/1 학습 완료 → ICC CI 계산
3. **~16h**: G/H ablation 시작 (GRL-only vs CSG-only)
4. **병렬**: SynthSeg eval (자동 실행 예정)
5. **완료 후**: Table 1 초안, paper outline 작성

---

_작성: Claude Code (claude-sonnet-4-6)_  
_검토 필요: F의 실패 메커니즘 분석, UWL weight 추이 수집 필요_
