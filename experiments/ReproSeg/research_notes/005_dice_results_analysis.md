# Research Note #005: Scan-Rescan Dice Results — Baseline vs ReproSeg

**작성일**: 2026-04-07
**카테고리**: evaluation
**상태**: 완료

---

## 배경

ICC(3,1)은 volumetric reproducibility를 측정하지만 voxel-level consistency는 다른 지표가 필요하다.
Scan-rescan Dice = Dice(seg(scan1), seg(scan2))로 동일 피험자 반복 촬영에 대한 세그멘테이션 일관성을 정량화.
GT가 없어도 측정 가능 — 임상적으로 중요한 reproducibility 지표.

---

## 결과

### ReproSeg 모델 (n=270 scan-rescan pairs, test set 20%)

| Structure | A (baseline) | G (GRL-only) | H (CSG-only) |
|-----------|-------------|-------------|-------------|
| hippocampus_L | 0.5993 | 0.6625 | 0.6616 |
| hippocampus_R | 0.5684 | 0.6612 | 0.6599 |
| amygdala_L | 0.4797 | 0.5690 | 0.5707 |
| amygdala_R | 0.5127 | 0.5646 | 0.5655 |
| entorhinal_L | 0.4171 | 0.4846 | 0.4851 |
| entorhinal_R | 0.4132 | 0.4756 | 0.4766 |
| ventricle | 0.6025 | 0.7011 | 0.7007 |
| white_matter | 0.6337 | 0.6553 | 0.6550 |
| cortical_L | 0.4493 | 0.5110 | 0.5119 |
| cortical_R | 0.4750 | 0.5137 | 0.5141 |
| subcortical | 0.7752 | 0.8235 | 0.8233 |
| **MEAN** | **0.5387** | **0.6020** | **0.6022** |

### Baselines (cached segmentation, different n)

| Method | Mean Dice | n_pairs | Note |
|--------|-----------|---------|------|
| SynthSeg | 0.6177 | 64 | only where cache exists |
| FastSurfer | 0.4383 | 60 | only where cache exists |

---

## 핵심 관찰

### 1. G ≈ H (0.6020 vs 0.6022) — ICC 결과와 완전히 일치
- ICC: G=0.9319, H=0.9318 (Δ=0.0001)
- Dice: G=0.6020, H=0.6022 (Δ=0.0002)
- GRL (explicit adversarial)과 CSG (implicit purification)이 두 메트릭에서 동등

### 2. A → G/H: +11.7% / +11.8% 향상
- 모든 구조에서 일관된 향상
- entorhinal (가장 낮음 ~0.42→0.48)부터 subcortical (0.78→0.82)까지 고르게 개선
- reproducibility training이 voxel-level consistency에도 효과적

### 3. SynthSeg 비교 주의사항 (비공정 비교)
- SynthSeg: n=64 (cache hit만), 우리: n=270 (전체 테스트셋)
- 동일 64쌍에서 비교 시 결과가 다를 수 있음
- G/H=0.602 < SynthSeg=0.618 이지만 sample 크기가 4배 다름
- ICC에서는 G/H(0.932) > SynthSeg(0.926) — volumetric reproducibility 우세

### 4. FastSurfer < A baseline — FastSurfer의 낮은 scan-rescan 일관성
- FastSurfer 0.4383 < A 0.5387 — 우리 baseline도 FastSurfer보다 우수
- FreeSurfer 기반 atlas-matching이 scan-rescan 변동에 취약함을 시사

---

## 코드 정보

- `eval_dice.py` — `scan_rescan` mode, n=270 test pairs (seed=42, 20% split)
- `eval_dice_baselines.py` — SynthSeg/FastSurfer cached segs 평가
- 결과: `results/dice_scan_rescan.json`, `results/dice_baselines.json`

---

## inv_only mode 메커니즘 확인

H(inv_only) 학습 분석:
- `loss_inv = 0.0` 항상 — GRL site classifier가 inv_only에서 비활성화 (line 516)
- `use_csg = True` — CSG forward pass는 활성화 (line 498)
- Gate entropy regularizer 활성화 (line 538-543)
- 결론: H는 "seg loss + CSG 아키텍처 + gate entropy"만으로 학습
- 시사점: **CSG의 reproducibility 향상은 explicit adversarial loss 없이도 달성 가능**
  → forward pass의 feature purification이 충분히 effective

---

## 다음 단계

1. H_seed0/1 ep30 완료 후 ICC → 3-seed mean±std
2. G/H Dice on same 64 pairs as SynthSeg (공정 비교를 위해)
3. Table 1 초안: ICC + Dice integrated comparison
4. Per-site Dice breakdown (NACC vs ADNI vs OASIS)

---

## 관련 문서

> - [#003 G/H ablation findings](003_ablation_g_h_findings.md) — G≈H first observation in ICC
> - [#004 Final ICC results](004_final_icc_results_update.md) — H ICC clean re-eval = 0.9318
