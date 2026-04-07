# ReproSeg: Reproducibility-Equivariant Dual-Stream Brain Segmentation

**Version**: V1
**Status**: Smoke test 완료 → Ablation 실행 준비
**Target**: NeurIPS 2026 / Medical Image Analysis

---

## 가설
MNI 공간 뇌 segmentation의 test-retest 재현성은 loss function이 아닌 **아키텍처 수준에서 scanner-invariance를 구조적으로 보장**함으로써 개선할 수 있다. ReproSeg는 dual-stream 설계로 anatomy 신호와 scanner noise를 분리하여, 재현성을 학습 목표가 아닌 구조적 속성으로 만든다.

## 데이터
- **Primary**: V4 native space (`/home/vlm/data/preprocessed_v4/`)
- **Cross-sectional**: 2,397명 (manifest), train 1,724 / val 325 / test 348
- **Longitudinal**: 6,554 sessions (paired scans for TCL)
- **Sites**: NACC / OASIS / ADNI / AJU (4-class adversarial)
- **Shape**: 192×224×192, 1mm iso, RAS, 12-class segmentation

## 아키텍처
→ `ARCHITECTURE.md` 상세 참조

**핵심 구성**:
- Anatomy Stream: SwinUNETR (BrainSegFounder, 62M, pretrained)
- Invariance Stream: Lightweight CNN (5.6M, random init)
- Cross-Stream Gating ×4: anatomy에서 scanner noise 구조적 제거
- Site Classifier + GRL: adversarial scanner invariance
- Multi-Scale Feature Consistency: 4-scale paired-scan 일관성
- Volume Head: LR-aware 7-structure volume 예측

**Total**: 75.7M params (새 component 13.5M = 18%)

## 실험 목록 (Ablation A-F)

| EXP | Name | 활성 component | 상태 | Dice | CV |
|-----|------|---------------|------|:----:|:--:|
| A | `reproseg_v1_A_baseline` | Backbone + Decoder | 🔲 | — | — |
| B | `reproseg_v1_B_tcl` | + TCL + FeatCons | 🔲 | — | — |
| C | `reproseg_v1_C_dualstream` | + InvStream + CSG + GRL | 🔲 | — | — |
| D | `reproseg_v1_D_volume` | + Volume Head | 🔲 | — | — |
| E | `reproseg_v1_E_repro_inv` | C + B | 🔲 | — | — |
| **F** | **`reproseg_v1_F_full`** | **All** | 🔲 | — | — |

## 성공 기준
- [ ] F의 Dice ≥ A의 Dice (정확도 손실 없음)
- [ ] F의 CV < B의 CV (아키텍처가 loss-only보다 재현성 개선)
- [ ] Ablation 단조 개선: A → B → ... → F
- [ ] F의 Dice ≥ 0.920 (nnU-Net 수준)

## 위험 요소
- Smoothing 반론: "CSG가 단순히 smoothing하여 CV 감소" → Hausdorff distance로 방어
- Inv stream이 anatomy 포착: capacity 제한 (half channels)으로 완화
- GRL 학습 불안정: α ramp (0→1, 15ep)으로 완화

## 산출물 네이밍 규칙

### 결과 파일
```
results/reproseg_v1_{A-F}_{descriptor}/
  ├── checkpoints/best.pt
  └── logs/training_log.json
```

### Figure 파일
```
figures/
  ├── architecture/reproseg_v1_architecture.{png,pdf}
  ├── ablation/reproseg_v1_ablation_{dice,cv}.{png,pdf}
  ├── reproducibility/reproseg_v1_{cv_comparison,bland_altman}.{png,pdf}
  ├── per_class/reproseg_v1_per_class_dice.{png,pdf}
  └── qualitative/reproseg_v1_seg_overlay_{subject}.{png,pdf}
```

### 분석 파일
```
analysis/
  ├── cv_comparison/reproseg_v1_cv_summary.csv
  ├── metric_disagreement/reproseg_v1_metric_matrix.csv
  └── site_invariance/reproseg_v1_site_accuracy.csv
```

## 타겟 저널
- Primary: NeurIPS 2026 (마감 2026-05-06)
- Alternative: Medical Image Analysis / Nature Communications
