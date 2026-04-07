# ReproSeg Architecture Definition

**Version**: V1 (2026-04-01)

---

## Overview

```
INPUT [B, 1, 192, 224, 192]
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  SwinViT Patch Embedding + Stage 0                       │
│  → hs[0]: [B, 48, 96, 112, 96]                          │
└─────────┬───────────────────────────┬───────────────────┘
          │                           │
  ┌───────▼────────┐         ┌───────▼────────┐
  │ ANATOMY STREAM │         │ INVARIANCE     │
  │ (SwinUNETR)    │         │ STREAM (CNN)   │
  │ pretrained 62M │         │ random init 8M │
  │                │         │                │
  │ hs[0] 48ch ────┤    CSG0 ├──── inv[0]    │
  │ hs[1] 96ch ────┤    CSG1 ├──── inv[1]    │
  │ hs[2] 192ch ───┤    CSG2 ├──── inv[2]    │
  │ hs[3] 384ch ───┤    CSG3 ├──── inv[3] ──→ Site Classifier
  │ hs[4] 768ch    │         │       (GRL)     → site logits [B,4]
  └───────┬────────┘         └────────────────┘
          │
    [A_pure = A - σ(gate) ⊙ I]  ← Cross-Stream Gating
          │
  ┌───────▼────────┐     ┌──────────────────┐
  │ SEG DECODER    │     │ VOLUME HEAD      │
  │ (SwinUNETR)    │     │ LR-aware, 1.2M   │
  │ uses A_pure    │     │ from bottleneck   │
  │ as skip conn   │     └──────┬───────────┘
  └───────┬────────┘            │
          │                     │
  seg_logits [B,12,D,H,W]  vol_pred [B,7]
```

## Components

### 1. Shared Entry Point
- SwinViT의 patch embedding + stage 0을 공유
- 여기서 분기: anatomy stream은 SwinViT 계속 진행, invariance stream은 CNN 분기

### 2. Anatomy Stream (기존 BrainSegFounder)
- SwinUNETR encoder: depths=(2,2,2,2), heads=(3,6,12,24), feature_size=48
- UK-Biobank 41K pretrained weights 로딩
- 변경 없음 — encoder 자체는 수정하지 않음
- **skip connections만 CSG를 통과** (purified features)

### 3. Invariance Stream (새로 추가, ~5.6M params, half-channel design)
- Input: hs[0] (anatomy stage 0 output)
- 4-stage lightweight CNN: InvResBlock × 2 per stage
- InstanceNorm (BatchNorm이 아닌) — 배치 크기 작을 때 안정적
- 각 stage의 output이 anatomy stream의 대응 scale과 동일한 shape
- 학습 목표: scanner/site-specific variation 포착

### 4. Cross-Stream Gating (CSG, 4개, 새로 추가)
```python
gate_val = σ(Conv1x1(LayerNorm(Conv1x1([A; I]))))  # [0, 1]
A_pure = A - gate_val ⊙ I
```
- 각 encoder scale(0~3)에서 독립 적용
- gate가 1에 가까우면: I의 기여를 많이 빼줌 (scanner effect 강함)
- gate가 0에 가까우면: I 무시 (scanner effect 없음)
- **Gate entropy regularizer**: gate가 all-0 또는 all-1로 collapse 방지

### 5. Site Classifier (adversarial, 새로 추가)
- Input: inv_feats[3] (invariance stream 최종 output)
- GRL (Gradient Reversal Layer) 통과 후 site 예측
- 4-class: NACC / OASIS / ADNI / AJU
- GRL α는 epoch에 따라 0→1로 ramp up

### 6. Multi-Scale Feature Consistency (새로 추가)
- Paired scans (같은 피험자)의 A_pure features를 4개 scale에서 비교
- 각 scale: AdaptiveAvgPool3d(4) → Flatten → Linear → LayerNorm → 128-dim
- Cosine similarity 기반 consistency loss
- Scale별 가중치: (0.1, 0.2, 0.3, 0.4) — deeper scales에 더 높은 weight

### 7. Volume Head (VASNet에서 재사용)
- LRAwareVolumeHead: bottleneck [B,768,6,7,6] → L/R split → MLP → [B,7]
- Softplus activation (non-negative volume)
- 7 structures: hippocampus L/R, amygdala L/R, entorhinal L/R, ventricle

---

## Loss Function (v0.2 — Simplified, Uncertainty-Weighted)

기존 6-term + 수동 λ를 **4-term + 자동 balancing**으로 단순화.

**근거**:
- Isensee (MICCAI'24): loss 복잡도보다 training recipe가 중요
- Billot (MedIA'23): SynthSeg는 soft Dice만으로 brain parcellation SOTA
- Kendall (CVPR'18, 5000+ citations): 수동 λ 대신 자동 uncertainty weighting

```
L = (1/2σ₁²)·L_seg + (1/2σ₂²)·L_repro + (1/2σ₃²)·L_inv + (1/2σ₄²)·L_vol
    + log(σ₁·σ₂·σ₃·σ₄)    ← uncertainty regularizer
    + 0.01 × Σ gate_entropy ← CSG collapse prevention (minor)
```

### 4 Loss Terms

| Term | 구성 | 근거 |
|------|------|------|
| **L_seg** | Dice + CE | nnU-Net/SynthSeg 표준 (MICCAI'24) |
| **L_repro** | TCL(output) + Feature Consistency(4-scale) | TCL 검증됨(CV -8%). Feature consistency는 새 아키텍처 기여 |
| **L_inv** | CE(site_pred, site_gt) via GRL | Domain unlearning (PRIME'24). α ramp 0→1 over 15ep |
| **L_vol** | MSE(log1p) + 0.5×seg-vol consistency | VASNet 검증됨. Log-scale 안정적 |

### Uncertainty Weighting (수동 λ 제거)
- σ₁..σ₄는 학습 가능 파라미터 (초기값 σ=1.0)
- 학습 중 자동 task 난이도 기반 가중치 조절
- 수동 λ tuning 불필요 → 재현성 향상 + ablation 단순화

### 유일한 수동 hyperparameter
| Param | Value | 근거 |
|-------|-------|------|
| GRL α schedule | 0→1, 15 epochs | 표준 GRL practice |
| Gate entropy reg | 0.01 | CSG collapse 방지 |
| Grad clip | 1.0 | 기존과 동일 |

---

## Parameter Count Estimate

| Component | Params | Source |
|-----------|-------:|--------|
| Anatomy Stream (SwinUNETR) | 62.2M | BrainSegFounder pretrained |
| Invariance Stream (CNN) | 5.6M | Random init (half-channel) |
| Cross-Stream Gates (×4) | 0.7M | Random init |
| Site Classifier | 0.03M | Random init |
| Feature Consistency Projectors | 5.9M | Random init |
| Volume Head (LR-aware) | 1.2M | Random init |
| **TOTAL** | **75.7M** | |
| **New components only** | **13.5M** | **18% of total** |

---

## Ablation Plan

| Exp | Config | 변경점 | 비교 목적 |
|-----|--------|--------|----------|
| A | Baseline SwinUNETR | 없음 | 재현 기준선 |
| B | + TCL only | λ_tcl=1.0 | Loss-only reproducibility |
| C | + Invariance Stream + CSG | + inv + CSG | 아키텍처 기여 (site 제거) |
| D | + Adversarial | + GRL site classifier | adversarial 기여 |
| E | + Multi-Scale Feat Consistency | + L_feat | feature-level consistency 기여 |
| F | Full ReproSeg | 모든 component | 최종 모델 |

**성공 기준**:
- F의 CV < B의 CV (아키텍처가 loss-only보다 재현성 개선)
- F의 Dice ≥ A의 Dice (정확도 손실 없음)
- Ablation이 단조 개선: A < B < C < ... < F

---

## VRAM Estimate (B200, 183GB)

| Mode | VRAM |
|------|------|
| Single forward (batch=2) | ~45GB |
| Siamese forward (paired, batch=1×2) | ~55GB |
| Training (batch=2 + grad) | ~90GB |
| → 2-GPU DDP 가능 | ~45GB/GPU |

---

## Implementation Priority

1. `reproseg.py` — 모델 정의 (완료 ✅)
2. `train_reproseg.py` — 학습 스크립트 (다음)
3. `eval_reproseg.py` — 평가 스크립트
4. Ablation configs
