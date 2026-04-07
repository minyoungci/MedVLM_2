# Research Note #001: ReproSeg V1 실험 계획 및 각 Ablation 상세 기록

**작성일**: 2026-04-02
**작성자**: VLM Team / minyoung
**카테고리**: project / reproseg
**상태**: 진행 중

---

## 배경

ReproSeg V1은 뇌 MRI test-retest 재현성을 loss function이 아닌 **아키텍처 수준**에서 보장하는 dual-stream segmentation 모델이다.
BrainSegFounder(SwinUNETR, 62M)를 backbone으로 삼고, scanner-invariance를 위한 경량 컴포넌트를 추가한다.

### 기존 문제 및 수정 이력

| 날짜 | 버그 | 수정 |
|------|------|------|
| 2026-04-01 | Scheduler 5 param_groups vs 6 (UWL 후 추가) | UWL을 optimizer 생성 전에 추가 |
| 2026-04-01 | UWL log_var 미클램핑 → 음수 loss | `clamp(-6, 6)` 적용 |
| 2026-04-01 | GRL alpha dead code (계산만 하고 전달 안 됨) | `model.forward(grl_alpha=...)` 파라미터 추가 |
| 2026-04-01 | val Dice DDP all_reduce 없음 → 50% 데이터만 반영 | `torch.distributed.all_reduce` 추가 |
| 2026-04-01 | Gate entropy = mean의 entropy (per-voxel 아님) | per-element entropy mean으로 수정 |
| 2026-04-01 | Checkpoint에 scheduler/UWL state 미저장 | 두 항목 추가 |
| 2026-04-01 | manifest ↔ progression subject_id 불일치 → pairs=0 | longitudinal 독립 80/20 split |
| 2026-04-01 | DDP double-ready (feat_consistency 외부 호출) | model.forward 내부로 이동 |

---

## 전체 실험 계획

### Phase 1 — V1 Ablation (A~F)

목적: 각 컴포넌트의 Dice 및 재현성(CV) 기여 측정

| EXP | 이름 | 활성 컴포넌트 | 상태 | Best Dice | 비고 |
|-----|------|-------------|------|-----------|------|
| **A** | `reproseg_v1_A_baseline`   | Backbone+Decoder | ✅ 완료 | **0.9178** | ep21에서 peak |
| **B** | `reproseg_v1_B_tcl`        | +TCL+FeatCons    | 🔄 ep4 | 0.909 | pairs=1,080개 |
| **C** | `reproseg_v1_C_dualstream` | +InvStream+CSG+GRL | 🔄 ep5 | 0.907 | inv_w=0.80 |
| **D** | `reproseg_v1_D_volume`     | +VolumeHead      | 🔄 ep4 | 0.908 | vol_w=1.08 |
| **E** | `reproseg_v1_E_repro_inv`  | C+B              | ⏳ 대기 | — | GPU 확보 후 |
| **F** | `reproseg_v1_F_full`       | All              | ⏳ 대기 | — | GPU 확보 후 |

### Phase 2 — V2 방향 탐색 (파라미터 효율화)

Phase 1 완료 후 결과에 따라 선택적 진행.

| EXP | 이름 | 핵심 아이디어 | 예상 Total Params | 우선순위 |
|-----|------|-------------|-----------------|---------|
| **V2-A** | `reproseg_v2_A_feature_reuse`    | inv_stream 제거 → sg(hs[0]) 재사용 | ~63M | ⭐⭐⭐ |
| **V2-B** | `reproseg_v2_B_frozen_lora`      | Backbone freeze + LoRA adapters    | 64M (trainable 2M) | ⭐⭐⭐ |
| **V2-C** | `reproseg_v2_C_hypernet_csg`     | per-voxel gate → global hypernetwork CSG | ~62.5M | ⭐⭐ |

---

## Ablation별 상세 기록

---

### Ablation A: Baseline

**실험 이름**: `reproseg_v1_A_baseline`
**Mode**: `baseline`
**GPU**: 1, 7 → B200 × 2

#### 아키텍처

```
INPUT [B,1,192,224,192]
    │
[SwinUNETR Encoder] ─────────────── hs[0~4]
    │                                   │
[Cross-Stream Gating × 4]          (gate = 0, CSG bypass)
    │
[Seg Decoder] → [B, 12, 192,224,192]
```

CSG는 초기화 이후 baseline mode에서 사용되지 않음.
Volume Head, Site Classifier, Invariance Stream → forward는 하지만 loss 기여 없음.

#### 파라미터

| 컴포넌트 | 파라미터 수 | Trainable | Gradient |
|---------|-----------|-----------|---------|
| SwinUNETR backbone | 62,187,198 | ✅ | Dice+CE만 |
| Invariance Stream | 5,642,496 | ✅ | **없음** (loss inactive) |
| CSG × 4 | 687,600 | ✅ | **없음** |
| Site Classifier | 25,476 | ✅ | **없음** |
| Feat Consistency | 5,899,776 | ✅ | **없음** |
| UWL (log_vars × 4) | 4 | ✅ | seg만 active |
| **합계** | **74,442,550** | 74,442,550 | — |

> 실효 trainable (gradient 흐르는 파라미터): **62,187,198** (SwinUNETR만)

#### Gradient 흐름

```
loss_seg (DiceCE)
    │
    ▼
[SwinUNETR Encoder + Decoder]  ✅ gradient O
[UWL.log_vars[0] (seg)]        ✅ gradient O
[Inv Stream]                   ❌ gradient X (loss=0, active_mask=False)
[CSG]                          ❌ gradient X
[Site Classifier]              ❌ gradient X
[Feat Consistency]             ❌ gradient X
```

#### 학습 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| LR (backbone) | 1e-4 |
| LR schedule | warmup(3ep) + cosine decay |
| Batch size | 1/GPU × 2 GPU = 2 |
| Epochs | 30 |
| Optimizer | AdamW (wd=0.01) |
| Mixed precision | bf16 |
| Grad clip | 1.0 |

#### 소요 시간

| 구간 | 소요 시간 |
|------|----------|
| ep0 (warmup, validation 포함) | ~24분 |
| ep1~29 (평균) | ~9.8분/ep |
| **전체 30ep** | **~26시간** |

#### 결과

| Epoch | Val Dice | Train Loss | Seg Weight |
|-------|----------|------------|-----------|
| 0 | 0.5402 | 1.195 | 0.995 |
| 1 | 0.8844 | 0.477 | 1.030 |
| 5 | 0.9052 | -0.235 | 1.553 |
| 10 | 0.9131 | -0.590 | 2.389 |
| 15 | 0.9163 | -0.837 | 3.311 |
| **21** | **0.9178** | -1.014 | 4.133 |
| 29 | 0.9173 | -1.091 | 4.434 |

**Best Dice: 0.9178 (ep21)**

> Train loss 음수 이유: UWL 공식 `exp(-s)*L + s`에서 seg loss가 0.1 수준으로 작아지면 log_var 최적점이 음수가 됨. 학습은 정상 (val_dice 상승). C1 fix로 clamp(-6,6) 적용되어 발산 없음.

---

### Ablation B: +TCL + Multi-Scale Feature Consistency

**실험 이름**: `reproseg_v1_B_tcl`
**Mode**: `repro`
**GPU**: 0, 1 → B200 × 2

#### 아키텍처

```
INPUT [B,1,192,224,192]
    │
[SwinUNETR Encoder]
    │
[CSG × 4]  (gate 학습 안 함 — inv stream 없음)
    │
[Seg Decoder] → seg_logits

paired scans (i1, i2) 존재 시:
    ├── TCL: vol(seg_logits[i1]) vs vol(seg_logits[i2])
    └── FeatCons: cosine_sim(a_pure[i1], a_pure[i2]) @ 4 scales
         ← model.forward 내부에서 계산 (DDP double-ready 방지)
```

**Note**: Invariance Stream은 forward 실행되지만 loss 기여 없음.

#### 파라미터

| 컴포넌트 | 파라미터 수 | Gradient |
|---------|-----------|---------|
| SwinUNETR backbone | 62,187,198 | ✅ (Dice+TCL+FeatCons) |
| Invariance Stream | 5,642,496 | ❌ |
| CSG × 4 | 687,600 | ✅ (FeatCons via a_pure) |
| Feat Consistency projectors | 5,899,776 | ✅ (FeatCons loss) |
| Site Classifier | 25,476 | ❌ |
| UWL (log_vars) | 4 | ✅ (seg, repro active) |
| **합계** | **74,442,550** | — |

> 실효 trainable: **68,774,574** (backbone + CSG + FeatCons)

#### Gradient 흐름

```
loss_seg (DiceCE)         → SwinUNETR Encoder+Decoder, CSG
loss_repro = TCL + FeatCons
  ├── TCL                 → SwinUNETR Decoder (seg_logits 통해)
  └── FeatCons            → SwinUNETR Encoder (a_pure 통해), CSG, FeatCons projectors
UWL.log_vars[0,1]        → active
```

#### 학습 데이터 (A와 차이)

- Cross-sectional: 1,917개 (동일)
- **Longitudinal pairs: 1,080쌍** (A는 0쌍)
  - progression_df 독립 80/20 split으로 생성
  - pairs subject_id는 manifest subject_id와 완전 disjoint → 누수 없음

#### 소요 시간 (추정)

| 구간 | 소요 시간 |
|------|----------|
| ep0 | ~25분 |
| ep1~29 (평균) | ~11~13분/ep (paired forward × 2) |
| **전체 30ep** | **~30~32시간** |

#### 현재 결과 (진행 중)

| Epoch | Val Dice | Repro Weight |
|-------|----------|-------------|
| 0 | 0.5404 | 0.426 |
| 1 | 0.8798 | 0.476 |
| 4 | **0.9091** | **0.507** |

Repro weight 0.5 수준 → TCL+FeatCons 정상 작동 중.

---

### Ablation C: +InvStream + CSG + GRL

**실험 이름**: `reproseg_v1_C_dualstream`
**Mode**: `inv`
**GPU**: 2, 4 → B200 × 2

#### 아키텍처

```
INPUT [B,1,192,224,192]
    │
[SwinUNETR Encoder] ────── hs[0~4]
    │                            │
    │                    [Invariance Stream]
    │                    (lightweight CNN, random init)
    │                    inv[0~3]: [B,24/48/96/192, ...]
    │                            │
    └──── [CSG × 4] ────────────┘
           A_pure[i] = A[i] - σ(gate[i]) · proj(inv[i])
               │
    [Seg Decoder] → seg_logits
               │
    [Site Classifier] ← GRL(inv[3], alpha=grl_alpha)
          loss_inv = CE(site_logits, site_id)
```

**GRL alpha schedule**: `grl_alpha = min(1.0, epoch / 15.0)`
- ep0: α=0.0 (gradient reversal 없음)
- ep7: α=0.47
- ep15+: α=1.0 (full reversal)

#### 파라미터

| 컴포넌트 | 파라미터 수 | Gradient |
|---------|-----------|---------|
| SwinUNETR backbone | 62,187,198 | ✅ (Dice + GRL back-prop) |
| **Invariance Stream** | **5,642,496** | ✅ (GRL adversarial) |
| **CSG × 4** | **687,600** | ✅ (a_pure 통해 Dice) |
| Site Classifier | 25,476 | ✅ (site CE loss) |
| Feat Consistency | 5,899,776 | ❌ (pairs 없음) |
| Gate entropy regularizer | — | ✅ (CSG gate values) |
| UWL (log_vars) | 4 | ✅ (seg, inv active) |
| **합계** | **74,442,550** | — |

> 실효 trainable: **68,542,770** (backbone + inv + CSG + site_cls)

#### Gradient 흐름 (핵심)

```
loss_seg  → backbone encoder/decoder, CSG (a_pure 통해)
loss_inv  → site_classifier ← GRL ← inv_stream
             (GRL이 gradient를 반전시켜 inv_stream이 site를 잘 예측하도록 학습)
             동시에 backbone은 GRL로 인해 site 정보를 지우도록 학습

gate_entropy → CSG gate (gate collapse 방지 regularizer)
```

**중요**: GRL의 gradient 방향
- Site Classifier → (정방향) → inv_stream: inv_stream이 site를 잘 포착
- Site Classifier → (역방향 via GRL) → backbone: backbone이 site를 지움

이 두 힘이 동시에 작용 → anatomy/scanner 분리 강제.

#### 현재 결과 (진행 중)

| Epoch | Val Dice | Inv Weight | GRL alpha |
|-------|----------|-----------|-----------|
| 0 | 0.5412 | 0.691 | 0.000 |
| 3 | 0.9003 | 0.777 | 0.200 |
| 5 | **0.9066** | **0.799** | **0.333** |

Inv weight 0.8 → adversarial loss 정상 기여 중.

---

### Ablation D: +Volume Head

**실험 이름**: `reproseg_v1_D_volume`
**Mode**: `vol`
**GPU**: 6, 7 → B200 × 2

#### 아키텍처

```
[SwinUNETR Encoder] → bottleneck [B,768,6,7,6]
    │
[Seg Decoder] → seg_logits → seg_vol (stop-gradient)
    │
[LRAwareVolumeHead]
    in: bottleneck [B,768,6,7,6]
    out: vol_pred [B,7]  (7 structures)
    LR-aware: flip이면 L/R label swap

Structures: hipp_L/R, amyg_L/R, ento_L/R, ventricle
GT vol: seg label에서 on-the-fly 계산 (no pre-computed JSON)

loss_vol = MSE(log1p(vol_pred), log1p(gt_vol))
         + 0.5 * MSE(vol_pred, log1p(seg_vol))
```

#### 파라미터

| 컴포넌트 | 파라미터 수 | Gradient |
|---------|-----------|---------|
| SwinUNETR backbone | 62,187,198 | ✅ (Dice + Vol) |
| **LRAwareVolumeHead** | added | ✅ |
| Invariance Stream | 5,642,496 | ❌ |
| CSG × 4 | 687,600 | ❌ (vol mode에서 CSG는 통과하지만 inv 없음) |
| Site Classifier | 25,476 | ❌ |
| Feat Consistency | 5,899,776 | ❌ |
| UWL (log_vars) | 4 | ✅ (seg, vol active) |

> VolumeHead 파라미터: `in_ch=768 → FC(768→256) → FC(256→7)` = 약 197K

#### Gradient 흐름

```
loss_seg  → backbone encoder/decoder
loss_vol  → volume_head, backbone encoder (bottleneck 통해)
seg_vol   → stop-gradient (seg에서 계산되지만 vol loss에 gradient 안 흐름)
```

#### 현재 결과 (진행 중)

| Epoch | Val Dice | Vol Weight |
|-------|----------|-----------|
| 0 | 0.5519 | 0.868 |
| 2 | 0.8978 | 0.998 |
| 4 | **0.9077** | **1.080** |

Vol weight 1.08 → volume loss가 seg보다 약간 더 강하게 가중됨 (초기 정상 범위).

---

### Ablation E: +Repro + Inv (B+C)

**실험 이름**: `reproseg_v1_E_repro_inv`
**Mode**: `repro` + `inv` 동시 활성 (별도 mode 없음 → 코드 수정 필요)
**GPU**: 미정 (B, C 완료 후)
**상태**: ⏳ 대기

#### 아키텍처 (예정)

```
- Invariance Stream: C와 동일 (GRL 포함)
- TCL + FeatCons: B와 동일 (1,080 pairs)
- Volume Head: 없음
```

#### 파라미터 (예정)

실효 trainable: **74,417,070** (backbone + inv + CSG + site_cls + feat_cons)

---

### Ablation F: Full

**실험 이름**: `reproseg_v1_F_full`
**Mode**: `full`
**GPU**: 미정 (E 완료 후 또는 병렬)

#### 아키텍처

모든 컴포넌트 활성:
```
Backbone + Decoder     ← Dice+CE
Invariance Stream      ← GRL adversarial
CSG × 4               ← a_pure 생성
Site Classifier        ← site CE (GRL)
FeatCons              ← cosine @ 4 scales (paired)
TCL                   ← volume consistency (paired)
Volume Head           ← MSE vol regression
UWL × 4              ← auto weight balancing
Gate Entropy          ← CSG collapse 방지
```

#### Gradient 흐름 (전체)

```
loss_seg   → backbone + CSG (via a_pure)
loss_repro → backbone decoder (TCL) + backbone encoder + CSG + FeatCons (FeatCons)
loss_inv   → inv_stream (GRL→양방향) + backbone (GRL→scanner 제거)
loss_vol   → volume_head + backbone (bottleneck)
gate_ent   → CSG gate values
```

#### 성공 기준

- [ ] Dice ≥ 0.920
- [ ] Dice ≥ A (0.9178)
- [ ] CV < B (TCL 없이 reproducibility 달성)
- [ ] 단조 개선: A → B → ... → F

---

## Phase 2 — V2 설계 방향

### V2-A: Feature Reuse (Inv Stream 제거)

**핵심 아이디어**: 별도 Invariance Stream 없이, SwinUNETR의 **hs[0]**을 scanner signal로 사용.

```python
# 현재 V1:
inv_feats = self.inv_stream(hs[0])  # 5.6M 추가 stream

# V2-A:
inv_feats = [sg(hs[0])] * 4        # stop-gradient low-level features
# 또는 단일 스케일에서 4스케일로 stride pooling
```

**파라미터 절감**: -11.5M (inv_stream 5.6M + feat_cons 5.9M 제거 가능)
**총 파라미터**: ~63M

**논문 claim**: "Low-level SwinUNETR features 자체가 scanner signal을 충분히 인코딩한다. 별도 encoder 불필요."

**검증 방법**: V1 C와 V2-A의 Dice/CV 비교.

---

### V2-B: Frozen Backbone + LoRA

**핵심 아이디어**: SwinUNETR 완전 동결 (0 gradient). Attention layer에만 LoRA 추가.

```python
# LoRA rank=8 적용 위치: SwinViT의 각 attention Q,K,V projection
# 추가 파라미터: ~400K (rank 8, 12 attention layers)
# backbone frozen: 62.2M → 0 gradient

# 결과:
# Trainable params: LoRA(400K) + CSG(687K) + Inv(필요시) + FeatCons(5.9M) ≈ 7M
# Total params: 62M + 7M = 69M (하지만 trainable은 7M)
```

**논문 claim**: "Scannar invariance를 7M trainable parameter로 달성 (기존 fine-tuning 대비 10× 효율적)."

---

### V2-C: Hypernetwork CSG

**핵심 아이디어**: per-voxel gate → global channel-wise scale.

```python
# 현재 CSG: gate [B, C, D, H, W] — 공간적으로 vary
# V2-C: 
global_stats = AdaptiveAvgPool3d(1)(hs[3])  # [B, 384, 1,1,1]
gate_scale = HyperNet(global_stats)          # [B, C]  channel만
A_pure = A * gate_scale.unsqueeze(-1)...    # broadcast
```

**근거**: Scanner effect는 contrast/intensity 등 global shift. Per-voxel CSG가 과도한 표현.
**절감**: CSG 687K → 약 100K (gate FC만)

---

## 비교 기준 (논문 Table 예정)

| 모델 | Params | Dice | CV (예정) | 비고 |
|------|--------|------|-----------|------|
| BrainSegFounder | 62M | ~0.91 | — | 기존 SOTA |
| SDNet (Chartsias 2019) | ~8M | ~0.87 | — | cardiac, 직접 비교 어려움 |
| **ReproSeg V1-A** | 74M | **0.9178** | — | baseline |
| **ReproSeg V1-F** | 74M | TBD | TBD | full model |
| **ReproSeg V2-A** | ~63M | TBD | TBD | feature reuse |
| **ReproSeg V2-B** | 69M (7M trainable) | TBD | TBD | LoRA |

---

## 결론 및 다음 단계

### 즉시 할 일

1. B/C/D 완료 대기 (각 ~30시간, 순차 결과 확인)
2. B vs A: TCL+FeatCons의 Dice 영향 확인
3. C vs A: Dual-stream의 순수 segmentation 영향 확인 (감소? 유지?)
4. D vs A: Volume head의 Dice 영향 확인
5. B/C/D 결과 보고 E, F 설계

### 중기 할 일 (Phase 2 트리거 조건)

- C의 Dice ≤ A (inv stream이 segmentation을 해치는 경우) → V2-A 우선 진행
- F vs A Dice 개선 < 0.005 → V2 방향 재검토
- CV 측정을 위한 test-retest evaluation 스크립트 작성 필요

---

## 관련 문서

> 관련:
> - [README.md](../README.md) — 실험 목록 및 성공 기준
> - [SCRATCHPAD.md](../SCRATCHPAD.md) — 현재 실험 상태 및 결과 추적
> - [ARCHITECTURE.md](../ARCHITECTURE.md) — 아키텍처 버전 이력
> - [reproseg.py](../scripts/reproseg.py) — 모델 아키텍처 코드
> - [train_reproseg.py](../scripts/train_reproseg.py) — 학습 루프 코드
