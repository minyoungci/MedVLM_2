---
date: 2026-04-07
project: brainclip
status: completed
---

# BrainCLIP: Cross-Modal Contrastive Learning for Brain MRI and Clinical Text

## 1. 배경 및 목표

3D structural brain MRI와 구조화된 임상 텍스트(나이, 진단, CDR, 교육년수, 인종 등)를
공유 embedding 공간에 contrastive learning으로 정렬하는 실험이다.

**핵심 가설**: 두 모달리티의 embedding을 정렬하면
(1) zero-shot cross-modal retrieval이 가능해지고,
(2) MRI 표현이 임상 의미를 학습하여 downstream 진단 분류에서 BrainIAC-only 대비 향상이 나타난다.

**참조 방법론**: NT-Xent (SimCLR, Chen et al. 2020), ConVIRT (Zhang et al. 2022), BioViL (Bannur et al. 2023)

---

## 2. 데이터 및 실험 설정

### 2.1 데이터

| Dataset | N (MRI+Clinical) | 임상 변수 | 비고 |
|---------|:----------------:|-----------|------|
| NACC    | 768              | CDR, GDS, age, sex, race, education, diagnosis | 100% 완전 |
| ADNI    | 704              | CDR-SB (50.7% 결측), diagnosis, age, scanner | diagnosis+age는 100% |
| AJU     | 719              | age, sex, diagnosis만 | 학습 제외, zero-shot eval 전용 |
| **학습 총계** | **1,472** |  | NACC+ADNI only |

- MRI 경로: `/home/vlm/data/preprocessed_v4/cross_sectional/{NACC,ADNI}/`
- MRI 포맷: shape (192, 224, 192), float32, RAS, z-normalized
- Test split: 213명 (전체의 ~14.5%)

### 2.2 텍스트 구성 전략

- **EXP01/03/04**: diagnosis 포함 (age/CDR/GDS/race/education + 진단명)
- **EXP02**: diagnosis 제외 (age/CDR/GDS만) — "MRI 독립 표현 학습" 검증용

NACC 예시 템플릿:
```
"{age}-year-old {sex} {race}, {educ} years education.
CDR global {cdr_global} ({severity}), CDR sum of boxes {cdr_sum}.
Memory domain: {memory}/3. Clinical diagnosis: {diagnosis}."
```

### 2.3 아키텍처

```
MRI branch:
  3D MRI [B, 1, 192, 224, 192]
  → BrainIAC encoder (frozen by default, 346MB, UK-Biobank pretrained)
  → [B, 768]
  → Projection head: Linear(768→256) + LayerNorm + GELU + Linear(256→128)
  → L2-normalized embedding [B, 128]

Text branch:
  Clinical text → PubMedBERT tokenizer + encoder (top-2 layers fine-tuned)
  → [CLS] token [B, 768]
  → Projection head: Linear(768→256) + LayerNorm + GELU + Linear(256→128)
  → L2-normalized embedding [B, 128]

Loss: NT-Xent (bidirectional), in-batch negatives
  τ (temperature): 0.07 learnable (EXP01/02/03), 0.02 fixed (EXP04)
  Batch size: 32 (EXP01/02/03), 64 (EXP04)
  Epochs: 30
```

**학습 파라미터 수** (EXP01 기준):
- MRI projection head: ~200K
- PubMedBERT top-2 layers + projection: ~12M
- Temperature: 1 scalar
- Total trainable: ~12.2M (BrainIAC frozen)

EXP03 (MRI fine-tune):
- BrainIAC top-2 blocks 추가 fine-tune (별도 lr=1e-5)
- Total trainable: ~29.4M / Total: ~199.7M

### 2.4 실험 구성

| EXP | 이름 | 변경사항 | Backbone | τ | Batch |
|-----|------|----------|----------|---|-------|
| 01  | baseline | 기준 (diagnosis 포함) | frozen | 0.07 learnable | 32 |
| 02  | no_diag_text | diagnosis 텍스트에서 제외 | frozen | 0.07 learnable | 32 |
| 03  | mri_finetune | BrainIAC top-2 blocks fine-tune | top-2 tuned | 0.07 learnable | 32 |
| 04  | fixed_temp | τ=0.02 고정, batch 64 | frozen | 0.02 fixed | 64 |

---

## 3. 결과

### 3.1 Cross-Modal Retrieval (Recall@K, test N=213)

| EXP | 방향 | R@1 | R@5 | R@10 |
|-----|------|:---:|:---:|:----:|
| Random baseline | — | 0.0047 | 0.0235 | 0.0469 |
| **EXP01** (w/ diag) | MRI→Text | **0.0235** | 0.0516 | 0.1033 |
| **EXP01** (w/ diag) | Text→MRI | **0.0282** | 0.0751 | 0.1315 |
| EXP02 (no diag) | MRI→Text | 0.0047 | 0.0329 | 0.0798 |
| EXP02 (no diag) | Text→MRI | 0.0141 | 0.0563 | 0.1033 |
| EXP03 (mri_finetune) | MRI→Text | 0.0141 | 0.0329 | 0.0845 |
| EXP03 (mri_finetune) | Text→MRI | 0.0047 | 0.0376 | 0.0986 |
| EXP04 (fixed_temp) | MRI→Text | 0.0047 | 0.0376 | 0.0657 |
| EXP04 (fixed_temp) | Text→MRI | **0.0282** | 0.0704 | 0.0939 |

### 3.2 Linear Probe Classification (CN/MCI/AD, test N=213)

| EXP | Mode | Accuracy | BAcc | AUROC | vs BrainIAC-only (ΔBAcc) |
|-----|------|:--------:|:----:|:-----:|:------------------------:|
| BrainIAC-only | backbone | 0.4789 | 0.3551 | 0.5681 | — (baseline) |
| EXP01 (w/ diag) | brainclip | 0.5728 | 0.3961 | 0.6234 | **+0.0410** |
| EXP02 (no diag) | brainclip | 0.5305 | 0.3777 | 0.5889 | +0.0226 |
| EXP03 (mri_finetune) | backbone | 0.4413 | 0.3312 | 0.5577 | −0.0239 vs frozen |
| EXP03 (mri_finetune) | brainclip | 0.5681 | 0.3884 | 0.6195 | +0.0572 vs EXP03 backbone |
| EXP04 (fixed_temp) | brainclip | 0.5681 | **0.4257** | 0.6084 | **+0.0706** vs EXP04 backbone |
| EXP04 (fixed_temp) | backbone | 0.4789 | 0.3551 | 0.5681 | — (EXP01 backbone와 동일) |

> 주의: EXP03 backbone은 MRI fine-tune으로 인해 BrainIAC pretrained 표현이 일부 망가짐
> (BAcc 0.3312 < frozen backbone 0.3551). BrainCLIP projection은 이를 보정함.

### 3.3 학습 곡선 요약 (Last Epoch, epoch 29)

| EXP | Train Loss | Val Loss | Val R@1 (in-batch) | τ (final) |
|-----|:----------:|:--------:|:------------------:|:---------:|
| EXP01 (w/ diag) | 4.6105 | 4.5796 | 0.0078 | 0.0713 |
| EXP02 (no diag) | 4.7536 | 4.5252 | 0.0304 | 0.0713 |
| EXP03 (mri_finetune) | 3.9785 | 4.0999 | 0.0156 | 0.0709 |
| EXP04 (fixed_temp) | 4.7274 | 4.5469 | 0.0280 | 0.0200 |

---

## 4. 핵심 발견 / 해석

### 4.1 Best experiment

**Retrieval 기준**: EXP01 (w/ diagnosis)
- MRI→Text R@1=0.0235 (+5.0× vs random), Text→MRI R@1=0.0282 (+6.0× vs random)
- 성공 기준 R@1 > 5% (0.05)에는 미달. 그러나 random 대비 5~6배 향상은 실질적 alignment가 일어났음을 시사.

**Linear probe BAcc 기준**: EXP04 (fixed_temp, τ=0.02)
- BAcc 0.4257 — BrainIAC-only 0.3551 대비 **+0.0706** 향상
- EXP01과 동일 구조(frozen backbone, diagnosis 포함)에서 τ=0.02 고정 + batch 64가 표현 품질을 더 높임

### 4.2 진단 텍스트의 영향 (EXP01 vs EXP02)

- Retrieval에서 diagnosis 포함(EXP01) vs 제외(EXP02): MRI→Text R@1 0.0235 vs 0.0047
  - diagnosis label이 retrieval의 주요 신호임이 확인됨
- Linear probe에서는 EXP01 (BAcc 0.3961) > EXP02 (0.3777) — 차이 0.018
  - EXP02도 BrainIAC-only(0.3551) 대비 유의미하게 향상 → diagnosis 정보 없이도 CDR/GDS/age 신호만으로 MRI 표현이 임상적 의미를 학습

### 4.3 MRI Fine-tuning의 역효과 (EXP03)

- EXP03 backbone BAcc 0.3312 < frozen backbone 0.3551: BrainIAC pretrained weights를 변경하면 UK-Biobank로 학습된 일반 표현이 손상됨
- BrainCLIP projection이 이를 부분적으로 보정(BAcc 0.3884)하지만, EXP01 brainclip(0.3961)에 미달
- Retrieval도 EXP03이 EXP01보다 낮음 — 1,472 pairs로 MRI backbone fine-tune은 과적합 위험이 큼

### 4.4 Temperature의 역할 (EXP04)

- EXP01~03 모두 학습 중 τ가 0.07→0.071로 거의 변하지 않음: learnable temperature가 실질적으로 작동하지 않은 것
- EXP04에서 τ=0.02 고정: linear probe BAcc +0.0706으로 가장 높음, 그러나 MRI→Text R@1은 0.0047로 감소
- 해석: 낮은 τ는 positive/negative 구분을 강화하지만, 소규모 데이터에서 false negative를 크게 처벌해 retrieval 다양성이 감소할 수 있음

---

## 5. 한계 및 다음 방향

### 5.1 한계

1. **1,472 pairs 소규모**: contrastive learning의 in-batch negatives 효과가 제한됨 (batch_size=32 → 31 negatives).
   collapse는 발생하지 않았으나 alignment signal이 약함. 성공 기준 R@1 > 5% 미달.

2. **ADNI CDR-SB 50.7% 결측**: ADNI 데이터 704명 중 절반 이상에서 CDR-SB 없음.
   대체 텍스트("CDR-SB not available")가 텍스트 다양성을 낮추어 alignment 신호를 희석.

3. **AJU zero-shot 미검증**: 한국인 환자 719명 데이터에 대한 zero-shot 평가 미실시.
   서양인(NACC/ADNI) 학습 모델의 cross-ethnic generalization 여부 불명확.

4. **Temperature 고착 (EXP01-03)**: learnable τ가 초기값 0.07에서 거의 변하지 않음.
   contrastive learning이 온전히 작동하지 않았을 가능성 있음.

5. **텍스트 다양성 제한**: 템플릿 기반 텍스트 생성 → 유사한 임상 프로필 간 텍스트 유사도가 높아
   hard negative가 사실상 false negative로 작동할 수 있음.

6. **NACCMMSE 99.5% 결측**: MMSE 지표는 텍스트에서 제외됨.

### 5.2 다음 방향

| 우선순위 | 실험 | 설명 |
|:--------:|------|------|
| High | EXP05: AJU zero-shot | NACC+ADNI 학습 모델로 AJU 719명 retrieval 평가. cross-ethnic 일반화 확인 |
| High | 데이터 확대 | AJU 외 추가 데이터셋 발굴 또는 augmentation (stochastic phrasing, MRI augmentation) |
| Medium | EXP06: proj dim 확대 | projection 128→256 + hidden 512. 표현 용량 증가 효과 검증 |
| Medium | Hard negative mining | 같은 진단 내에서 CDR/GDS 차이가 큰 샘플을 explicit negative로 활용 |
| Low | Multi-modal conditioning | 진단 예측 시 text embedding을 conditioning으로 활용 (cross-attention) |

---

## 참조

- SimCLR: Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations.
- ConVIRT: Zhang et al. (2022). Contrastive Learning of Medical Visual Representations from Paired Images and Text.
- BioViL: Bannur et al. (2023). Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing.
- BrainIAC: [pretrained 3D brain encoder, UK-Biobank] [VERIFY: 논문 미공개 가능성]
- PubMedBERT: Gu et al. (2021). Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing.

---

_결과 파일 경로:_
- `/home/vlm/minyoung2/experiments/BrainCLIP/results/BrainCLIP_Report.md`
- `/home/vlm/minyoung2/experiments/BrainCLIP/results/exp{01..04}_*/logs/{retrieval_test,linear_probe_results}.txt`
