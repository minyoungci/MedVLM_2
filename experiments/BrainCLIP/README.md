# BrainCLIP — 3D Brain MRI × Clinical Text Contrastive Alignment

**Version**: V1 (2026-04-06)
**Status**: 🟡 Setup

---

## Hypothesis

3D structural brain MRI and structured clinical text (age, diagnosis, CDR, education, race)
can be aligned in a shared embedding space via contrastive learning.
This alignment enables zero-shot clinical profile retrieval from MRI and vice versa,
and produces richer representations for downstream diagnosis tasks.

---

## Data

| Dataset | N (MRI+Clinical) | Clinical Variables | Notes |
|---------|------------------|--------------------|-------|
| NACC | 768 | CDR, GDS, age, sex, race, education, diagnosis | 100% complete |
| ADNI | 704 | CDR-SB (50.7%), diagnosis, age, scanner | Full diagnosis coverage |
| AJU | 719 | age, sex, diagnosis only | Excluded from training (no rich text) |
| **Total** | **1,472** | | NACC+ADNI only |

- **MRI**: `/home/vlm/data/preprocessed_v4/cross_sectional/{NACC,ADNI}/`
- **NACC clinical**: `/home/vlm/data/raw/NACC/NACC-Clinical/commercial_nacc70.csv`
- **ADNI clinical**: `/home/vlm/data/raw/ADNI/adni34_t1w_with_clinical.csv`
- **Manifest**: `/home/vlm/data/metadata/v4_manifest.csv`
- **MRI shape**: 192×224×192, float32, RAS, z-normalized

---

## Architecture

```
3D MRI [B, 1, 192, 224, 192]
  → BrainIAC encoder (frozen, 346MB)   # /home/vlm/minyoung/pretrain/brainiac/BrainIAC.ckpt
  → [B, 768]
  → MRI projection head (Linear 768→256 + LayerNorm + GELU + Linear 256→128)
  → L2-normalized embedding [B, 128]

Clinical Text  e.g. "72-year-old White female, 16 years education, CDR 1.0 (moderate)..."
  → PubMedBERT tokenizer + encoder (top-2 layers fine-tuned)
  → [CLS] token [B, 768]
  → Text projection head (Linear 768→256 + LayerNorm + GELU + Linear 256→128)
  → L2-normalized embedding [B, 128]

Loss: NT-Xent (SimCLR) bidirectional contrastive
  τ (temperature): 0.07 (learnable)
  In-batch negatives: batch_size=32 → 31 negatives per sample
```

**Trainable parameters**:
- MRI projection head: ~200K
- PubMedBERT top-2 layers + projection: ~12M
- Temperature: 1 scalar
- **Total trainable: ~12.2M** (BrainIAC frozen)

---

## Experiment Plan

| EXP | Name | Description | Status |
|-----|------|-------------|--------|
| 01 | `brainclip_v1_baseline` | BrainIAC frozen + PubMedBERT top-2 fine-tune | 🔲 |
| 02 | `brainclip_v1_mri_finetune` | BrainIAC top-2 layers fine-tuned | 🔲 |
| 03 | `brainclip_v1_nacc_only` | NACC only (ablation: dataset effect) | 🔲 |
| 04 | `brainclip_v1_aju_zeroshot` | Zero-shot eval on AJU (cross-ethnic) | 🔲 |

---

## Evaluation Tasks

1. **Cross-modal retrieval** (primary)
   - MRI → Text: given MRI, retrieve correct clinical profile (Recall@1, @5, @10)
   - Text → MRI: given clinical text, retrieve correct MRI (Recall@1, @5, @10)
   - Baseline: random = 1/1472 = 0.07%

2. **Linear probe classification** (secondary)
   - Freeze both encoders after CLIP training
   - Train linear classifier on BrainCLIP embeddings: CN/MCI/AD
   - Compare: BrainCLIP vs MRI-only BrainIAC embeddings

3. **Zero-shot cross-ethnic** (exploratory)
   - Train on NACC+ADNI → zero-shot retrieve on AJU
   - Does Western-trained alignment generalize to Korean patients?

---

## Success Criteria

- [ ] Recall@1 (MRI→Text) > 5% (random baseline: 0.07%)
- [ ] Recall@10 (MRI→Text) > 20%
- [ ] Linear probe CN/MCI/AD accuracy > BrainIAC-only baseline
- [ ] Training converges without collapse (monitor embedding variance)

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| 1,472 pairs → collapse/overfitting | HIGH | Augmentation ×5, strict val monitoring |
| ADNI CDRSB 50.7% missing | MEDIUM | Use diagnosis+age as fallback text |
| Template text too repetitive (low diversity) | MEDIUM | Add stochastic phrasing variants |
| BrainIAC frozen representation too coarse | MEDIUM | EXP 02: try top-2 layer fine-tune |
| AJU zero-shot gap (Korean vs Western) | LOW (expected) | Document and discuss |

---

## Naming Convention

```
brainclip_v{VERSION}_{DESCRIPTOR}
e.g. brainclip_v1_baseline
     brainclip_v1_mri_finetune
     brainclip_v2_larger_proj   ← version bump if architecture changes
```

---

## Target Venue

- Primary: **MICCAI 2026** (submission: March 2026 — may be too late, check)
- Alternative: **Medical Image Analysis** (MedIA), **NeurIPS 2026 workshop**
- Note: NeurIPS main track feasibility → depends on retrieval results

---

## References

- SimCLR (Chen et al., 2020) — NT-Xent contrastive loss
- BioViL (Bannur et al., 2023) — CXR-text CLIP
- ConVIRT (Zhang et al., 2022) — medical image-text contrastive
- BrainIAC (pretrained 3D brain encoder, UK-Biobank)
- PubMedBERT (Gu et al., 2021) — biomedical language model
