# Data Guide — V2 vs V3.5 vs V4

## V2 (`/home/vlm/data/preprocessed/`)

| 항목 | 값 |
|------|-----|
| Space | MNI152 |
| Shape | 182×218×182 |
| Resolution | 1mm isotropic |
| Subjects | 6,193 (ADNI 1,747 / OASIS 1,786 / NACC 1,652 / AJU 1,008) |
| BET | FSL BET cascade (4-stage fallback) |
| Bias Correction | FSL FAST (post-BET) |
| Registration | FLIRT 2-stage + FNIRT (fallback) |
| Segmentation | FastSurfer DKT (87 labels) |
| QC | Mostly GOOD/MARGINAL |

### Per-Subject Files
```
{subject}/
├── original_t1w.nii.gz          # Original T1w
├── bet.nii.gz                   # Brain-extracted (native)
├── bet_biascorr.nii.gz          # + bias correction
├── bet_biascorr_norm.nii.gz     # + intensity normalization
├── seg_native.nii.gz            # FastSurfer seg (native)
├── native2mni.mat               # FLIRT affine matrix
├── mni_t1w.nii.gz               # T1w in MNI space
├── mni_segmentation.nii.gz      # Seg in MNI space
├── volumes.json                 # Per-label volumes (mm³)
├── tissue_volumes.json          # GM/WM/CSF volumes
├── qc_metrics.json              # QC metrics + category
├── slices.npz                   # 32-slice bundle (10ax+12cor+10sag)
├── roi_mask.npz                 # Binary ROI masks
├── slice_metadata.json          # Slice extraction info
├── fastsurfer/                  # FastSurfer output
└── qc/                          # QC visualization JPEGs
```

---

## V3.5 (`/home/vlm/data/preprocessed_V3/`)

| 항목 | 값 |
|------|-----|
| Space | MNI152 |
| Shape | 182×218×182 |
| Resolution | 1mm isotropic |
| Subjects | 3,361 (ADNI 1,747 / NACC 1,142 / OASIS 451 / AJU 21) |
| BET | HD-BET (deep learning) |
| Bias Correction | ANTsPy N4 (pre-registration) |
| Registration | FLIRT affine only (NO FNIRT) |
| Segmentation | FastSurfer DKT (96 labels) |
| QC | ADNI 99% FAILED, NACC 80% FAILED |

### Per-Subject Files
```
{subject}/
├── original_t1w.nii.gz
├── n4_corrected.nii.gz          # N4 bias-corrected
├── mni_t1w.nii.gz               # T1w in MNI space
├── mni_segmentation.nii.gz      # Seg in MNI space
├── mni_brain_mask.nii.gz        # HD-BET mask in MNI
├── native2mni.mat               # FLIRT affine
├── volumes.json                 # Per-label volumes
├── qc_metrics.json              # QC metrics
├── fastsurfer/                  # FastSurfer output
└── qc/                          # QC JPEGs (6 panels)
```

---

## V4 Native Space (미완성)

| 항목 | 목표 값 |
|------|---------|
| Space | Native (subject-specific) |
| Shape | 192×224×192 (padded) |
| Resolution | 1mm isotropic |
| Registration | None |
| Label Warp Loss | ~0% (vs V2의 25-33%) |

### 핵심 차이
- V2/V3.5: MNI warp → label voxel 25-33% 손실, atrophy 신호 약화
- V4: Native space → 형태학적 정보 완전 보존

---

## Task별 데이터 선정 근거

| Experiment | Data | Why |
|------------|------|-----|
| **ReproSeg V1** | **V4 native** | **2,397명 manifest, 6,554 longitudinal sessions** |
| Dir1: Cross-Ethnic | V2 | AJU 1,008명 필수. V3.5는 21명뿐 |
| Dir2: Implicit Biomarker | V2 | volumes.json + slices.npz 완비 |
| Dir3: Brain VLM | V2 | 최대 피험자 수 + 32-slice bundle |
| Dir5: Native vs MNI | V2 + V4 | V4 대규모 실행 후 비교 |

## Manifests
- V4: `/home/vlm/data/metadata/v4_manifest.csv` (2,397 subjects, GOOD+MARGINAL)
- V2: `/home/vlm/data/metadata/final_3841/manifest.csv` (3,841 subjects)
