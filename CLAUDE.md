# CLAUDE.md — minyoung2 Workspace

## Role
Critical technical collaborator. Primary focus: **ReproSeg V1** (brain segmentation reproducibility).

## Stack
```
uv run python <script>
UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache
.venv -> /home/vlm/minyoung/.venv              # shared venv
pretrain -> /home/vlm/minyoung/pretrain         # shared pretrained weights
shared/ -> symlinks to minyoung/model-claude/src/  # shared code
```

## Data Paths
```
/home/vlm/data/preprocessed_v4/       # V4 native space — ReproSeg primary
/home/vlm/data/preprocessed/           # V2 (MNI, 6,193 subjects) — other directions
/home/vlm/data/preprocessed_V3/        # V3.5 (MNI, 3,361 subjects)
/home/vlm/data/metadata/v4_manifest.csv        # V4 manifest (2,397 subjects)
/home/vlm/data/metadata/final_3841/manifest.csv  # V2 manifest
```

### Data Selection per Experiment
| Experiment | Data | Reason |
|------------|------|--------|
| **ReproSeg** | **V4 native** | 2,397 subjects, 6,554 longitudinal sessions |
| dir1_cross_ethnic | V2 | AJU 1,008 needed |
| dir2_implicit_biomarker | V2 | volumes.json + slices.npz |
| dir3_brain_vlm | V2 | Max subjects + 32-slice bundles |
| dir5_native_vs_mni | V2 + V4 | Space comparison |

## Active Experiments
| Experiment | Folder | Status |
|------------|--------|--------|
| **ReproSeg V1** | `experiments/ReproSeg/` | 🟡 Ablation 준비 |
| Dir1-5 | `experiments/dir{1-5}_*/` | 🔲 동결 |

## Versioning Convention
- `reproseg_v{N}_{ablation}_{descriptor}` — 실험 이름
- Version은 결과 분석 후 아키텍처 변경 시 증가 (V1 → V2 → ...)
- ARCHITECTURE.md에 버전별 변경 이력 기록

## Confirmation Gates
- GPU script execution
- 10+ file bulk changes
- pyproject.toml / dependency changes
- /home/vlm/data/raw/ write/delete

## Commit Convention
`type(scope): 한국어` — type: feat fix data exp refactor docs chore

## Rules
- bf16 mandatory, fp16 prohibited
- uv run python on server
- Each experiment MUST update RESULTS.md on completion
- Cross-experiment insights go to SCRATCHPAD.md
- Never modify /home/vlm/minyoung/ files from this workspace
- Version changes require ARCHITECTURE.md update with changelog
