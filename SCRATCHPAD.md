# SCRATCHPAD — minyoung2 전체 실험 현황

**최종 갱신**: 2026-04-01

---

## 활성 프로젝트

| 프로젝트 | 상태 | 위치 |
|----------|------|------|
| **ReproSeg V1** | 🟡 Smoke test 완료, Ablation 준비 | `experiments/ReproSeg/` |
| Dir1: Cross-Ethnic | 🔲 동결 | `experiments/dir1_cross_ethnic/` |
| Dir2: Implicit Biomarker | 🔲 동결 | `experiments/dir2_implicit_biomarker/` |
| Dir3: Brain VLM | ⚠️ 동결 (LoV3D 위기) | `experiments/dir3_brain_vlm/` |
| Dir5: Native vs MNI | 🔲 동결 | `experiments/dir5_native_vs_mni/` |

## ReproSeg V1 Ablation 현황

| EXP | Name | 상태 | Dice | CV |
|-----|------|------|:----:|:--:|
| A | `reproseg_v1_A_baseline` | 🔲 다음 실행 | — | — |
| B | `reproseg_v1_B_tcl` | 🔲 대기 | — | — |
| C | `reproseg_v1_C_dualstream` | 🔲 대기 | — | — |
| D | `reproseg_v1_D_volume` | 🔲 대기 | — | — |
| E | `reproseg_v1_E_repro_inv` | 🔲 대기 | — | — |
| F | `reproseg_v1_F_full` | 🔲 대기 | — | — |

**Smoke test**: 3ep baseline, Dice 0.899 (정상 학습 확인)

## NeurIPS 2026 타임라인

| 주차 | 기간 | 목표 |
|------|------|------|
| Week 1 | 4/1-4/7 | ReproSeg Ablation A-C 실행 |
| Week 2 | 4/7-4/14 | Ablation D-F + CV 평가 |
| Week 3 | 4/14-4/21 | 재현성 벤치마크 + 메트릭 분석 |
| Week 4 | 4/21-4/28 | 논문 작성 + figure |
| Week 5 | 4/28-5/06 | 최종 수정 + 제출 |

---

_Last updated: 2026-04-01_
