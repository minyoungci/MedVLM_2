# Experiment Index

## ReproSeg V1 (Active)

| EXP | Name | 설명 | 데이터 | 상태 |
|-----|------|------|--------|------|
| A | `reproseg_v1_A_baseline` | Backbone + Decoder only | V4 | 🔲 |
| B | `reproseg_v1_B_tcl` | + TCL + Feature Consistency | V4 | 🔲 |
| C | `reproseg_v1_C_dualstream` | + InvStream + CSG + GRL | V4 | 🔲 |
| D | `reproseg_v1_D_volume` | + Volume Head | V4 | 🔲 |
| E | `reproseg_v1_E_repro_inv` | C + B | V4 | 🔲 |
| F | `reproseg_v1_F_full` | All components | V4 | 🔲 |

---

## 동결 (ReproSeg 완료 후 재개)

### Dir1: Cross-Ethnic Brain AI Gap
| EXP | 설명 | 데이터 | 상태 |
|-----|------|--------|------|
| dir1/EXP-01 | Western-train → AJU-test pilot | V2 | 🔲 동결 |

### Dir2: Implicit Biomarker Discovery
| EXP | 설명 | 데이터 | 상태 |
|-----|------|--------|------|
| dir2/EXP-01 | FreeSurfer residual CI pilot | V2 | 🔲 동결 |

### Dir3: Brain VLM Report Generation
⚠️ LoV3D 경쟁 논문으로 동결

### Dir5: Native vs MNI
V4 데이터 완성 후 재개
