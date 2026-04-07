# SCRATCHPAD — ReproSeg V1

**최종 갱신**: 2026-04-01
**상태**: 🟡 Smoke test 완료 → 코드 감사 수정 완료 → Ablation 실행 준비
**아키텍처**: ReproSeg V1 (75.7M params, 새 component 13.5M)

---

## Smoke Test 결과 (reproseg_v1, baseline mode, 3 epochs)

| Epoch | Train Loss | Val Loss | Mean Dice | 시간 |
|:-----:|:----------:|:--------:|:---------:|:----:|
| 0 | 1.142 | 0.774 | 0.552 | 25분 |
| 1 | 0.251 | 0.228 | 0.889 | 25분 |
| **2** | **-0.363** | **0.160** | **0.899** | 25분 |

### Epoch 2 구조별 Dice
| 구조 | Dice |
|------|:----:|
| subcortical | 0.954 |
| ventricle | 0.939 |
| white_matter | 0.936 |
| hippocampus_R | 0.920 |
| hippocampus_L | 0.912 |
| amygdala_R | 0.904 |
| cortical_L | 0.891 |
| amygdala_L | 0.890 |
| cortical_R | 0.888 |
| entorhinal_R | 0.831 |
| entorhinal_L | 0.825 |

### Smoke Test에서 발견된 버그 (모두 수정 완료)
- [x] **[C1]** UWL zero-loss weight explosion → `active_mask` 도입
- [x] **[C2]** Paired scan flip 독립 적용 → `shared_flip` 도입
- [x] **[H1]** InvarianceStream 22M → 5.6M (채널 절반)
- [x] **[H3]** CUDA seed 미설정 → `torch.cuda.manual_seed` 추가

---

## 기존 minyoung/ 참고 결과

### TCL λ Sweep (minyoung/ Phase 1)
| λ | Dice | CV (전체) |
|---|:----:|:---------:|
| baseline | 0.918 | 4.29% |
| **1.0** | **0.922** | **3.95% (-8.0%)** |
| 1.0+TTA | 0.922 | **3.69%** |

---

## Ablation 실행 계획

| 순서 | EXP | GPU | Epochs | 예상 시간 |
|:----:|-----|-----|:------:|:---------:|
| 1 | `reproseg_v1_A_baseline` | 1,7 | 30 | ~12.5h |
| 2 | `reproseg_v1_B_tcl` | 1,7 | 30 | ~14h |
| 3 | `reproseg_v1_C_dualstream` | 1,7 | 30 | ~14h |
| 4 | `reproseg_v1_D_volume` | 1,7 | 30 | ~13h |
| 5 | `reproseg_v1_E_repro_inv` | 1,7 | 30 | ~15h |
| 6 | `reproseg_v1_F_full` | 1,7 | 30 | ~15h |

**총 예상**: ~84h (3.5일, 순차 실행 시)

---

## Gate 체크리스트

- [ ] A baseline Dice ≥ 0.915 (backbone 정상 작동)
- [ ] B의 CV < A의 CV (TCL 효과)
- [ ] C의 site accuracy 감소 (GRL 작동)
- [ ] E의 CV < B의 CV (아키텍처 > loss-only)
- [ ] F의 Dice ≥ 0.920 AND CV 최저
- [ ] Ablation 단조: A < B < ... < F

## Smoothing 반론 방어

1. Hausdorff Distance 95th: smoothing이면 HD도 개선되어야 함
2. Per-voxel entropy: 전체 smooth vs boundary-only consistent 구분
3. Downstream CI 유지: 임상 discriminative power 손실 없음

## 관련 문헌

- Kondrateva et al. 2025 (arxiv 2504.15931): FastSurfer/SynthSeg 재현성 벤치마크
- Isensee et al. 2024 (MICCAI): nnU-Net Revisited — training recipe > architecture
- Kendall et al. 2018 (CVPR): Uncertainty weighting — 5000+ citations
- Domain Unlearning (PRIME 2024): Multi-stage domain confusion for brain MRI

---

_Last updated: 2026-04-01_
