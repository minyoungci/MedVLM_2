# Results — ReproSeg V1

**최종 갱신**: 2026-04-05 09:50 (A~F 확정값 반영)  
**상태**: A~F 완료 ✓ | Seed 실험 진행 중 | G/H 대기 중 | SynthSeg 대기 중

---

## 핵심 수치 (A~F 확정값)

| Metric | Value | 비교 대상 |
|--------|-------|----------|
| Best ICC — C (InvStream+CSG) | **0.8906** | A baseline 0.8366 |
| ΔICC (C vs A) | **+0.054** | — |
| Worst ICC — F (Full model) | 0.7507 | A보다 −0.086 |
| ICC range | 0.7507 ~ 0.8906 | — |

---

## 실험별 확정 결과 (CONFIRMED, deterministic eval)

| EXP | 구성 | ICC | CV% | ΔICC | 판정 |
|-----|------|:---:|:---:|:----:|------|
| **A** | Baseline (SwinUNETR) | 0.8366 | 8.39% | — | 기준 |
| **B** | +TCL | 0.7996 | — | −0.037 | ❌ 역효과 |
| **C** | +InvStream+CSG+GRL | **0.8906** | — | **+0.054** | ✅ BEST |
| **D** | +VolumeHead | 0.8355 | — | −0.001 | ➖ 효과 없음 |
| **E** | +InvStream+GRL+PCGrad | 0.8640 | — | +0.027 | ✅ 개선 |
| **F** | Full (모든 loss) + PCGrad | 0.7507 | — | −0.086 | ❌❌ 최악 |

### 핵심 발견
1. **C가 명확히 최선** — InvStream+CSG 메커니즘이 재현성에 직접 기여
2. **D ≈ A** — Volume auxiliary는 재현성에 사실상 무효
3. **F가 최악** — 전체 loss 조합이 baseline보다 크게 저하. UWL weight 불안정 의심
4. **B < A** — TCL 단독은 오히려 해로움

---

## 진행 중 실험

| EXP | 상태 | 목적 | ETA |
|-----|------|------|-----|
| A_seed0 | 🔄 학습 중 (GPU 1,2) | 통계 CI | ~+15h |
| A_seed1 | 🔄 학습 중 (GPU 4,5) | 통계 CI | ~+15h |
| C_seed0 | ⏳ 큐 대기 | 통계 CI | GPU 여유 후 |
| C_seed1 | ⏳ 큐 대기 | 통계 CI | GPU 여유 후 |
| G (GRL-only) | ⏳ 큐 대기 | Ablation | GPU 여유 후 |
| H (CSG-only) | ⏳ 큐 대기 | Ablation | GPU 여유 후 |
| SynthSeg ICC | ⏳ 자동 실행 예정 | SOTA 비교 | 재평가 후 |

---

## 결론 (잠정)

**C (InvarianceStream + CrossStreamGating + GRL)**이 제안 방법으로 확정.  
ΔICC +0.054 (0.8366 → 0.8906)으로 MICCAI 논문 기준 유의미한 개선.  
G/H ablation 완료 후 InvStream vs CSG 개별 기여 분리 가능.

---

## 다음 단계

1. Seed 실험 (A×3, C×3) → 95% CI 계산 → 통계 유의성 확인
2. G/H ablation → mechanism 명확화
3. SynthSeg ICC 비교
4. Table 1 초안 작성 (모든 결과 취합 후)
