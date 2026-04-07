# Results — BrainCLIP V1

**최종 갱신**: 2026-04-06
**상태**: 🔲 미시작

---

## 진행 현황

| Phase | 상태 | ETA |
|-------|------|-----|
| 데이터 파이프라인 구축 | 🟡 진행중 | — |
| 텍스트 생성 파이프라인 | 🔲 대기 | — |
| EXP 01: baseline 학습 | 🔲 대기 | GPU 승인 후 |
| EXP 01: retrieval 평가 | 🔲 대기 | — |
| EXP 02: MRI fine-tune | 🔲 대기 | — |
| EXP 04: AJU zero-shot | 🔲 대기 | — |

---

## 결과 (실험 실행 후 갱신)

### Retrieval Performance

| EXP | Recall@1 | Recall@5 | Recall@10 | MRI→Text | Text→MRI | 텍스트 |
|-----|:--------:|:--------:|:---------:|:--------:|:--------:|--------|
| Random baseline | 0.07% | 0.34% | 0.68% | — | — | — |
| 01 baseline | — | — | — | — | — | diagnosis 포함 |
| 02 no_diag_text | — | — | — | — | — | diagnosis **제외** |

### Linear Probe (CN/MCI/AD)

| EXP | Accuracy | BAcc | AUROC | vs BrainIAC-only | 해석 |
|-----|:--------:|:----:|:-----:|:----------------:|------|
| BrainIAC-only (baseline) | — | — | — | — | frozen BrainIAC raw 768d |
| 01 BrainCLIP (w/ diag) | — | — | — | — | label alignment 측정 |
| 02 BrainCLIP (no diag) | — | — | — | — | **MRI 독립 표현 측정** ← 핵심 |

---

> 실험 완료 후 자동 갱신 예정
