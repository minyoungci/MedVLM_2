# SCRATCHPAD — BrainCLIP

**최종 갱신**: 2026-04-06
**상태**: 🟡 스크립트 완료, 의존성 승인 대기 + build_clinical_text 실행 대기

---

## 데이터 현황 (확인 완료 2026-04-06)

### 사용 가능 페어
- NACC: 768명, 임상 변수 100% (CDR/GDS/RACE/EDUC)
- ADNI: 704명, 진단+나이 100%, CDRSB 50.7%
- AJU: 719명, 임상 텍스트 없음 → 학습 제외, zero-shot eval 전용
- **총 학습 페어: 1,472명**

### MRI 포맷
- Shape: (192, 224, 192), dtype: float32
- 경로: `/home/vlm/data/preprocessed_v4/cross_sectional/{NACC,ADNI}/`

### 주의사항
- NACCMMSE 결측 99.5% → 텍스트에서 제외
- ADNI CDRSB 결측 시 → "CDR-SB not available" 또는 diagnosis만 사용
- NACC NACCUDSD 코드: 1=CN, 2=MCI, 3=AD, 4=AD+other 가능성 (코드북 확인 필요)

---

## 텍스트 생성 전략 (설계 중)

```python
# NACC 텍스트 템플릿 예시
template = (
    "{age}-year-old {sex} {race}, {educ} years education. "
    "CDR global {cdr_global} ({cdr_severity}), CDR sum of boxes {cdr_sum}. "
    "Memory domain: {memory}/3. "
    "{gds_str}"  # GDS available → add, else skip
    "Clinical diagnosis: {diagnosis}."
)

# ADNI 텍스트 템플릿 (CDR-SB 있을 때)
template_adni = (
    "{age}-year-old patient. "
    "CDR-SB {cdrsb}. "
    "Clinical group: {research_group}. "  # CN/MCI/AD
    "Diagnosis: {diagnosis}."
)
```

---

## 구현 TODO

- [x] `scripts/build_clinical_text.py` — NACC/ADNI 변수 → 텍스트 변환
- [x] `scripts/dataset_brainclip.py` — MRI + 텍스트 데이터셋 클래스
- [x] `scripts/model_brainclip.py` — BrainCLIP 모델 정의
  - BrainIACEncoder interface 버그 3개 수정 (checkpoint_path, Encoder3DOutput, backbone.blocks 경로)
- [x] `scripts/extract_mri_embeddings.py` — 오프라인 BrainIAC feature 사전 추출
- [x] `scripts/train_brainclip.py` — DDP 학습 스크립트 (warmup+cosine LR, bf16, 체크포인트)
- [x] `scripts/eval_retrieval.py` — Recall@K 평가 (MRI→Text, Text→MRI)
- [x] `scripts/eval_linear_probe.py` — CN/MCI/AD linear probe
- [x] `configs/exp01_baseline.toml` — 기본 실험 설정

## Ablation 설계 (EXP 01 vs EXP 02)

| 항목 | EXP 01 baseline | EXP 02 no_diag_text |
|------|----------------|---------------------|
| 텍스트 | diagnosis 포함 | diagnosis **제외** (age/CDR/GDS만) |
| 텍스트 CSV | `clinical_texts.csv` | `clinical_texts_no_diag.csv` |
| Config | `exp01_baseline.toml` | `exp02_no_diag_text.toml` |
| Retrieval R@K | 높을 것 (diagnosis가 강한 신호) | 낮을 것 (예상) |
| Linear probe | "label alignment" 측정 | **"MRI 독립 표현 학습" 측정** ← 논문 핵심 claim |

→ EXP 02 linear probe가 EXP 01보다 낮더라도 정직한 수치. BrainIAC-only 대비 개선이면 기여 성립.

## 블로커 (Min 승인 필요)

1. **pyproject.toml 의존성 추가** (승인 필요):
   ```
   transformers>=4.40.0
   scikit-learn>=1.4
   matplotlib>=3.9
   ```
2. **build_clinical_text.py 실행 — 두 버전**:
   ```bash
   cd /home/vlm/minyoung2
   # EXP 01용 (diagnosis 포함)
   UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
       experiments/BrainCLIP/scripts/build_clinical_text.py \
       --output experiments/BrainCLIP/data/clinical_texts.csv --preview 5

   # EXP 02용 (diagnosis 제외)
   UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
       experiments/BrainCLIP/scripts/build_clinical_text.py \
       --output experiments/BrainCLIP/data/clinical_texts_no_diag.csv \
       --no_diagnosis --preview 5
   ```
3. **extract_mri_embeddings.py 실행** (GPU 필요, BrainIAC 추출):
   ```bash
   UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
       experiments/BrainCLIP/scripts/extract_mri_embeddings.py \
       --output_dir experiments/BrainCLIP/data/mri_embeddings --batch_size 4
   ```

---

## 의존성 추가 필요 (pyproject.toml)

현재 없음 → 추가 필요 (Min 승인 후):
- `transformers>=4.40.0` (PubMedBERT tokenizer + model)
- `scikit-learn>=1.4` (linear probe)
- `matplotlib>=3.9` (figure 생성)

---

## 다음 단계

1. NACC 코드북 확인 → SEX/RACE/NACCUDSD 매핑 테이블 작성
2. `build_clinical_text.py` 구현 → 실제 텍스트 품질 확인
3. pyproject.toml dependency 추가 (승인 필요)
4. BrainIAC feature extraction smoke test

---

_Last updated: 2026-04-06_
