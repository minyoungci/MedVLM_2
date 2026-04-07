# Direction 3: Brain VLM for Dementia Report Generation

## 가설
Brain MRI에서 직접 자연어 임상 리포트를 생성하는 VLM은 기존 pipeline(FreeSurfer → 수동 해석)을 대체할 수 있으며, 전 세계적으로 최초의 시도이다.

## 데이터
- **Primary**: V2 (`/home/vlm/data/preprocessed/`)
- **Subjects**: manifest 3,841명
- **Image Input**: mni_t1w.nii.gz (3D) 또는 slices.npz (32-slice 2D bundle)
- **Structured Input**: volumes.json (ROI volumes), tissue_volumes.json
- **Report GT**: 생성 필요 (핵심 병목)

## 방법

### Phase 0: Report Ground Truth 생성 전략 (1주)
1. **Template-based**: FreeSurfer volumes → z-score → 규칙 기반 리포트 자동 생성
2. **LLM-assisted**: GPT-4/Claude로 volumes + diagnosis → 리포트 초안 생성 → 전문의 검수
3. **Hybrid**: Template 기반 structured report + LLM polish
4. 전문의 협업 가능성 확인

### Phase 1: Structured Report Generation (1-2주)
1. Image → 12 ROI별 위축 등급 (Normal/Mild/Moderate/Severe) 예측
2. 종합 소견 생성: "주요 위축 부위", "추정 진단", "권고 사항"
3. 기존 Region-specific MLP (V2, Grade Acc 74.3%) 확장

### Phase 2: Brain VQA (1-2주)
1. QA 데이터셋 구축: "Is there hippocampal atrophy?", "What is the most affected region?"
2. Image + Question → Answer (yes/no, region name, severity)
3. VQA 모델: Vision encoder + LLM decoder (Qwen2.5-3B or Mistral-7B)

### Phase 3: Natural Language Report (2주)
1. Image → Full-text clinical report
2. 기존 V3 PoC (Mistral-7B, Faithfulness 78.1%) 개선
3. Hallucination rate 목표: < 3%
4. Evaluation: faithfulness, hallucination, clinical utility (전문의 평가)

### Phase 4: Grounded Explanation (1주)
1. Classification 근거를 3D attention overlay + 자연어로 설명
2. "이 환자는 양측 해마 위축(Grade 3)과 뇌실 확장으로 AD 가능성이 높습니다"

## 실험 목록
| EXP | 설명 | 상태 | 핵심 결과 |
|-----|------|------|----------|
| EXP-00 | Report GT 생성 파이프라인 | 🔲 미시작 | |
| EXP-01 | Structured atrophy grading | 🔲 대기 | |
| EXP-02 | Brain VQA dataset + model | 🔲 대기 | |
| EXP-03 | NL report generation | 🔲 대기 | |
| EXP-04 | Grounded explanation | 🔲 대기 | |

## 성공 기준
- [ ] Report GT ≥ 2,000개 구축 — Phase 0 gate
- [ ] Structured grading: Grade Acc > 75%, CI > 0.76 — Phase 1 gate
- [ ] VQA accuracy > 80% on held-out set — Phase 2 gate
- [ ] NL report: Faithfulness > 85%, Hallucination < 3% — Phase 3 gate

## 위험 요소
- **Report GT 구축이 최대 병목** — 전문의 인력/시간 필요. Template-based로 먼저 시작
- **Hallucination** — 의료 도메인에서 치명적. 강력한 faithfulness 검증 필수
- **BrainIAC 팀의 VLM 확장 가능성** — 시간 경쟁. 빠른 PoC 확보가 중요
- **3D → 2D slice 변환 시 정보 손실** — 3D VLM vs 2D slice VLM 비교 필요

## 타겟 저널
- Primary: Nature Medicine / Lancet Digital Health
- Alternative: NeurIPS / Medical Image Analysis
