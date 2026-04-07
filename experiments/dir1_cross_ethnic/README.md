# Direction 1: Cross-Ethnic Brain AI Gap

## 가설
서양인 데이터로 학습된 brain AI 모델은 한국인(AJU) 치매 환자에서 체계적 성능 하락을 보이며, 특정 뇌 구조와 질환 스테이지에서 gap이 집중된다.

## 데이터
- **Primary**: V2 (`/home/vlm/data/preprocessed/`)
- **Western**: NACC (1,652) + OASIS (1,786) + ADNI (1,747) = 5,185
- **Korean**: AJU (1,008)
- **Manifest**: `/home/vlm/data/metadata/final_3841/manifest.csv` (3,841명, train/val/test split)
- **필터링**: manifest에 포함된 피험자만 사용 (QC 통과 + 진단 라벨 있는 경우)

## 방법

### Phase 1: Pilot (1-2일) — Gap 존재 확인
1. Manifest에서 Western-only (NACC+OASIS+ADNI) train/val split 생성
2. 기존 E-4 모델 (BrainIAC LoRA rank=4, brain-only pooling, Focal loss) 재학습
3. AJU 전체를 test set으로 평가
4. Per-class BAcc, per-region CI 비교

### Phase 2: Gap 분석 (1주)
1. 구조별 성능 하락 정량화 (hippocampus, amygdala, entorhinal 등)
2. 질환 스테이지별 gap (CN vs MCI vs AD)
3. Scanner/site confound 분석 (AJU 8개 병원)
4. BrainIAC pretrained feature t-SNE → ethnic clustering 확인

### Phase 3: Gap 해소 (2주)
1. Few-shot Korean adaptation: AJU 50/100/200/500명으로 fine-tune
2. Domain adaptation: CORAL, GRL, ComBat harmonization
3. Cross-ethnic transfer learning curve 생성

### Phase 4: 논문 figure + external validation
1. Gap heatmap (region × stage)
2. Adaptation curve (Korean N vs performance)
3. 추가 한국 데이터로 external validation (available 시)

## 실험 목록
| EXP | 설명 | 상태 | 핵심 결과 |
|-----|------|------|----------|
| EXP-01 | Pilot: Western-train → AJU-test | 🔲 미시작 | |
| EXP-02 | Region-specific gap analysis | 🔲 대기 | |
| EXP-03 | Few-shot Korean adaptation | 🔲 대기 | |
| EXP-04 | Domain adaptation comparison | 🔲 대기 | |

## 성공 기준
- [ ] AJU BAcc drop > 5% (Gap 존재 확인) — Phase 1 gate
- [ ] Region-specific gap pattern identified — Phase 2 gate
- [ ] Few-shot adaptation closes gap > 50% with N≤200 — Phase 3 gate
- [ ] External validation on new Korean data — Phase 4 gate

## 위험 요소
- **Gap이 작을 수 있음** → Phase 1 pilot로 빠르게 확인 (kill/go 결정)
- **AJU scanner heterogeneity** → 8개 병원 간 confound가 ethnic gap과 혼동될 수 있음
- **한국 데이터 추가 시점 불확실** → 현재 AJU만으로 Phase 1-3 완결 가능하게 설계

## 타겟 저널
- Primary: Lancet Digital Health / Nature Medicine
- Alternative: MICCAI 2026
