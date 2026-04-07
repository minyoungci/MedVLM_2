# Direction 2: Implicit Neuroanatomical Biomarker Discovery

## 가설
Vision encoder는 FreeSurfer가 정의한 ROI 외의 영역에서 추가적인 치매 관련 정보를 학습하며, 이 "implicit biomarker"는 FreeSurfer volume을 regress out한 후에도 통계적으로 유의한 예측력을 가진다.

## 데이터
- **Primary**: V2 (`/home/vlm/data/preprocessed/`)
- **Subjects**: manifest 3,841명 (train/val/test split 유지)
- **Features**: mni_t1w.nii.gz (3D 이미지) + volumes.json (34 FreeSurfer features) + slices.npz (32-slice bundle)
- **Ground Truth**: diagnosis (CN/MCI/AD), volumes.json (FreeSurfer ROI volumes)

## 방법

### Phase 1: Pilot (1-2일) — Residual 예측력 확인
1. 기존 E-4 vision encoder의 embedding 추출 (3,841명)
2. FreeSurfer 34 volumes로 embedding → diagnosis prediction (linear probe)
3. FreeSurfer volumes를 regress out한 residual embedding 생성
4. Residual embedding → diagnosis prediction (linear probe)
5. **Gate**: Residual CI > 0.55이면 viable

### Phase 2: Implicit Biomarker 탐색 (1-2주)
1. Grad-CAM / attention map 추출 (3,841명 전체)
2. 3D attention map을 FreeSurfer ROI mask로 분해: ROI-내 vs ROI-외 activation 비율
3. CN/MCI/AD 그룹별 평균 attention map → "implicit biomarker map" 생성
4. FreeSurfer ROI 외부에서 일관된 고활성 영역 식별

### Phase 3: 임상 검증 (1주)
1. 발견된 implicit region이 알려진 neuropathology와 일치하는지 문헌 대조
2. White matter hyperintensity, periventricular changes, cerebrovascular pattern과의 관계
3. Cross-ethnic 비교: 한국인 vs 서양인에서 implicit biomarker가 다른가?

### Phase 4: Head-to-head 비교표 (1주)
1. FreeSurfer(6h) vs FastSurfer(1min) vs SynthSeg(30s) vs End-to-end DL
2. 정확도 + 재현성 + 속도 + 임상 유용성 종합 비교

## 실험 목록
| EXP | 설명 | 상태 | 핵심 결과 |
|-----|------|------|----------|
| EXP-01 | Pilot: FreeSurfer residual CI | 🔲 미시작 | |
| EXP-02 | 3D attention map extraction | 🔲 대기 | |
| EXP-03 | Implicit region identification | 🔲 대기 | |
| EXP-04 | Cross-ethnic implicit biomarker | 🔲 대기 | |
| EXP-05 | Pipeline comparison table | 🔲 대기 | |

## 성공 기준
- [ ] Residual CI > 0.55 (FreeSurfer 외 정보 존재 확인) — Phase 1 gate
- [ ] ≥3 consistent non-ROI regions identified — Phase 2 gate
- [ ] At least 1 region matches known neuropathology — Phase 3 gate

## 위험 요소
- **Residual 예측력이 없을 수 있음** → Phase 1 pilot로 빠르게 kill/go
- **3D attention map이 noisy** → 대규모 평균화(3,841명)로 noise 감소, permutation test 필요
- **BrainIAC Grad-CAM 해상도** → 96³ input에서 6³ patch → 16mm 해상도. 미세 구조 식별 어려울 수 있음

## 타겟 저널
- Primary: NeuroImage / Nature Communications
- Alternative: NeurIPS (방법론 프레이밍)
