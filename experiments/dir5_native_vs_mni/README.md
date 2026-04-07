# Direction 5: Native vs MNI Space — Registration Considered Harmful?

## 가설
MNI 공간 변환(FLIRT/FNIRT)은 DL 기반 치매 분류에서 정보 손실을 야기하며, native space 입력이 동등하거나 우수한 성능을 보인다. 특히 한국인 뇌에서 MNI template bias가 더 크다.

## 데이터
- **MNI data**: V2 (`/home/vlm/data/preprocessed/`) — 6,193 subjects
- **Native data**: V4 (미완성, `/home/vlm/data/preprocessed_V3/` 일부) — 대규모 실행 필요
- **Manifest**: 3,841명

## 방법

### Phase 0: V4 데이터 생산 (전제 조건)
1. V4 native space pipeline 대규모 실행 (Min 승인 필요)
2. 최소 1,000명 이상 처리 후 Phase 1 시작 가능

### Phase 1: Information Loss 정량화 (1주)
1. 동일 피험자에 대해 V2(MNI) vs V4(native) label voxel count 비교
2. 구조별 warp loss 정량화: hippocampus, amygdala, entorhinal 등
3. Warp loss가 진단(CN/MCI/AD)과 상관있는지 확인 (atrophic brain에서 더 큰 loss?)

### Phase 2: 분류 성능 비교 (1-2주)
1. 동일 모델(BrainIAC LoRA), V2 vs V4 입력 → BAcc, CI 비교
2. Segmentation 성능 비교: Dice on native vs MNI
3. Cross-ethnic 비교: MNI warp가 한국인 뇌에서 더 큰 왜곡을 만드는가?

### Phase 3: Registration-free DL (1주)
1. Native space + spatial transformer network (STN) vs rigid augmentation
2. Arbitrary orientation 처리 가능한 모델 설계

## 실험 목록
| EXP | 설명 | 상태 | 핵심 결과 |
|-----|------|------|----------|
| EXP-00 | V4 대규모 파이프라인 실행 | 🔲 Blocked (V4 데이터 필요) | |
| EXP-01 | Warp loss 정량화 | 🔲 대기 | |
| EXP-02 | Classification: V2 vs V4 | 🔲 대기 | |
| EXP-03 | Cross-ethnic warp analysis | 🔲 대기 | |
| EXP-04 | Registration-free model | 🔲 대기 | |

## 성공 기준
- [ ] V4 데이터 ≥ 1,000명 생산 — Phase 0 gate
- [ ] Warp loss > 20% for ≥2 structures — Phase 1 gate
- [ ] Native-space model BAcc ≥ MNI-space model BAcc — Phase 2 gate

## 위험 요소
- **V4 파이프라인 미완성** — Phase 0이 전제 조건. 다른 방향 우선 진행
- **성능 차이가 미미할 수 있음** — MNI 변환이 DL에서는 무해할 가능성
- **Native space 모델이 data augmentation에 더 민감** — 추가 실험 필요

## 타겟 저널
- Primary: NeuroImage / Medical Image Analysis
- Alternative: MICCAI

## 비고
이 방향은 V4 데이터 의존성이 높아 **우선순위가 가장 낮음**. Dir1-4 진행 중 V4 파이프라인 완성 시 시작.
