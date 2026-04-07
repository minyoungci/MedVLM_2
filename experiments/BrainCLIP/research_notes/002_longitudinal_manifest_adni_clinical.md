# Research Note #001: BrainCLIP Longitudinal Manifest — ADNI 임상 데이터 커버리지 개선

**작성일**: 2026-04-07
**작성자**: VLM Team
**카테고리**: data
**상태**: 완료

---

## 배경

BrainCLIP 실험을 위한 longitudinal manifest를 구성하던 중, ADNI 세션의 임상 데이터(age, cdr_sum) 커버리지가 심각하게 낮은 문제가 발생했다.

초기 버전(`v1`)에서는 `adni_t1w_with_clinical_matched.csv`를 단일 소스로 사용했는데, 이 파일은 `entry_age`가 7,362행 중 1,300행(17.7%)에만 채워져 있었다. 결과적으로 `age OR cdr_sum` 필터에서 대부분의 ADNI 세션이 탈락해 최종 manifest가 2,594 세션(ADNI 674)으로 줄어들었다 — base manifest의 4,009 ADNI 세션 대비 83% 손실.

---

## 분석

### 문제 진단

| 파일 | entry_age | CDRSB | 비고 |
|------|-----------|-------|------|
| `adni_t1w_with_clinical_matched.csv` | 17.7% (1,300/7,362) | 15.1% (1,111/7,362) | 기존 소스 |
| `All_Subjects_UCSFFSX7_10Nov2025.csv` | ✗ | ✗ | 뇌 형태 측정값만 (337컬럼) |
| `All_Subjects_CDR_02Nov2025.csv` | ✗ | **98.3%** (14,364/14,608) | PTID+VISDATE 매칭 가능 |
| `All_Subjects_Study_Entry_02Nov2025.csv` | **100%** (4,933/4,933) | ✗ | Subject당 baseline 1회 |

### 매칭 전략

```
CDR 매칭:
  key: (PTID == subject_id) AND (VISDATE ≈ visit_date ± 90일)
  → 정확 일치 우선, 실패 시 window 내 최근접 평가 선택
  → 이유: MRI scan date와 CDR assessment date는 같은 날이 아닐 수 있음

Age 주입:
  key: subject_id → entry_age (Study Entry)
  → baseline 1회 측정, 모든 longitudinal 세션에 동일값 적용
  → 주의: 수년 차이 있을 수 있음 (최대 ~10년)
```

### 스크립트

`experiments/BrainCLIP/scripts/build_brainclip_longitudinal_manifest.py`

핵심 함수:
- `build_adni_cdr_lookup()` — CDR CSV 로드, PTID+VISDATE 인덱싱
- `build_adni_age_lookup()` — Study Entry → {subject_id: entry_age} dict
- `find_closest_cdr()` — ±90일 window 내 최근접 CDR 행 반환

---

## 결과

| 항목 | v1 (이전) | v2 (개선) |
|------|-----------|-----------|
| 총 세션 | 2,594 | **5,902** |
| ADNI 세션 | 674 | **3,982** |
| OASIS 세션 | 1,296 | 1,296 |
| NACC 세션 | 624 | 624 |
| age coverage | ~44% | **100%** |
| cdr_sum coverage | ~89.5% | **90.6%** |
| cdr_global coverage | N/A | **90.6%** |

ADNI CDR 매칭 결과:
- 3,667/4,009 세션 CDR 확보 (91.5%)
- 미매칭 342세션: CDR assessment window ±90일 내 평가 없음 → age만으로 통과

---

## 결론 및 다음 단계

### 결론
- `adni_t1w_with_clinical_matched.csv` 단독 사용은 fill rate 문제로 부적절
- CDR + Study Entry 분리 소스를 조합하면 full coverage 달성 가능
- ADNI 세션 74%→100% age coverage, 91.5% CDR 커버리지 달성

### 주의 사항
1. **Age는 baseline 값**: Study Entry의 entry_age는 등록 시 나이. 추후 text 생성 시 "at study entry"로 명시하거나 세션별 delta 계산 필요
2. **CDR 날짜 불일치**: ±90일 window 매칭 사용 — 임상 평가와 MRI 촬영이 동일 방문에서 이루어지지 않을 수 있음
3. **Diagnosis는 "Unknown"**: ADNI diagnosis 컬럼이 대부분 빈값. BrainCLIP text 생성 시 diagnosis 대신 CDR/age 기반 텍스트 사용해야 함

### 다음 단계
1. `build_clinical_text.py` 확장 → longitudinal manifest 기반 세션별 텍스트 생성
2. EXP05 설계: longitudinal full data (~5,900 세션) + backbone fine-tune (EXP03의 ~4.6× 확장)
3. Age delta 계산: Study Entry date + visit_date로 session-level age 추정 가능 여부 확인

---

## 관련 문서

> 관련:
> - [ReproSeg #001 — V1 Ablation Plan](../ReproSeg/001_reproseg_v1_ablation_plan.md) — 공통 데이터 파이프라인 배경
