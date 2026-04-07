"""
build_clinical_text.py
NACC / ADNI 구조화 임상 변수 → 자연어 텍스트 변환

Usage:
    # 기본 (진단 포함): EXP 01 baseline용
    cd /home/vlm/minyoung2
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
        experiments/BrainCLIP/scripts/build_clinical_text.py \
        --output experiments/BrainCLIP/data/clinical_texts.csv \
        --preview 5

    # 진단 제외: EXP 02 ablation용
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
        experiments/BrainCLIP/scripts/build_clinical_text.py \
        --output experiments/BrainCLIP/data/clinical_texts_no_diag.csv \
        --no_diagnosis --preview 5

⚠ include_diagnosis=True (기본): linear probe 결과가 "label alignment" 품질을 측정함.
   include_diagnosis=False:       linear probe 결과가 "MRI 독립 표현 학습" 품질을 측정함.
   논문에서 두 버전을 모두 보고해야 EXP 01 vs EXP 02 비교가 의미를 가짐.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

# ── 경로 상수 ──────────────────────────────────────────────────────────────
MANIFEST       = Path("/home/vlm/data/metadata/v4_manifest.csv")
NACC_CLINICAL  = Path("/home/vlm/data/raw/NACC/NACC-Clinical/commercial_nacc70.csv")
ADNI_CLINICAL  = Path("/home/vlm/data/raw/ADNI/adni34_t1w_with_clinical.csv")

# ── NACC 코드북 매핑 ───────────────────────────────────────────────────────
SEX_MAP = {1: "male", 2: "female"}

RACE_MAP = {
    1: "White",
    2: "Black or African American",
    3: "American Indian or Alaska Native",
    4: "Native Hawaiian or Pacific Islander",
    5: "Asian",
    50: "Other",
    99: "Unknown",
}

NACCUDSD_MAP = {
    1: "cognitively normal",
    2: "mild cognitive impairment",
    3: "Alzheimer's dementia",
    4: "Alzheimer's dementia with other conditions",
    5: "non-Alzheimer's dementia",
}

CDR_SEVERITY = {
    0.0: "normal",
    0.5: "very mild impairment",
    1.0: "mild impairment",
    2.0: "moderate impairment",
    3.0: "severe impairment",
}

MEMORY_MAP = {
    0.0: "intact",
    0.5: "questionable",
    1.0: "mild impairment",
    2.0: "moderate impairment",
    3.0: "severe impairment",
}

# ── NACC 텍스트 생성 ───────────────────────────────────────────────────────
def _nacc_row_to_text(row: pd.Series, include_diagnosis: bool = True) -> str:
    parts = []

    # 인구통계
    age  = int(row["NACCAGE"]) if pd.notna(row.get("NACCAGE")) else None
    sex  = SEX_MAP.get(int(row["SEX"]), "unknown") if pd.notna(row.get("SEX")) else None
    race = RACE_MAP.get(int(row["RACE"]), "unknown") if pd.notna(row.get("RACE")) else None
    educ = int(row["EDUC"]) if pd.notna(row.get("EDUC")) and row.get("EDUC") not in (-4, 99) else None

    demo_parts = []
    if age:  demo_parts.append(f"{age}-year-old")
    if race: demo_parts.append(race)
    if sex:  demo_parts.append(sex)
    if educ: demo_parts.append(f"with {educ} years of education")
    if demo_parts:
        parts.append(" ".join(demo_parts) + ".")

    # CDR
    cdr_g = row.get("CDRGLOB")
    cdr_s = row.get("CDRSUM")
    if pd.notna(cdr_g) and float(cdr_g) >= 0:
        severity = CDR_SEVERITY.get(float(cdr_g), "impaired")
        cdr_text = f"CDR global {float(cdr_g):.1f} ({severity})"
        if pd.notna(cdr_s) and float(cdr_s) >= 0:
            cdr_text += f", CDR sum of boxes {float(cdr_s):.1f}"
        parts.append(cdr_text + ".")

    # 기억 서브스케일
    mem = row.get("MEMORY")
    if pd.notna(mem) and float(mem) >= 0:
        mem_desc = MEMORY_MAP.get(float(mem), f"{float(mem)}")
        parts.append(f"Memory domain: {mem_desc}.")

    # GDS (우울)
    gds = row.get("NACCGDS")
    if pd.notna(gds) and int(gds) not in (-4, -1, 88):
        parts.append(f"GDS depression score: {int(gds)}/15.")

    # 진단 (include_diagnosis=False 시 제외 → linear probe ablation 용)
    if include_diagnosis:
        diag_code = row.get("NACCUDSD")
        if pd.notna(diag_code) and int(diag_code) in NACCUDSD_MAP:
            parts.append(f"Clinical diagnosis: {NACCUDSD_MAP[int(diag_code)]}.")

    return " ".join(parts) if parts else "Clinical information not available."


# ── ADNI 텍스트 생성 ───────────────────────────────────────────────────────
ADNI_DIAG_MAP = {
    "CN": "cognitively normal",
    "MCI": "mild cognitive impairment",
    "AD": "Alzheimer's dementia",
    "Dementia": "dementia",
}

def _adni_row_to_text(row: pd.Series, include_diagnosis: bool = True) -> str:
    parts = []

    age = row.get("entry_age")
    if pd.notna(age):
        parts.append(f"{int(age)}-year-old patient.")

    cdrsb = row.get("CDRSB")
    if pd.notna(cdrsb):
        parts.append(f"CDR sum of boxes: {float(cdrsb):.1f}.")

    # 진단 및 research group (include_diagnosis=False 시 제외)
    if include_diagnosis:
        diag = row.get("diagnosis_cdrsb") or row.get("diagnosis")
        if pd.notna(diag):
            diag_text = ADNI_DIAG_MAP.get(str(diag).strip(), str(diag))
            parts.append(f"Clinical diagnosis: {diag_text}.")

        group = row.get("entry_research_group")
        if pd.notna(group) and str(group).strip() not in ("", "nan"):
            parts.append(f"Research group: {str(group).strip()}.")

    return " ".join(parts) if parts else "Clinical information not available."


# ── 메인 ──────────────────────────────────────────────────────────────────
def build_texts(output_path: Path, preview: int = 0,
                include_diagnosis: bool = True) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)

    records = []

    # ── NACC ──────────────────────────────────────────────────────────────
    nacc_df = pd.read_csv(NACC_CLINICAL, low_memory=False)
    nacc_ids = manifest[manifest["dataset"] == "nacc"]["subject_id"].tolist()

    nacc_matched = (
        nacc_df[nacc_df["NACCID"].isin(nacc_ids)]
        .sort_values("NACCVNUM")
        .groupby("NACCID")
        .last()
        .reset_index()
    )

    for _, row in nacc_matched.iterrows():
        text = _nacc_row_to_text(row, include_diagnosis=include_diagnosis)
        records.append({
            "subject_id": row["NACCID"],
            "dataset": "nacc",
            "clinical_text": text,
            # diagnosis 컬럼은 split/linear probe용으로 항상 보존
            "diagnosis": NACCUDSD_MAP.get(int(row["NACCUDSD"]), "unknown")
                         if pd.notna(row.get("NACCUDSD")) and int(row.get("NACCUDSD", -1)) in NACCUDSD_MAP
                         else "unknown",
        })

    print(f"NACC: {len(records)} subjects → 텍스트 생성 완료")

    # ── ADNI ──────────────────────────────────────────────────────────────
    adni_df = pd.read_csv(ADNI_CLINICAL, low_memory=False)
    adni_ids = manifest[manifest["dataset"] == "adni"]["subject_id"].tolist()

    adni_col = "subject_id" if "subject_id" in adni_df.columns else "PTID"
    adni_matched = (
        adni_df[adni_df[adni_col].isin(adni_ids)]
        .sort_values("image_date" if "image_date" in adni_df.columns else adni_col)
        .groupby(adni_col)
        .last()
        .reset_index()
    )

    nacc_n = len(records)
    for _, row in adni_matched.iterrows():
        text = _adni_row_to_text(row, include_diagnosis=include_diagnosis)
        sid = row[adni_col]
        records.append({
            "subject_id": sid,
            "dataset": "adni",
            "clinical_text": text,
            "diagnosis": str(row.get("diagnosis_cdrsb") or row.get("diagnosis") or "unknown"),
        })

    print(f"ADNI: {len(records) - nacc_n} subjects → 텍스트 생성 완료")
    print(f"총합: {len(records)} 페어")

    df = pd.DataFrame(records)

    # 출력
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n저장 완료: {output_path}")

    # 미리보기
    if preview > 0:
        print(f"\n── 샘플 {preview}개 ──")
        for _, r in df.sample(min(preview, len(df)), random_state=42).iterrows():
            print(f"[{r['subject_id']} / {r['dataset']} / {r['diagnosis']}]")
            print(f"  {r['clinical_text']}")
            print()

    # 통계
    print("── 텍스트 길이 통계 ──")
    lengths = df["clinical_text"].str.len()
    print(f"  평균: {lengths.mean():.0f}자  최소: {lengths.min()}자  최대: {lengths.max()}자")
    print(f"  빈 텍스트: {(lengths < 30).sum()}개")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/BrainCLIP/data/clinical_texts.csv"))
    parser.add_argument("--preview", type=int, default=5)
    parser.add_argument("--no_diagnosis", action="store_true",
                        help="진단 레이블 텍스트 제외 (EXP 02 ablation용)")
    args = parser.parse_args()
    build_texts(args.output, args.preview, include_diagnosis=not args.no_diagnosis)


if __name__ == "__main__":
    main()
