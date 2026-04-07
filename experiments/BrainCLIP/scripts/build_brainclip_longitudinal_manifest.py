"""
build_brainclip_longitudinal_manifest.py

BrainCLIP 실험 전용 longitudinal manifest 생성.

소스:
  - /home/vlm/data/metadata/v4_longitudinal_manifest.csv  (QC, NACC/OASIS 임상)
  - /home/vlm/data/raw/ADNI/clinical data/All_Subjects_CDR_02Nov2025.csv   (ADNI CDRSB, 98.3%)
  - /home/vlm/data/raw/ADNI/clinical data/All_Subjects_Study_Entry_02Nov2025.csv (ADNI baseline age)

주의:
  - 기존 manifest 파일 수정/덮어쓰기 금지
  - 출력: experiments/BrainCLIP/data/brainclip_longitudinal_manifest.csv (신규 생성)
  - 기존 cross-sectional manifest(v4_manifest.csv)와 subject 겹침 없음 (이미 분리됨)
  - 모든 세션은 split=train (test set은 cross-sectional로만 구성)

ADNI 임상 매칭 전략:
  - CDR: (subject_id, visit_date) ↔ (PTID, VISDATE) 정확 매칭 우선,
          실패 시 ±90일 내 가장 가까운 평가 선택
  - Age: Study Entry baseline age per subject (1회 측정, 모든 세션에 적용)

필터 기준:
  - QC FAILED 제외 (GOOD, MARGINAL 포함)
  - MRI 파일 실재 확인
  - 임상 정보 최소 1개 이상 (age 또는 cdr_sum)

Usage:
    cd /home/vlm/minyoung2
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
        experiments/BrainCLIP/scripts/build_brainclip_longitudinal_manifest.py
"""

from pathlib import Path
import pandas as pd

# ── 경로 상수 (읽기 전용) ──────────────────────────────────────────────────
BASE_MANIFEST  = Path("/home/vlm/data/metadata/v4_longitudinal_manifest.csv")
ADNI_CDR       = Path("/home/vlm/data/raw/ADNI/clinical data/All_Subjects_CDR_02Nov2025.csv")
ADNI_ENTRY     = Path("/home/vlm/data/raw/ADNI/clinical data/All_Subjects_Study_Entry_02Nov2025.csv")
OUTPUT_PATH    = Path("experiments/BrainCLIP/data/brainclip_longitudinal_manifest.csv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# CDR 매칭 허용 창 (일 단위)
CDR_WINDOW_DAYS = 90


def build_adni_cdr_lookup(cdr_csv: Path) -> pd.DataFrame:
    """
    ADNI CDR CSV 로드 → (PTID, VISDATE) 인덱스.
    PTID = manifest subject_id, VISDATE = YYYY-MM-DD.
    """
    df = pd.read_csv(cdr_csv)
    df["VISDATE"] = pd.to_datetime(df["VISDATE"], errors="coerce")
    df = df.dropna(subset=["PTID", "VISDATE"])
    print(f"[ADNI CDR] {len(df)} rows, {df['PTID'].nunique()} subjects")
    print(f"  CDRSB non-null: {df['CDRSB'].notna().sum()}/{len(df)} "
          f"({100*df['CDRSB'].notna().mean():.1f}%)")
    return df


def build_adni_age_lookup(entry_csv: Path) -> dict:
    """
    ADNI Study Entry CSV → {subject_id: entry_age} lookup.
    Baseline age per subject (used for all sessions).
    """
    df = pd.read_csv(entry_csv)
    lookup = dict(zip(df["subject_id"], df["entry_age"]))
    print(f"[ADNI Entry] {len(lookup)} subjects with baseline age")
    return lookup


def find_closest_cdr(subject_rows: pd.DataFrame, visit_date: pd.Timestamp,
                     window_days: int = CDR_WINDOW_DAYS):
    """
    한 subject의 CDR 행들 중 visit_date에 가장 가까운 행 반환.
    window_days 초과 시 None.
    """
    if subject_rows.empty:
        return None
    deltas = (subject_rows["VISDATE"] - visit_date).abs()
    idx = deltas.idxmin()
    if deltas[idx].days <= window_days:
        return subject_rows.loc[idx]
    return None


def main():
    print(f"소스: {BASE_MANIFEST}")
    base = pd.read_csv(BASE_MANIFEST)
    print(f"  총 {len(base)} 세션 로드")

    # ── 1. ADNI 임상 데이터 준비 ─────────────────────────────────────────────
    cdr_df    = build_adni_cdr_lookup(ADNI_CDR)
    age_lookup = build_adni_age_lookup(ADNI_ENTRY)

    # CDR를 subject별 그룹으로 미리 분류 (속도)
    cdr_by_subject = {ptid: grp for ptid, grp in cdr_df.groupby("PTID")}

    # ── 2. ADNI 세션에 임상 정보 주입 ────────────────────────────────────────
    adni_mask = base["dataset"] == "adni"
    adni_rows = base[adni_mask].copy()
    non_adni  = base[~adni_mask].copy()

    cdr_matched = 0
    age_matched = 0

    for idx, row in adni_rows.iterrows():
        sid  = row["subject_id"]
        vdate = pd.to_datetime(str(row.get("visit_date", "")), errors="coerce")

        # ── CDR 매칭 ────────────────────────────────────────────────────────
        if sid in cdr_by_subject and pd.notna(vdate):
            match = find_closest_cdr(cdr_by_subject[sid], vdate)
            if match is not None:
                adni_rows.at[idx, "cdr_sum"]    = match.get("CDRSB")
                adni_rows.at[idx, "cdr_global"] = match.get("CDGLOBAL")
                cdr_matched += 1

        # ── Baseline age 주입 ────────────────────────────────────────────────
        if sid in age_lookup and pd.isna(row.get("age")):
            adni_rows.at[idx, "age"] = age_lookup[sid]
            age_matched += 1

    print(f"[ADNI] CDR 매칭: {cdr_matched}/{len(adni_rows)} 세션")
    print(f"[ADNI] Age 주입: {age_matched}/{len(adni_rows)} 세션 (baseline)")

    df = pd.concat([adni_rows, non_adni], ignore_index=True)

    # ── 3. 품질 필터 ────────────────────────────────────────────────────────
    n_before = len(df)
    df = df[df["qc_status"] != "FAILED"].copy()
    print(f"\nQC FAILED 제외: {n_before} → {len(df)}")

    # ── 4. MRI 파일 실재 확인 ───────────────────────────────────────────────
    df["mri_exists"] = df["mri_path"].apply(lambda p: Path(p).exists())
    missing = (~df["mri_exists"]).sum()
    if missing > 0:
        print(f"MRI 파일 없음 {missing}건 제외")
    df = df[df["mri_exists"]].drop(columns=["mri_exists"])

    # ── 5. 임상 정보 최소 필터 (age or cdr_sum 중 하나 이상) ─────────────────
    has_clinical = df["age"].notna() | df["cdr_sum"].notna()
    n_before = len(df)
    df = df[has_clinical].copy()
    print(f"임상 정보 없음 제외: {n_before} → {len(df)}")

    # ── 6. split=train 강제 ─────────────────────────────────────────────────
    df["split"] = "train"

    # ── 7. 통계 출력 ────────────────────────────────────────────────────────
    print("\n=== 최종 Manifest 통계 ===")
    print(f"총 세션: {len(df)}")
    print(f"\n데이터셋별:")
    print(df["dataset"].value_counts().to_string())
    print(f"\nQC 분포:")
    print(df["qc_status"].value_counts().to_string())
    print(f"\n진단 분포:")
    print(df["diagnosis"].fillna("Unknown").value_counts().to_string())
    print(f"\n임상 변수 coverage:")
    for col in ["age", "cdr_global", "cdr_sum", "gds", "mmse"]:
        n = df[col].notna().sum()
        print(f"  {col}: {n}/{len(df)} ({100*n/len(df):.1f}%)")

    # ── 8. 저장 ─────────────────────────────────────────────────────────────
    cols = ["subject_id", "session_id", "dataset", "mri_path",
            "diagnosis", "age", "cdr_global", "cdr_sum", "gds", "mmse",
            "visit_date", "qc_status", "split"]
    df[cols].to_csv(OUTPUT_PATH, index=False)
    print(f"\n저장 완료: {OUTPUT_PATH}")
    print(f"  (기존 v4_manifest.csv, v4_longitudinal_manifest.csv 수정 없음)")


if __name__ == "__main__":
    main()
