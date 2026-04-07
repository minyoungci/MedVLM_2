"""
build_longitudinal_manifest.py
Session-level longitudinal manifest for BrainCLIP training.

Each row = one MRI session with matched clinical variables.

Matching strategy:
  NACC  : ses-IXXXXXXX -> sort I-numbers per subject -> match to NACCVNUM 1,2,3... in order
           Clinical: commercial_nacc70.csv (VISITMO/DAY/YR, NACCAGE, CDRSUM, CDRGLOB, NACCGDS, NACCUDSD)
  ADNI  : ses-YYYYMMDD -> parse date -> match image_date in adni34_t1w_with_clinical.csv
           Clinical: CDRSB, diagnosis_cdrsb, entry_age
  OASIS : ses-dXXXX -> days=XXXX -> nearest CDR assessment within +-180 days
           Clinical: OASIS3_UDSb4_cdr.csv (MMSE, CDRSUM, CDRTOT) + OASIS3_UDSd1_diagnoses.csv

Usage:
    cd /home/vlm/minyoung2
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \\
        experiments/BrainCLIP/scripts/build_longitudinal_manifest.py \\
        --output /home/vlm/data/metadata/v4_longitudinal_manifest.csv

Output columns:
    subject_id, session_id, dataset, mri_path, diagnosis, age,
    cdr_global, cdr_sum, mmse, gds, visit_date, qc_status, split
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
LONG_ROOT   = Path("/home/vlm/data/preprocessed_v4/longitudinal")
V4_MANIFEST = Path("/home/vlm/data/metadata/v4_manifest.csv")

NACC_CLINICAL   = Path("/home/vlm/data/raw/NACC/NACC-Clinical/commercial_nacc70.csv")
NACC_MRI_DIR    = Path("/home/vlm/data/raw/NACC/MRI")

ADNI_CLINICAL   = Path("/home/vlm/data/raw/ADNI/adni34_t1w_with_cdrsb.csv")

OASIS_ZIP       = Path("/home/vlm/data/raw/oasis3/OASIS3_data_files.zip")
OASIS_CDR_PATH  = "OASIS3_data_files/scans/UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/resources/csv/files/OASIS3_UDSb4_cdr.csv"
OASIS_DX_PATH   = "OASIS3_data_files/scans/UDSd1-Form_D1__Clinician_Diagnosis___Cognitive_Status_and_Dementia/resources/csv/files/OASIS3_UDSd1_diagnoses.csv"
OASIS_GDS_PATH  = "OASIS3_data_files/scans/UDSb6-Form_B6__Behavioral_Assessment___GDS/resources/csv/files/OASIS3_UDSb6_gds.csv"
OASIS_DEMO_PATH = "OASIS3_data_files/scans/demo-demographics/resources/csv/files/OASIS3_demographics.csv"

# Within ±OASIS_MATCH_DAYS days of MRI session, find closest clinical visit
OASIS_MATCH_DAYS = 180

# ── NACC diagnosis mapping ─────────────────────────────────────────────────
# NACCUDSD: 1=CN, 2=MCI, 3=AD dementia, 4=AD+other, 5=non-AD dementia
NACCUDSD_TO_DX = {
    1: "CN",
    2: "MCI",
    3: "AD",
    4: "AD",
    5: "Other",
}

# ── ADNI diagnosis mapping ─────────────────────────────────────────────────
ADNI_DIAG_MAP = {
    "CN": "CN",
    "MCI": "MCI",
    "AD": "AD",
    "Dementia": "AD",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "SMC": "CN",
}

# ── QC status from qc_metrics.json ────────────────────────────────────────
import json


def _read_qc_status(session_dir: Path) -> str:
    qc_file = session_dir / "qc_metrics.json"
    if not qc_file.exists():
        return "UNKNOWN"
    try:
        with open(qc_file) as f:
            d = json.load(f)
        status = d.get("overall_status", "UNKNOWN")
        # Normalize to GOOD / MARGINAL / FAIL / UNKNOWN
        if status in ("PASS", "GOOD"):
            return "GOOD"
        elif status == "MARGINAL":
            return "MARGINAL"
        elif status in ("FAIL", "ERROR"):
            return "FAIL"
        return status
    except Exception:
        return "UNKNOWN"


# ── NACC ──────────────────────────────────────────────────────────────────

def _build_nacc_image_id_map() -> dict:
    """
    Parse SCAN_NACCXXXXXX_IXXXXXXX_*.zip filenames to build:
        {naccid: sorted list of int image_ids}
    Only T1w-like filenames are included.
    """
    t1w_keywords = {"MPRAGE", "T1", "t1", "mprage"}
    pattern = re.compile(r"SCAN_(NACC\d+)_I(\d+)_(.+)\.zip")
    mapping: dict[str, list[int]] = {}

    for fname in NACC_MRI_DIR.iterdir():
        m = pattern.match(fname.name)
        if m is None:
            continue
        naccid, image_id_str, scan_name = m.group(1), m.group(2), m.group(3)
        # Keep only T1w scans
        if not any(kw in scan_name for kw in t1w_keywords):
            continue
        mapping.setdefault(naccid, []).append(int(image_id_str))

    # Sort I-numbers per subject (ascending = chronological)
    for naccid in mapping:
        mapping[naccid].sort()

    return mapping


def build_nacc_records(v4_manifest: pd.DataFrame) -> list[dict]:
    """
    For each NACC session in longitudinal directory, match to clinical visit.
    Matching: rank I-numbers per subject -> map to NACCVNUM 1,2,3...
    """
    print("\n── NACC ─────────────────────────────────────────────────────────")

    # Load NACC clinical
    nacc_cli = pd.read_csv(
        NACC_CLINICAL, low_memory=False,
        usecols=["NACCID", "NACCVNUM", "VISITMO", "VISITDAY", "VISITYR",
                 "NACCAGE", "CDRGLOB", "CDRSUM", "NACCGDS", "NACCUDSD",
                 "SEX", "RACE", "EDUC", "BIRTHMO", "BIRTHYR"],
    )
    nacc_cli = nacc_cli.sort_values(["NACCID", "NACCVNUM"]).reset_index(drop=True)
    # Group by subject -> list of visit rows in NACCVNUM order
    nacc_visits: dict[str, pd.DataFrame] = {
        nid: grp.reset_index(drop=True)
        for nid, grp in nacc_cli.groupby("NACCID", sort=False)
    }

    # Build I-number order map from zip filenames
    image_id_map = _build_nacc_image_id_map()
    print(f"  Image-ID map: {len(image_id_map)} subjects with T1w zips")

    # Test-set subjects from cross-sectional manifest (preserve split)
    test_subjects = set(
        v4_manifest.loc[v4_manifest["split"] == "test", "subject_id"]
    )

    records = []
    no_clinical = 0
    no_image_map = 0
    ok = 0

    subj_dirs = sorted((LONG_ROOT / "NACC").iterdir())
    for subj_dir in subj_dirs:
        naccid = subj_dir.name
        if not subj_dir.is_dir():
            continue

        # Get sorted session dirs
        ses_dirs = sorted(subj_dir.iterdir(), key=lambda p: p.name)

        # Get clinical visit rows for this subject
        visits_df = nacc_visits.get(naccid)
        if visits_df is None:
            no_clinical += len(ses_dirs)
            continue

        # Get ordered I-numbers for this subject (T1w only)
        ordered_ids = image_id_map.get(naccid, [])

        for ses_dir in ses_dirs:
            if not ses_dir.is_dir():
                continue
            ses_name = ses_dir.name  # e.g. ses-I10386095
            # Extract I-number from session name
            m = re.match(r"ses-I?(\d+)", ses_name)
            if m is None:
                continue
            image_id_int = int(m.group(1))

            # Determine visit rank: position of image_id in sorted I-number list
            try:
                rank = ordered_ids.index(image_id_int)  # 0-based
            except ValueError:
                # Not in T1w map (might be FLAIR etc.) - use alphabetical order as fallback
                all_ses_names = sorted(
                    [re.match(r"ses-I?(\d+)", s.name).group(1)
                     for s in ses_dirs if re.match(r"ses-I?(\d+)", s.name)],
                    key=int
                )
                try:
                    rank = all_ses_names.index(str(image_id_int))
                except ValueError:
                    rank = 0

            # Match to NACCVNUM row (1-indexed, rank 0 -> row 0)
            visit_idx = min(rank, len(visits_df) - 1)
            visit_row = visits_df.iloc[visit_idx]

            # Build visit_date string
            mo = visit_row.get("VISITMO")
            dy = visit_row.get("VISITDAY")
            yr = visit_row.get("VISITYR")
            if pd.notna(mo) and pd.notna(dy) and pd.notna(yr):
                visit_date = f"{int(yr):04d}-{int(mo):02d}-{int(dy):02d}"
            else:
                visit_date = None

            # Diagnosis
            naccudsd = visit_row.get("NACCUDSD")
            if pd.notna(naccudsd) and int(naccudsd) in NACCUDSD_TO_DX:
                diagnosis = NACCUDSD_TO_DX[int(naccudsd)]
            else:
                diagnosis = "Unknown"

            # CDR
            cdr_g = visit_row.get("CDRGLOB")
            cdr_s = visit_row.get("CDRSUM")
            cdr_global = float(cdr_g) if pd.notna(cdr_g) and float(cdr_g) >= 0 else None
            cdr_sum = float(cdr_s) if pd.notna(cdr_s) and float(cdr_s) >= 0 else None

            # Age
            age = visit_row.get("NACCAGE")
            age = float(age) if pd.notna(age) and float(age) > 0 else None

            # GDS
            gds = visit_row.get("NACCGDS")
            gds = int(gds) if pd.notna(gds) and int(gds) not in (-4, -1, 88) else None

            # Split: keep test if subject was test in cross-sectional
            split = "test" if naccid in test_subjects else "train"

            # QC
            qc_status = _read_qc_status(ses_dir)

            mri_path = ses_dir / "native_t1w.nii.gz"
            if not mri_path.exists():
                continue

            records.append({
                "subject_id": naccid,
                "session_id": ses_name,
                "dataset": "nacc",
                "mri_path": str(mri_path),
                "diagnosis": diagnosis,
                "age": age,
                "cdr_global": cdr_global,
                "cdr_sum": cdr_sum,
                "mmse": None,  # Not in commercial_nacc70 at visit level
                "gds": gds,
                "visit_date": visit_date,
                "qc_status": qc_status,
                "split": split,
            })
            ok += 1

    print(f"  OK: {ok}  |  No clinical: {no_clinical}  |  No image map: {no_image_map}")
    return records


# ── ADNI ──────────────────────────────────────────────────────────────────

def build_adni_records(v4_manifest: pd.DataFrame) -> list[dict]:
    """
    Match ses-YYYYMMDD to image_date in adni34_t1w_with_clinical.csv.
    Sessions not in ADNI3/4 CSV get diagnosis=Unknown.
    """
    print("\n── ADNI ─────────────────────────────────────────────────────────")

    adni_cli = pd.read_csv(ADNI_CLINICAL, low_memory=False)
    # Build lookup: (subject_id, image_date) -> row
    adni_cli["image_date"] = pd.to_datetime(
        adni_cli["image_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    adni_lookup: dict[tuple, pd.Series] = {
        (row["subject_id"], row["image_date"]): row
        for _, row in adni_cli.iterrows()
        if pd.notna(row.get("image_date"))
    }

    test_subjects = set(
        v4_manifest.loc[v4_manifest["split"] == "test", "subject_id"]
    )

    records = []
    matched = 0
    unmatched = 0

    subj_dirs = sorted((LONG_ROOT / "ADNI").iterdir())
    for subj_dir in subj_dirs:
        subject_id = subj_dir.name
        if not subj_dir.is_dir():
            continue

        ses_dirs = sorted(subj_dir.iterdir(), key=lambda p: p.name)
        for ses_dir in ses_dirs:
            if not ses_dir.is_dir():
                continue
            ses_name = ses_dir.name  # ses-YYYYMMDD

            # Parse date from session name
            m = re.match(r"ses-(\d{4})(\d{2})(\d{2})$", ses_name)
            if m is None:
                continue
            image_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

            mri_path = ses_dir / "native_t1w.nii.gz"
            if not mri_path.exists():
                continue

            qc_status = _read_qc_status(ses_dir)
            split = "test" if subject_id in test_subjects else "train"

            key = (subject_id, image_date)
            if key in adni_lookup:
                row = adni_lookup[key]
                diag_raw = str(row.get("diagnosis_cdrsb") or row.get("diagnosis") or "").strip()
                # Treat empty/"nan" strings as Unknown
                if diag_raw in ("", "nan", "None"):
                    diagnosis = "Unknown"
                else:
                    diagnosis = ADNI_DIAG_MAP.get(diag_raw, diag_raw)

                cdrsb = row.get("CDRSB")
                cdr_sum = float(cdrsb) if pd.notna(cdrsb) else None

                age = row.get("entry_age")
                age = float(age) if pd.notna(age) else None

                matched += 1
            else:
                diagnosis = "Unknown"
                cdr_sum = None
                age = None
                unmatched += 1

            records.append({
                "subject_id": subject_id,
                "session_id": ses_name,
                "dataset": "adni",
                "mri_path": str(mri_path),
                "diagnosis": diagnosis,
                "age": age,
                "cdr_global": None,   # ADNI cdrsb CSV has CDR-SB (sum of boxes), not CDR global
                "cdr_sum": cdr_sum,
                "mmse": None,
                "gds": None,
                "visit_date": image_date,
                "qc_status": qc_status,
                "split": split,
            })

    print(f"  Matched (with CDRSB clinical data): {matched}  |  "
          f"Unmatched (ADNI1/2 era or no clinical): {unmatched}")
    return records


# ── OASIS ─────────────────────────────────────────────────────────────────

def _load_oasis_zip_csv(zf: zipfile.ZipFile, inner_path: str) -> pd.DataFrame:
    with zf.open(inner_path) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), low_memory=False)


def _oasis_diagnosis(d1_row: pd.Series) -> str:
    """Map OASIS D1 flags (float) to CN/MCI/AD/Unknown.
    Columns are float64: 1.0 = True, 0.0 = False, NaN = missing.
    """
    normcog  = d1_row.get("NORMCOG")
    demented = d1_row.get("DEMENTED")
    mciamem  = d1_row.get("MCIAMEM")
    impnomci = d1_row.get("IMPNOMCI")

    if pd.isna(normcog) and pd.isna(demented):
        return "Unknown"
    if pd.notna(normcog) and float(normcog) == 1.0:
        return "CN"
    if pd.notna(demented) and float(demented) == 1.0:
        return "AD"
    if (pd.notna(mciamem) and float(mciamem) == 1.0) or \
       (pd.notna(impnomci) and float(impnomci) == 1.0):
        return "MCI"
    # Has data but no positive flag -> likely CN with all zeros
    if pd.notna(normcog) and float(normcog) == 0.0 and pd.notna(demented) and float(demented) == 0.0:
        return "MCI"  # impaired but not classified as full dementia
    return "Unknown"


def build_oasis_records(v4_manifest: pd.DataFrame) -> list[dict]:
    """
    Match ses-dXXXX (days) to nearest OASIS clinical assessment within ±OASIS_MATCH_DAYS.
    """
    print("\n── OASIS ────────────────────────────────────────────────────────")

    # Load OASIS CSVs from zip
    with zipfile.ZipFile(OASIS_ZIP, "r") as zf:
        cdr_df  = _load_oasis_zip_csv(zf, OASIS_CDR_PATH)
        dx_df   = _load_oasis_zip_csv(zf, OASIS_DX_PATH)
        gds_df  = _load_oasis_zip_csv(zf, OASIS_GDS_PATH)
        demo_df = _load_oasis_zip_csv(zf, OASIS_DEMO_PATH)

    # Ensure days_to_visit is numeric
    for df in (cdr_df, dx_df, gds_df):
        df["days_to_visit"] = pd.to_numeric(df["days_to_visit"], errors="coerce")

    # Build per-subject clinical records indexed by days
    # CDR: OASISID, days_to_visit, age at visit, MMSE, CDRSUM, CDRTOT
    cdr_grp  = {oid: g.sort_values("days_to_visit") for oid, g in cdr_df.groupby("OASISID")}
    dx_grp   = {oid: g.sort_values("days_to_visit") for oid, g in dx_df.groupby("OASISID")}
    gds_grp  = {oid: g.sort_values("days_to_visit") for oid, g in gds_df.groupby("OASISID")}
    # Demographics (one row per subject)
    demo_idx = demo_df.set_index("OASISID") if "OASISID" in demo_df.columns else {}

    def _nearest_row(grp_df: pd.DataFrame, days: int, max_delta: int):
        """Return row with smallest |days_to_visit - days| within max_delta, or None."""
        deltas = (grp_df["days_to_visit"] - days).abs()
        idx = deltas.idxmin()
        if deltas[idx] <= max_delta:
            return grp_df.loc[idx]
        return None

    test_subjects = set(
        v4_manifest.loc[v4_manifest["split"] == "test", "subject_id"]
    )

    records = []
    matched = 0
    unmatched = 0

    subj_dirs = sorted((LONG_ROOT / "OASIS").iterdir())
    for subj_dir in subj_dirs:
        oasisid = subj_dir.name  # e.g. OAS30001
        if not subj_dir.is_dir():
            continue

        ses_dirs = sorted(subj_dir.iterdir(), key=lambda p: p.name)
        for ses_dir in ses_dirs:
            if not ses_dir.is_dir():
                continue
            ses_name = ses_dir.name  # ses-d0129

            m = re.match(r"ses-d(\d+)$", ses_name)
            if m is None:
                continue
            scan_days = int(m.group(1))

            mri_path = ses_dir / "native_t1w.nii.gz"
            if not mri_path.exists():
                continue

            qc_status = _read_qc_status(ses_dir)
            split = "test" if oasisid in test_subjects else "train"

            # Match CDR
            cdr_row = None
            if oasisid in cdr_grp:
                cdr_row = _nearest_row(cdr_grp[oasisid], scan_days, OASIS_MATCH_DAYS)

            # Match diagnosis
            dx_row = None
            if oasisid in dx_grp:
                dx_row = _nearest_row(dx_grp[oasisid], scan_days, OASIS_MATCH_DAYS)

            # Match GDS
            gds_row = None
            if oasisid in gds_grp:
                gds_row = _nearest_row(gds_grp[oasisid], scan_days, OASIS_MATCH_DAYS)

            if cdr_row is not None or dx_row is not None:
                matched += 1
            else:
                unmatched += 1

            # Extract values
            if cdr_row is not None:
                age = cdr_row.get("age at visit")
                age = float(age) if pd.notna(age) else None
                mmse = cdr_row.get("MMSE")
                mmse = int(mmse) if pd.notna(mmse) else None
                cdr_sum_val = cdr_row.get("CDRSUM")
                cdr_sum = float(cdr_sum_val) if pd.notna(cdr_sum_val) else None
                cdr_tot = cdr_row.get("CDRTOT")
                cdr_global = float(cdr_tot) if pd.notna(cdr_tot) else None
            else:
                # Fallback to demographics for age
                age, mmse, cdr_sum, cdr_global = None, None, None, None
                if isinstance(demo_idx, pd.DataFrame) and oasisid in demo_idx.index:
                    age_entry = demo_idx.loc[oasisid, "AgeatEntry"]
                    age = float(age_entry) if pd.notna(age_entry) else None

            gds_val = None
            if gds_row is not None:
                g = gds_row.get("GDS")
                gds_val = int(g) if pd.notna(g) else None

            diagnosis = _oasis_diagnosis(dx_row) if dx_row is not None else "Unknown"

            records.append({
                "subject_id": oasisid,
                "session_id": ses_name,
                "dataset": "oasis",
                "mri_path": str(mri_path),
                "diagnosis": diagnosis,
                "age": age,
                "cdr_global": cdr_global,
                "cdr_sum": cdr_sum,
                "mmse": mmse,
                "gds": gds_val,
                "visit_date": None,   # OASIS uses days-since-enrollment, not calendar date
                "qc_status": qc_status,
                "split": split,
            })

    print(f"  Matched: {matched}  |  Unmatched (no clinical within ±{OASIS_MATCH_DAYS}d): {unmatched}")
    return records


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build v4 longitudinal manifest CSV")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/home/vlm/data/metadata/v4_longitudinal_manifest.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["nacc", "adni", "oasis"],
        choices=["nacc", "adni", "oasis"],
        help="Datasets to include",
    )
    args = parser.parse_args()

    print(f"Building longitudinal manifest → {args.output}")
    print(f"Datasets: {args.datasets}")

    v4_manifest = pd.read_csv(V4_MANIFEST)
    print(f"v4_manifest loaded: {len(v4_manifest)} cross-sectional subjects")

    all_records: list[dict] = []

    if "nacc" in args.datasets:
        all_records.extend(build_nacc_records(v4_manifest))

    if "adni" in args.datasets:
        all_records.extend(build_adni_records(v4_manifest))

    if "oasis" in args.datasets:
        all_records.extend(build_oasis_records(v4_manifest))

    df = pd.DataFrame(all_records, columns=[
        "subject_id", "session_id", "dataset", "mri_path",
        "diagnosis", "age", "cdr_global", "cdr_sum", "mmse", "gds",
        "visit_date", "qc_status", "split",
    ])

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n════════════════════════════════════════════════════════════════")
    print(f"Total sessions: {len(df)}")
    print(f"\nDataset distribution:")
    print(df["dataset"].value_counts().to_string())
    print(f"\nDiagnosis distribution:")
    print(df["diagnosis"].value_counts().to_string())
    print(f"\nQC status distribution:")
    print(df["qc_status"].value_counts().to_string())
    print(f"\nSplit distribution:")
    print(df["split"].value_counts().to_string())

    # Coverage
    total = len(df)
    def pct(col):
        n = df[col].notna().sum()
        return f"{n}/{total} ({100*n/total:.1f}%)"

    print(f"\nClinical variable coverage:")
    print(f"  diagnosis != Unknown : {(df['diagnosis'] != 'Unknown').sum()}/{total} "
          f"({100*(df['diagnosis'] != 'Unknown').mean():.1f}%)")
    print(f"  age                 : {pct('age')}")
    print(f"  cdr_global          : {pct('cdr_global')}")
    print(f"  cdr_sum             : {pct('cdr_sum')}")
    print(f"  mmse                : {pct('mmse')}")
    print(f"  gds                 : {pct('gds')}")

    # Failure cases
    unknown_dx = (df["diagnosis"] == "Unknown").sum()
    no_age = df["age"].isna().sum()
    print(f"\nFailure cases:")
    print(f"  Unknown diagnosis   : {unknown_dx}")
    print(f"  Missing age         : {no_age}")
    print("════════════════════════════════════════════════════════════════")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
