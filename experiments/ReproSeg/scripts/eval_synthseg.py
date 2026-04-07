"""SynthSeg ICC Evaluation — compute scan-rescan reproducibility using SynthSeg.

Runs SynthSeg on the same held-out 270 test pairs as eval_reproducibility.py,
maps FreeSurfer labels → our grouped structures, computes ICC(3,1) and CV%.

Usage:
    FREESURFER_HOME=/home/vlm/hyerin/tools/freesurfer \
    CUDA_VISIBLE_DEVICES=0 python eval_synthseg.py \
        --output results/icc_synthseg.json
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, '/home/vlm/minyoung/model-claude')
sys.path.insert(0, '/home/vlm/minyoung/model-claude/experiments/ros_temporal_consistency/scripts')

from src.data.seg_groups import GROUP_NAMES, NUM_GROUPED_CLASSES
from train_ros import build_longitudinal_pairs
from train_reproseg import PROGRESSION

FREESURFER_HOME = Path(os.environ.get('FREESURFER_HOME',
                                       '/home/vlm/hyerin/tools/freesurfer'))
SYNTHSEG_BIN    = FREESURFER_HOME / 'bin' / 'mri_synthseg'
FS_LICENSE      = Path('/home/jovyan/.license')

# FreeSurfer label → our group index
# SynthSeg outputs standard FreeSurfer parcellation labels
FS_TO_GROUP = {
    # hippocampus_L (1)
    17: 1,
    # hippocampus_R (2)
    53: 2,
    # amygdala_L (3)
    18: 3,
    # amygdala_R (4)
    54: 4,
    # entorhinal_L (5) — entorhinal cortex
    1006: 5,
    # entorhinal_R (6)
    2006: 6,
    # ventricle (7) — lateral + 3rd ventricle
    4: 7, 43: 7, 14: 7, 15: 7,
    # white_matter (8)
    2: 8, 41: 8,
    # cortical_L (9) — lh cortical ribbon
    3: 9,
    # cortical_R (10) — rh cortical ribbon
    42: 10,
    # subcortical (11) — caudate, putamen, pallidum, thalamus, accumbens
    11: 11, 50: 11, 12: 11, 51: 11, 13: 11, 52: 11,
    10: 11, 49: 11, 26: 11, 58: 11,
}


def icc_3_1(y1, y2):
    y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
    n = len(y1)
    if n < 2:
        return float('nan')
    mean_s = (y1 + y2) / 2.0
    grand  = mean_s.mean()
    SS_b   = 2.0 * np.sum((mean_s - grand) ** 2)
    SS_w   = np.sum((y1 - mean_s) ** 2 + (y2 - mean_s) ** 2)
    MS_b   = SS_b / (n - 1)
    MS_w   = SS_w / n
    denom  = MS_b + MS_w
    if denom < 1e-12:
        return 1.0
    return float((MS_b - MS_w) / denom)


def cv_percent(y1, y2):
    y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
    mean = (y1 + y2) / 2.0 + 1e-6
    return float(np.mean(np.abs(y1 - y2) / mean) * 100.0)


def get_test_pairs(max_pairs=2000):
    progression_df = pd.read_csv(str(PROGRESSION))
    random.seed(SEED)
    np.random.seed(SEED)
    all_pairs = build_longitudinal_pairs(progression_df, max_pairs=max_pairs)
    pair_subjects = sorted({p['subject_id'] for p in all_pairs})
    np.random.shuffle(pair_subjects)
    n_train = int(0.8 * len(pair_subjects))
    test_ids = set(pair_subjects[n_train:])
    test_pairs = [p for p in all_pairs if p.get('subject_id', '') in test_ids]
    print(f"  Test pairs: {len(test_pairs)} ({len(test_ids)} subjects)")
    return test_pairs


def run_synthseg(vol_path: Path, out_dir: Path) -> Path:
    """Run SynthSeg on one volume, return segmentation path."""
    # Use hash of full path to avoid collisions when stems are identical (e.g. mni_t1w)
    import hashlib
    path_hash = hashlib.md5(str(vol_path).encode()).hexdigest()[:12]
    out_path = out_dir / f"{path_hash}_synthseg.nii.gz"
    if out_path.exists():
        return out_path  # cache hit

    env = os.environ.copy()
    env['FREESURFER_HOME'] = str(FREESURFER_HOME)
    env['FS_LICENSE']      = str(FS_LICENSE)
    env['PATH']            = f"{FREESURFER_HOME}/bin:{env.get('PATH','')}"

    cmd = [str(SYNTHSEG_BIN), '--i', str(vol_path), '--o', str(out_path),
           '--threads', '4']
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"SynthSeg failed: {result.stderr[-500:]}")
    return out_path


def seg_to_volumes(seg_path: Path, vox_vol: float) -> np.ndarray:
    """Map FreeSurfer labels → group volumes (mm³)."""
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32).astype(np.int32)
    volumes = np.zeros(NUM_GROUPED_CLASSES, dtype=float)
    for fs_label, group_idx in FS_TO_GROUP.items():
        volumes[group_idx] += float((seg == fs_label).sum()) * vox_vol
    return volumes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/icc_synthseg.json')
    parser.add_argument('--max-pairs', type=int, default=2000)
    parser.add_argument('--max-eval-pairs', type=int, default=0,
                        help='Subsample this many pairs from test set (0 = all). '
                             'Uses SEED for reproducibility.')
    parser.add_argument('--cache-dir', default='/tmp/synthseg_cache')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SynthSeg Reproducibility Evaluation")
    print(f"  FreeSurfer: {FREESURFER_HOME}")
    print(f"{'='*60}")

    test_pairs = get_test_pairs(args.max_pairs)
    if args.max_eval_pairs > 0 and len(test_pairs) > args.max_eval_pairs:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(test_pairs), args.max_eval_pairs, replace=False)
        test_pairs = [test_pairs[i] for i in sorted(idx)]
        print(f"  Subsampled to {len(test_pairs)} pairs (--max-eval-pairs)")

    records = []
    for i, pair in enumerate(test_pairs):
        v1 = Path(pair['vol1'])
        v2 = Path(pair['vol2'])
        if not v1.exists() or not v2.exists():
            continue

        try:
            # Voxel size from header
            img1 = nib.load(str(v1))
            vox_vol = float(np.prod(img1.header.get_zooms()[:3]))

            seg1_path = run_synthseg(v1, cache_dir)
            seg2_path = run_synthseg(v2, cache_dir)

            vol1 = seg_to_volumes(seg1_path, vox_vol)
            vol2 = seg_to_volumes(seg2_path, vox_vol)

            for c in range(1, NUM_GROUPED_CLASSES):
                records.append({
                    'subject_id': pair.get('subject_id', f'subj_{i}'),
                    'class_idx': c,
                    'class_name': GROUP_NAMES.get(c, f'class_{c}'),
                    'vol1_mm3': vol1[c],
                    'vol2_mm3': vol2[c],
                })
        except Exception as e:
            print(f"  [skip] pair {i}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_pairs)} pairs done...")

    df = pd.DataFrame(records)
    struct_results, all_icc, all_cv = {}, [], []

    print(f"\n  {'Structure':<20} {'ICC':>7} {'CV%':>7}")
    print(f"  {'-'*38}")
    for c in range(1, NUM_GROUPED_CLASSES):
        name = GROUP_NAMES.get(c, f'class_{c}')
        sub  = df[df['class_idx'] == c]
        if len(sub) < 5:
            continue
        icc = icc_3_1(sub['vol1_mm3'].values, sub['vol2_mm3'].values)
        cv  = cv_percent(sub['vol1_mm3'].values, sub['vol2_mm3'].values)
        struct_results[name] = {'icc': round(icc, 4), 'cv_pct': round(cv, 2), 'n': len(sub)}
        all_icc.append(icc)
        all_cv.append(cv)
        print(f"  {name:<20} {icc:>7.4f} {cv:>7.2f}%")

    mean_icc = float(np.mean(all_icc)) if all_icc else float('nan')
    mean_cv  = float(np.mean(all_cv))  if all_cv  else float('nan')
    print(f"\n  {'MEAN':<20} {mean_icc:>7.4f} {mean_cv:>7.2f}%")

    result = {
        'label': 'SynthSeg',
        'mean_icc': round(mean_icc, 4),
        'mean_cv_pct': round(mean_cv, 2),
        'per_structure': struct_results,
        'n_pairs': len(test_pairs),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump([result], f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == '__main__':
    main()
