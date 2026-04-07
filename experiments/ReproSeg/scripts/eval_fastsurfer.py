"""FastSurfer ICC Evaluation — compute scan-rescan reproducibility using FastSurferCNN.

Runs FastSurferCNN (segmentation only, no surface recon) on the same held-out
test pairs as eval_reproducibility.py, maps FreeSurfer labels → our grouped
structures, computes ICC(3,1) and CV%.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -u eval_fastsurfer.py \
        --output results/icc_fastsurfer.json \
        --cache-dir /tmp/fastsurfer_cache \
        --max-eval-pairs 60
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

SEED = 42
FASTSURFER_DIR = Path('/home/vlm/minyoung/tools/FastSurfer')
sys.path.insert(0, str(FASTSURFER_DIR))
sys.path.insert(0, '/home/vlm/minyoung/model-claude')
sys.path.insert(0, '/home/vlm/minyoung/model-claude/experiments/ros_temporal_consistency/scripts')

from src.data.seg_groups import GROUP_NAMES, NUM_GROUPED_CLASSES
from train_ros import build_longitudinal_pairs
from train_reproseg import PROGRESSION

# Same label mapping as SynthSeg (both use FreeSurfer DKT parcellation)
FS_TO_GROUP = {
    17: 1,   # hippocampus_L
    53: 2,   # hippocampus_R
    18: 3,   # amygdala_L
    54: 4,   # amygdala_R
    1006: 5, # entorhinal_L
    2006: 6, # entorhinal_R
    4: 7, 43: 7, 14: 7, 15: 7,   # ventricle
    2: 8, 41: 8,                   # white_matter
    3: 9,                          # cortical_L
    42: 10,                        # cortical_R
    11: 11, 50: 11, 12: 11, 51: 11, 13: 11, 52: 11, 10: 11, 49: 11, 26: 11, 58: 11,  # subcortical
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


def run_fastsurfer(vol_path: Path, out_dir: Path, device: str = 'cuda') -> Path:
    """Run FastSurferCNN seg-only on one volume; returns aseg segmentation path."""
    import hashlib
    path_hash = hashlib.md5(str(vol_path).encode()).hexdigest()[:12]
    seg_path = out_dir / f"{path_hash}_aparc_aseg.mgz"
    if seg_path.exists():
        return seg_path

    # FastSurferCNN run_prediction.py interface
    subj_dir = out_dir / path_hash
    subj_dir.mkdir(parents=True, exist_ok=True)
    # When CUDA_VISIBLE_DEVICES is set, subprocess sees the GPU as cuda:0
    inner_device = 'cuda:0' if device.startswith('cuda') else device
    cmd = [
        sys.executable,
        str(FASTSURFER_DIR / 'FastSurferCNN' / 'run_prediction.py'),
        '--t1',              str(vol_path),
        '--asegdkt_segfile', str(seg_path),
        '--sid',             path_hash,
        '--sd',              str(out_dir),
        '--device',          inner_device,
        '--batch_size',      '8',
    ]
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=600,
        cwd=str(FASTSURFER_DIR),
        env={**os.environ, 'PYTHONPATH': str(FASTSURFER_DIR)},
    )
    if result.returncode != 0 or not seg_path.exists():
        # Fallback: output may land in sid subdir
        for candidate in [
            out_dir / path_hash / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz',
            out_dir / path_hash / 'mri' / 'aseg.auto_noCC.mgz',
        ]:
            if candidate.exists():
                import shutil
                shutil.copy(candidate, seg_path)
                return seg_path
        raise RuntimeError(f"FastSurfer failed (rc={result.returncode}): {result.stderr[-500:]}")
    return seg_path


def seg_to_volumes(seg_path: Path, vox_vol: float) -> np.ndarray:
    """Map FreeSurfer labels → group volumes (mm³)."""
    img = nib.load(str(seg_path))
    seg = np.asarray(img.dataobj, dtype=np.int32)
    volumes = np.zeros(NUM_GROUPED_CLASSES, dtype=float)
    for fs_label, group_idx in FS_TO_GROUP.items():
        volumes[group_idx] += float((seg == fs_label).sum()) * vox_vol
    return volumes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',         default='results/icc_fastsurfer.json')
    parser.add_argument('--max-pairs',      type=int, default=2000)
    parser.add_argument('--max-eval-pairs', type=int, default=0,
                        help='Subsample N pairs from test set (0=all).')
    parser.add_argument('--cache-dir',      default='/tmp/fastsurfer_cache')
    parser.add_argument('--device',         default='cuda')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FastSurfer ICC Evaluation")
    print(f"  FastSurfer: {FASTSURFER_DIR}")
    print(f"  Device: {args.device}")
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
            hdr = nib.load(str(v1)).header
            vox_vol = float(np.prod(np.abs(hdr.get_zooms()[:3])))
            s1 = run_fastsurfer(v1, cache_dir, args.device)
            s2 = run_fastsurfer(v2, cache_dir, args.device)
            vols1 = seg_to_volumes(s1, vox_vol)
            vols2 = seg_to_volumes(s2, vox_vol)
            records.append({'vols1': vols1, 'vols2': vols2,
                             'subject_id': pair.get('subject_id', '')})
        except Exception as e:
            print(f"  [skip] pair {i}: {e}")
            continue
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_pairs)} pairs done...")

    if not records:
        print("ERROR: no pairs processed.")
        sys.exit(1)

    all_icc, all_cv = [], []
    per_structure = {}
    struct_names = list(GROUP_NAMES.values()) if hasattr(GROUP_NAMES, 'values') else [
        '', 'hippocampus_L','hippocampus_R','amygdala_L','amygdala_R',
        'entorhinal_L','entorhinal_R','ventricle','white_matter',
        'cortical_L','cortical_R','subcortical'
    ]
    for c in range(1, NUM_GROUPED_CLASSES):
        v1s = [r['vols1'][c] for r in records]
        v2s = [r['vols2'][c] for r in records]
        sub  = [(a, b) for a, b in zip(v1s, v2s) if a + b > 1.0]
        if len(sub) < 5:
            continue
        y1, y2  = zip(*sub)
        icc_val = icc_3_1(list(y1), list(y2))
        cv_val  = cv_percent(list(y1), list(y2))
        name = struct_names[c] if c < len(struct_names) else f'struct_{c}'
        per_structure[name] = {'icc': round(icc_val, 4), 'cv_pct': round(cv_val, 2), 'n': len(sub)}
        all_icc.append(icc_val)
        all_cv.append(cv_val)

    mean_icc = float(np.mean(all_icc))
    mean_cv  = float(np.mean(all_cv))

    print(f"\n  {'Structure':<24} {'ICC':>6}  {'CV%':>6}  {'n':>6}")
    print(f"  {'-'*50}")
    for name, v in per_structure.items():
        print(f"  {name:<24} {v['icc']:>6.4f}  {v['cv_pct']:>5.2f}%  {v['n']:>6}")
    print(f"\n  {'MEAN':<24} {mean_icc:>6.4f}  {mean_cv:>5.2f}%")

    result = [{
        'label':        'FastSurfer',
        'mean_icc':     round(mean_icc, 4),
        'mean_cv_pct':  round(mean_cv, 2),
        'per_structure': per_structure,
        'n_pairs':      len(records),
    }]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, 'w'), indent=2)
    print(f"\n  Saved → {args.output}")


if __name__ == '__main__':
    main()
