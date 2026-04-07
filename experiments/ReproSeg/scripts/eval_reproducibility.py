"""Reproducibility Evaluation: ICC and CV for brain segmentation.

Computes scan-rescan reproducibility metrics on held-out longitudinal pairs.
Compares multiple model checkpoints (e.g., A-baseline vs F-full).

Usage:
    # Single model
    CUDA_VISIBLE_DEVICES=0 uv run python eval_reproducibility.py \
        --checkpoints A:/path/to/best.pt

    # Compare multiple models
    CUDA_VISIBLE_DEVICES=0 uv run python eval_reproducibility.py \
        --checkpoints A:results/reproseg_v1_A_baseline/checkpoints/best.pt \
                      F:results/reproseg_v1_F_full_pcgrad/checkpoints/best.pt \
        --output results/icc_comparison.json

Metrics:
    ICC(3,1): Intraclass Correlation Coefficient (two-way mixed, absolute agreement)
              >0.90 = excellent, 0.75-0.90 = good, 0.50-0.75 = moderate
    CV:       Coefficient of Variation (%) = |v1-v2| / mean(v1,v2) × 100
              Lower is better. Clinical target: <5% for large structures.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MINYOUNG = Path('/home/vlm/minyoung/model-claude')
sys.path.insert(0, str(MINYOUNG))
sys.path.insert(0, str(MINYOUNG / 'experiments' / 'ros_temporal_consistency' / 'scripts'))
sys.path.insert(0, str(MINYOUNG / 'experiments' / 'vasnet' / 'scripts'))

from src.data.seg_groups import NUM_GROUPED_CLASSES, GROUP_NAMES
from train_ros import build_longitudinal_pairs, norm_zs, _pad
from train_vasnet import VOL_CLASSES, K, LR_SPLIT
from reproseg import ReproSeg, count_params
from train_reproseg import PROGRESSION, MANIFEST, CKPT_PATH, SITE_MAP, N_SITES

# Monai SwinUNETR
from monai.networks.nets import SwinUNETR

EXP_BASE  = Path(__file__).resolve().parent.parent
LONG_DIR  = Path('/home/vlm/data/preprocessed_v4/longitudinal')
PAD_TO    = (192, 224, 192)
SEED      = 42


# =============================================================================
# ICC / CV helpers
# =============================================================================

def icc_3_1(y1: np.ndarray, y2: np.ndarray) -> float:
    """ICC(3,1): two-way mixed effects, absolute agreement, single measures.

    Formula (Shrout & Fleiss 1979, Case 3):
        ICC = (MSb - MSw) / (MSb + MSw)
    where MSb = between-subjects mean square, MSw = within-subjects mean square
    for k=2 measurements.
    """
    y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
    n = len(y1)
    if n < 2:
        return float('nan')
    mean_s = (y1 + y2) / 2.0
    grand  = mean_s.mean()
    SS_b   = 2.0 * np.sum((mean_s - grand) ** 2)         # between subjects
    SS_w   = np.sum((y1 - mean_s) ** 2 + (y2 - mean_s) ** 2)  # within subjects
    MS_b   = SS_b / (n - 1)
    MS_w   = SS_w / n
    denom  = MS_b + MS_w
    if denom < 1e-12:
        return 1.0
    return float((MS_b - MS_w) / denom)


def cv_percent(y1: np.ndarray, y2: np.ndarray) -> float:
    """Mean coefficient of variation (%) across subject pairs."""
    y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
    mean   = (y1 + y2) / 2.0 + 1e-6
    return float(np.mean(np.abs(y1 - y2) / mean) * 100.0)


# =============================================================================
# Model loading
# =============================================================================

def build_model(device: torch.device) -> ReproSeg:
    """Build ReproSeg with same architecture as training."""
    from train_vasnet import LRAwareVolumeHead
    backbone = SwinUNETR(
        in_channels=1, out_channels=NUM_GROUPED_CLASSES,
        feature_size=48, use_checkpoint=False, spatial_dims=3,
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
    ).to(device)
    volume_head = LRAwareVolumeHead(in_ch=768, K=K).to(device)
    model = ReproSeg(backbone, n_sites=N_SITES, volume_head=volume_head).to(device)
    return model


def load_checkpoint(ckpt_path: Path, device: torch.device) -> ReproSeg:
    model = build_model(device)
    ckpt  = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}")
    model.eval()
    return model


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def run_inference(model: ReproSeg, vol_path: Path,
                  device: torch.device) -> np.ndarray:
    """Load a scan, run segmentation, return per-class voxel volumes (mm³)."""
    img  = nib.load(str(vol_path))
    data = img.get_fdata(dtype=np.float32)
    vox_vol = float(np.prod(img.header.get_zooms()[:3]))  # mm³ per voxel

    vol = norm_zs(data)
    vt  = torch.from_numpy(vol).float().unsqueeze(0)       # (1, H, W, D)
    vt  = _pad(vt, PAD_TO).unsqueeze(0).to(device)         # (1,1, H,W,D)

    with autocast('cuda', dtype=torch.bfloat16):
        outputs = model(vt, return_features=False)
        seg_logits = outputs[0]                            # (1, C, H, W, D)

    pred = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W, D)
    volumes = np.array([
        float((pred == c).sum()) * vox_vol
        for c in range(NUM_GROUPED_CLASSES)
    ])
    return volumes  # shape: (NUM_GROUPED_CLASSES,)


# =============================================================================
# Evaluation
# =============================================================================

def get_test_pairs(max_pairs: int = 2000):
    """Reconstruct held-out test pairs using same seed as training."""
    progression_df = pd.read_csv(str(PROGRESSION))
    random.seed(SEED)       # seeds Python random (used in build_longitudinal_pairs)
    np.random.seed(SEED)
    all_pairs = build_longitudinal_pairs(progression_df, max_pairs=max_pairs)
    pair_subjects = sorted({p['subject_id'] for p in all_pairs})  # sorted: deterministic
    np.random.shuffle(pair_subjects)

    n_train = int(0.8 * len(pair_subjects))
    test_pair_ids = set(pair_subjects[n_train:])          # held-out 20%

    test_pairs = [p for p in all_pairs
                  if p.get('subject_id', '') in test_pair_ids]
    print(f"  Test pairs: {len(test_pairs)} "
          f"({len(test_pair_ids)} subjects, held-out 20%)")
    return test_pairs


def evaluate_model(label: str, ckpt_path: Path, test_pairs: list,
                   device: torch.device) -> dict:
    """Run reproducibility evaluation for one checkpoint."""
    print(f"\n[{label}] Loading {ckpt_path.name}...")
    model = load_checkpoint(ckpt_path, device)

    records = []
    for i, pair in enumerate(test_pairs):
        v1_path = Path(pair['vol1'])
        v2_path = Path(pair['vol2'])
        if not v1_path.exists() or not v2_path.exists():
            continue

        vol1 = run_inference(model, v1_path, device)
        vol2 = run_inference(model, v2_path, device)

        for c in range(1, NUM_GROUPED_CLASSES):   # skip background
            records.append({
                'subject_id': pair.get('subject_id', f'subj_{i}'),
                'class_idx':  c,
                'class_name': GROUP_NAMES.get(c, f'class_{c}'),
                'vol1_mm3':   vol1[c],
                'vol2_mm3':   vol2[c],
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_pairs)} pairs processed...")

    df = pd.DataFrame(records)

    # Per-structure metrics
    struct_results = {}
    all_icc, all_cv = [], []
    print(f"\n  {'Structure':<25} {'ICC':>7} {'CV%':>7} {'n_pairs':>8}")
    print(f"  {'-'*50}")
    for c in range(1, NUM_GROUPED_CLASSES):
        name = GROUP_NAMES.get(c, f'class_{c}')
        sub  = df[df['class_idx'] == c]
        if len(sub) < 5:
            continue
        icc = icc_3_1(sub['vol1_mm3'].values, sub['vol2_mm3'].values)
        cv  = cv_percent(sub['vol1_mm3'].values, sub['vol2_mm3'].values)
        struct_results[name] = {'icc': round(icc, 4), 'cv_pct': round(cv, 2),
                                'n': len(sub)}
        all_icc.append(icc)
        all_cv.append(cv)
        print(f"  {name:<25} {icc:>7.4f} {cv:>7.2f}% {len(sub):>8}")

    mean_icc = float(np.mean(all_icc)) if all_icc else float('nan')
    mean_cv  = float(np.mean(all_cv))  if all_cv  else float('nan')
    print(f"\n  {'MEAN':<25} {mean_icc:>7.4f} {mean_cv:>7.2f}%")

    return {
        'label':      label,
        'ckpt':       str(ckpt_path),
        'mean_icc':   round(mean_icc, 4),
        'mean_cv_pct':round(mean_cv,  2),
        'per_structure': struct_results,
        'n_pairs':    len(test_pairs),
    }


# =============================================================================
# Comparison report
# =============================================================================

def compare_results(results: list[dict]) -> None:
    """Print delta table vs first checkpoint (baseline)."""
    if len(results) < 2:
        return
    baseline = results[0]
    print(f"\n{'='*60}")
    print(f"  Delta vs {baseline['label']} (baseline)")
    print(f"  {'Model':<15} {'ΔICC':>8} {'ΔCV%':>8} {'Verdict'}")
    print(f"  {'-'*55}")
    for r in results[1:]:
        d_icc = r['mean_icc']   - baseline['mean_icc']
        d_cv  = r['mean_cv_pct']- baseline['mean_cv_pct']
        verdict = ('✓ BETTER' if d_icc > 0.01 and d_cv < -0.5 else
                   '≈ SIMILAR' if abs(d_icc) <= 0.01 else
                   '✗ WORSE')
        print(f"  {r['label']:<15} {d_icc:>+8.4f} {d_cv:>+8.2f}%  {verdict}")
    print(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True,
                   metavar='LABEL:PATH',
                   help='Checkpoints to evaluate, format: LABEL:path/best.pt')
    p.add_argument('--output', type=str,
                   default=str(EXP_BASE / 'results' / 'icc_comparison.json'),
                   help='Output JSON path')
    p.add_argument('--max-pairs', type=int, default=2000)
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


def main():
    args  = parse_args()
    device = torch.device(args.device)

    # Parse checkpoint specs
    specs = []
    for spec in args.checkpoints:
        if ':' not in spec:
            raise ValueError(f"Format must be LABEL:PATH, got: {spec}")
        label, path = spec.split(':', 1)
        specs.append((label, Path(path)))

    print(f"\n{'='*60}")
    print(f"  ReproSeg Reproducibility Evaluation")
    print(f"  Device: {device}")
    print(f"  Models: {[s[0] for s in specs]}")
    print(f"{'='*60}")

    # Fix non-determinism: seed torch before inference
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    test_pairs = get_test_pairs(args.max_pairs)
    if not test_pairs:
        print("ERROR: No test pairs found. Check PROGRESSION path.")
        sys.exit(1)

    all_results = []
    for label, ckpt_path in specs:
        if not ckpt_path.exists():
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_path}")
            continue
        result = evaluate_model(label, ckpt_path, test_pairs, device)
        all_results.append(result)

    compare_results(all_results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()
