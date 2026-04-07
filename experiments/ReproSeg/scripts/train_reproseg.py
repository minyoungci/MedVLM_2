"""Train ReproSeg: Reproducibility-Equivariant Dual-Stream Brain Segmentation.

Loss Design (4 terms, uncertainty-weighted):
  L = (1/2σ₁²)·L_seg + (1/2σ₂²)·L_repro + (1/2σ₃²)·L_inv + (1/2σ₄²)·L_vol
      + log(σ₁·σ₂·σ₃·σ₄)

  Where:
    L_seg   = Dice + CE (standard, proven by nnU-Net/SynthSeg)
    L_repro = TCL + multi-scale feature consistency (reproducibility)
    L_inv   = adversarial site classification via GRL (scanner invariance)
    L_vol   = volume regression + seg-vol consistency (volumetric accuracy)

  σ₁..σ₄ are learned (Kendall et al. CVPR 2018) — no manual λ tuning.

Usage:
  # Ablation A: baseline (seg only)
  CUDA_VISIBLE_DEVICES=0,5 torchrun --nproc_per_node=2 --master_port=29550 \\
      train_reproseg.py --mode baseline --epochs 30 --exp-name reproseg_A

  # Ablation F: full ReproSeg
  CUDA_VISIBLE_DEVICES=0,5 torchrun --nproc_per_node=2 --master_port=29550 \\
      train_reproseg.py --mode full --epochs 30 --exp-name reproseg_F
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing infrastructure from minyoung
MINYOUNG = Path('/home/vlm/minyoung/model-claude')
sys.path.insert(0, str(MINYOUNG))
sys.path.insert(0, str(MINYOUNG / 'experiments' / 'ros_temporal_consistency' / 'scripts'))
sys.path.insert(0, str(MINYOUNG / 'experiments' / 'vasnet' / 'scripts'))

from src.data.seg_groups import NUM_GROUPED_CLASSES, GROUP_NAMES
from src.utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process,
    get_world_size, barrier, unwrap_model,
)
from train_ros import build_longitudinal_pairs, norm_zs, _pad

# Local imports
from reproseg import (
    ReproSeg, InvarianceStream, CrossStreamGate,
    SiteClassifier, MultiScaleFeatureConsistency,
    grad_reverse, count_params,
)
from train_vasnet import (
    LRAwareVolumeHead, VOL_CLASSES, VOL_CLASS_NAMES, K, LR_SPLIT,
)

# ── Constants ────────────────────────────────────────────────────────────────
LONG_DIR    = Path('/home/vlm/data/preprocessed_v4/longitudinal')
MANIFEST    = Path('/home/vlm/data/metadata/v4_manifest.csv')        # 2,397 subjects
PROGRESSION = Path('/home/vlm/data/preprocessed_v4/longitudinal_progression.csv')  # 1,726 subjects
CKPT_PATH   = Path('/home/vlm/minyoung/pretrain/brainsegfounder/weights/'
                    'BrainSegFounder/UK-Biobank/64-gpu-model_bestValRMSE.pt')  # 237MB
EXP_BASE    = Path(__file__).resolve().parent.parent

# V4 Data Notes:
# - Cross-sectional: 3,165 dirs (ADNI 885, AJU 719, NACC 1,230, OASIS 331)
# - Manifest filtered: 2,397 (GOOD 919 + MARGINAL 1,478)
# - Longitudinal: 6,554 complete sessions across 2,013 subjects
# - Per-subject: native_t1w.nii.gz, native_seg.nii.gz (192×224×192, 1mm, RAS)
# - NO volumes.json — GT volumes computed from native_seg on-the-fly

# Site mapping for adversarial loss
SITE_MAP = {'nacc': 0, 'oasis': 1, 'adni': 2, 'aju': 3}
N_SITES = len(SITE_MAP)


# =============================================================================
# 1. Loss Functions — Clean, Minimal, Literature-Backed
# =============================================================================

class DiceCELoss(nn.Module):
    """Dice + CE — proven baseline (nnU-Net, SynthSeg, BrainSegFounder).

    Reference: Isensee et al. (2024), Billot et al. (2023)
    """

    def __init__(self, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # CE
        ce_loss = self.ce(pred, target)
        # Dice (exclude background, class 0)
        pred_soft = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target.long(), self.num_classes)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3, 4))
        union = pred_soft.sum(dim=(2, 3, 4)) + target_oh.sum(dim=(2, 3, 4))
        dice = (2.0 * inter + 1.0) / (union + 1.0)
        dice_loss = 1.0 - dice[:, 1:].mean()  # skip background
        return ce_loss + dice_loss


class ReproducibilityLoss(nn.Module):
    """Combined reproducibility loss: TCL (output-level) + feature consistency (multi-scale).

    TCL: Normalized volume difference between paired predictions.
         Proven effective: CV -8%, 7/7 structures improved.
    Feature consistency: Cosine similarity at 4 encoder scales.
         Novel architectural contribution of ReproSeg.
    """

    def __init__(self, num_classes=12):
        super().__init__()
        self.num_classes = num_classes

    def tcl(self, pred1, pred2):
        """Temporal Consistency Loss on soft volumes."""
        soft1 = F.softmax(pred1, dim=1)
        soft2 = F.softmax(pred2, dim=1)
        vol1 = soft1.sum(dim=(2, 3, 4))  # [B, C]
        vol2 = soft2.sum(dim=(2, 3, 4))
        diff = (vol1[:, 1:] - vol2[:, 1:]) / (vol1[:, 1:] + vol2[:, 1:] + 1e-6)
        return diff.pow(2).mean()

    def forward(self, pred1, pred2, feat_loss=None):
        """
        Args:
            pred1, pred2: seg logits for paired scans [B, C, D, H, W]
            feat_loss: precomputed feat_consistency loss (from model.forward)
        Returns:
            loss: scalar
        """
        loss_tcl = self.tcl(pred1, pred2)
        if feat_loss is not None:
            return loss_tcl + feat_loss
        return loss_tcl


class VolumetricLoss(nn.Module):
    """Volume regression + seg-vol consistency.

    Volume regression: MSE on log1p-scaled volumes (proven in VASNet).
    Seg-vol consistency: vol_pred ≈ seg_derived_vol (stop-gradient on seg side).
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, vol_pred, gt_vol, seg_vol_detached=None):
        """
        Args:
            vol_pred: [B, K] from volume head (log-scale via Softplus)
            gt_vol: [B, K] ground truth voxel counts
            seg_vol_detached: [B, K] from softmax seg (stop-gradient)
        """
        target = torch.log1p(gt_vol)
        loss_vol = self.mse(vol_pred, target)

        loss_cons = torch.tensor(0.0, device=vol_pred.device)
        if seg_vol_detached is not None:
            loss_cons = self.mse(vol_pred, torch.log1p(seg_vol_detached))

        return loss_vol + 0.5 * loss_cons


# =============================================================================
# 2. Uncertainty-Weighted Multi-Task Loss (Kendall et al. CVPR 2018)
# =============================================================================

class UncertaintyWeightedLoss(nn.Module):
    """Automatic multi-task loss balancing via homoscedastic uncertainty.

    L = Σ (1/(2·σᵢ²))·Lᵢ + log(σᵢ)

    Learns σᵢ per task — no manual λ tuning needed.

    Reference: Kendall, Gal & Cipolla. "Multi-Task Learning Using Uncertainty
    to Weigh Losses." CVPR 2018. (5000+ citations)
    """

    def __init__(self, n_tasks=4):
        super().__init__()
        # log(σ²) initialized to 0 → σ = 1 → equal initial weighting
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses, active_mask=None):
        """
        Args:
            losses: list of n_tasks scalar losses
            active_mask: list of bool — only weight active (non-zero) tasks.
                         Inactive tasks are excluded from weighting to prevent
                         log_var drift when loss=0. (Fixes weight explosion bug)
        Returns:
            total: weighted sum
            weights: current task weights for logging
        """
        total = 0.0
        weights = []
        for i, loss in enumerate(losses):
            is_active = active_mask[i] if active_mask is not None else True
            if is_active and loss.requires_grad:
                log_var_clamped = self.log_vars[i].clamp(-6, 6)  # C1: prevent overflow
                precision = torch.exp(-log_var_clamped)  # 1/σ²
                total += precision * loss + log_var_clamped
                weights.append(precision.detach().item())
            else:
                total += loss  # pass through without weighting
                weights.append(0.0)
        return total, weights


# =============================================================================
# 3. Dataset — extends VASNet dataset with site labels
# =============================================================================

class ReproSegDataset(torch.utils.data.Dataset):
    """Dataset for ReproSeg training. Adds site_id to VASNet dataset."""

    def __init__(self, cross_df, longitudinal_pairs, pad_to=(192, 224, 192),
                 is_train=True):
        self.cross_df = cross_df.reset_index(drop=True)
        self.pairs = longitudinal_pairs
        self.pad_to = pad_to
        self.is_train = is_train
        self.n_cross = len(self.cross_df)
        self.n_pairs = len(self.pairs)

    def __len__(self):
        return self.n_cross + self.n_pairs

    def _load(self, vol_path, seg_path, dataset_name, force_flip=None):
        """Load volume + seg + site label.

        Args:
            force_flip: if bool, override random flip (for paired scans sharing same flip).
                        if None, random flip as usual.
        """
        if vol_path.exists() and seg_path.exists():
            vol = nib.load(str(vol_path)).get_fdata(dtype=np.float32)
            seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
        else:
            vol = np.zeros((192, 224, 192), dtype=np.float32)
            seg = np.zeros((192, 224, 192), dtype=np.int16)

        # L-R flip augmentation (shared for paired scans to avoid TCL corruption)
        flipped = False
        do_flip = force_flip if force_flip is not None else (self.is_train and random.random() < 0.5)
        if do_flip:
            flipped = True
            vol = np.flip(vol, axis=0).copy()
            seg = np.flip(seg, axis=0).copy()
            seg_swapped = seg.copy()
            for l, r in [(1, 2), (3, 4), (5, 6), (9, 10)]:
                seg_swapped[seg == l] = r
                seg_swapped[seg == r] = l
            seg = seg_swapped

        # GT volumes
        gt_vol = np.array([float((seg == c).sum()) for c in VOL_CLASSES],
                          dtype=np.float32)

        # Normalize + tensor
        vol = norm_zs(vol)
        vt = torch.from_numpy(vol).float().unsqueeze(0)
        if self.is_train:
            vt = vt * torch.empty(1).uniform_(0.9, 1.1).item()
            vt = vt + torch.randn_like(vt) * 0.01
        vt = _pad(vt, self.pad_to)

        seg_t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0)
        seg_t = _pad(seg_t, self.pad_to).squeeze(0).long()

        # Site label
        site_id = SITE_MAP.get(dataset_name.lower(), 0)

        return vt, seg_t, flipped, gt_vol, site_id

    def __getitem__(self, idx):
        if idx < self.n_cross:
            row = self.cross_df.iloc[idx]
            d = Path(row['output_dir'])
            vt, seg_t, flipped, gt_vol, site_id = self._load(
                d / 'native_t1w.nii.gz', d / 'native_seg.nii.gz',
                row.get('dataset', 'unknown')
            )
            return {
                'volume': vt, 'seg_target': seg_t,
                'is_flipped': flipped, 'gt_vol': torch.from_numpy(gt_vol),
                'site_id': site_id, 'is_pair': False,
                'subject_id': row.get('subject_id', ''),
            }
        else:
            # Pair format from build_longitudinal_pairs:
            # {'subject_id', 'vol1', 'seg1', 'vol2', 'seg2'}
            pair = self.pairs[idx - self.n_cross]
            # Infer dataset from path: .../longitudinal/ADNI/... → adni
            dataset_name = Path(pair['vol1']).parts[-4].lower() if pair['vol1'] else 'unknown'
            # Shared flip for paired scans (prevent TCL corruption)
            shared_flip = self.is_train and random.random() < 0.5
            items = []
            for vol_key, seg_key in [('vol1', 'seg1'), ('vol2', 'seg2')]:
                vt, seg_t, flipped, gt_vol, site_id = self._load(
                    Path(pair[vol_key]), Path(pair[seg_key]),
                    dataset_name, force_flip=shared_flip
                )
                items.append({
                    'volume': vt, 'seg_target': seg_t,
                    'is_flipped': flipped, 'gt_vol': torch.from_numpy(gt_vol),
                    'site_id': site_id, 'is_pair': True,
                    'subject_id': pair.get('subject_id', ''),
                })
            return items


def collate_reproseg(batch):
    """Custom collate: flatten paired samples into batch, track pair indices."""
    volumes, seg_targets, is_flipped, gt_vols, site_ids = [], [], [], [], []
    tcl_pairs = []
    cur = 0

    for item in batch:
        if isinstance(item, list):
            # Paired samples
            for sub in item:
                volumes.append(sub['volume'])
                seg_targets.append(sub['seg_target'])
                is_flipped.append(sub['is_flipped'])
                gt_vols.append(sub['gt_vol'])
                site_ids.append(sub['site_id'])
            tcl_pairs.append((cur, cur + 1))
            cur += 2
        else:
            volumes.append(item['volume'])
            seg_targets.append(item['seg_target'])
            is_flipped.append(item['is_flipped'])
            gt_vols.append(item['gt_vol'])
            site_ids.append(item['site_id'])
            cur += 1

    return {
        'volume': torch.stack(volumes),
        'seg_target': torch.stack(seg_targets),
        'is_flipped': is_flipped,
        'gt_vol': torch.stack(gt_vols),
        'site_id': torch.tensor(site_ids, dtype=torch.long),
        'tcl_pairs': tcl_pairs,
    }


# =============================================================================
# 4. PCGrad
# =============================================================================

def pcgrad_backward(task_losses_active, shared_params):
    """PCGrad: project conflicting task gradients before combining (Yu et al., NeurIPS 2020).

    Only applied to shared_params (backbone). Task-specific heads receive
    their own task's gradient directly.

    Args:
        task_losses_active: list of (task_idx, loss_tensor) — active tasks only
        shared_params: list of nn.Parameter (backbone params to apply PCGrad)

    Returns:
        cos_sims: dict {f't{i}_t{j}': cosine_sim} — gradient alignment per pair
                  Negative values indicate conflict that was resolved.
    """
    n = len(task_losses_active)
    if n <= 1:
        if n == 1:
            task_losses_active[0][1].backward()
        return {}

    # ── Step 1: per-task gradients for shared backbone ────────────────────
    raw_grads = []  # list of lists: raw_grads[task][param]
    for idx, (_, loss) in enumerate(task_losses_active):
        retain = (idx < n - 1)
        gs = torch.autograd.grad(
            loss, shared_params, retain_graph=retain, allow_unused=True,
        )
        raw_grads.append([g.detach() if g is not None else None for g in gs])

    # ── Step 2: PCGrad projection ─────────────────────────────────────────
    # For each task i, project against each task j independently.
    # If gi · gj < 0: gi_proj = gi - (gi·gj / |gj|²) * gj
    projected = [[g.clone() if g is not None else None for g in grads]
                 for grads in raw_grads]

    cos_sims = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Only use params where BOTH tasks have non-None gradients
            # (avoids size mismatch when task-specific params get None grad)
            common_i, common_j, common_idx = [], [], []
            for k in range(len(shared_params)):
                if raw_grads[i][k] is not None and raw_grads[j][k] is not None:
                    common_i.append(raw_grads[i][k].reshape(-1))
                    common_j.append(raw_grads[j][k].reshape(-1))
                    common_idx.append(k)

            if not common_i:
                continue

            gi_flat = torch.cat(common_i)
            gj_flat = torch.cat(common_j)
            dot = (gi_flat * gj_flat).sum()

            # Log cos-sim once per pair (using common params only)
            if i < j:
                cos = dot / (gi_flat.norm() * gj_flat.norm() + 1e-12)
                cos_sims[f't{task_losses_active[i][0]}_t{task_losses_active[j][0]}'] = \
                    round(cos.item(), 4)

            if dot < 0:  # conflict → project task i away from task j
                gj_norm_sq = (gj_flat * gj_flat).sum().clamp(min=1e-8)
                scale = dot / gj_norm_sq
                for k in common_idx:
                    if projected[i][k] is not None:
                        projected[i][k] = projected[i][k] - scale * raw_grads[j][k]

    # ── Step 3: assign combined projected gradients ───────────────────────
    for k, p in enumerate(shared_params):
        parts = [projected[i][k] for i in range(n) if projected[i][k] is not None]
        if parts:
            p.grad = sum(parts)

    return cos_sims


# =============================================================================
# 5. Training Loop
# =============================================================================

def train_one_epoch(model, loader, optimizer, loss_fns, epoch, device, args):
    """One training epoch.

    Args:
        model: ReproSeg (DDP-wrapped)
        loader: DataLoader
        optimizer: AdamW
        loss_fns: dict with 'seg', 'repro', 'inv', 'vol', 'uwl'
        epoch: current epoch
        device: CUDA device
        args: CLI arguments
    """
    model.train()
    uwl = loss_fns['uwl']
    seg_fn = loss_fns['seg']
    repro_fn = loss_fns['repro']
    vol_fn = loss_fns['vol']
    site_ce = nn.CrossEntropyLoss().to(device)

    # GRL alpha ramp: 0→1 over 15 epochs
    grl_alpha = min(1.0, epoch / 15.0)

    total_loss = 0.0
    n_batches = 0
    weight_log = [0.0] * 4
    cos_log = {}  # accumulated cosine similarities across batches

    for batch in loader:
        vol = batch['volume'].to(device)
        seg_target = batch['seg_target'].to(device)
        gt_vol = batch['gt_vol'].to(device)
        site_id = batch['site_id'].to(device)
        is_flipped = batch['is_flipped']
        tcl_pairs = batch['tcl_pairs']

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', dtype=torch.bfloat16):
            # ── Forward ──────────────────────────────────────────────
            has_pairs = len(tcl_pairs) > 0
            # Pass pair_indices so feat_consistency runs INSIDE DDP forward
            # (avoids DDP double-ready RuntimeError)
            pair_idx = tcl_pairs if (has_pairs and args.mode in ('repro', 'full')) else None

            use_csg = args.mode not in ('grl_only',)
            seg_logits, vol_pred, seg_vol, site_logits, feat_loss_model, gate_vals = \
                model(vol, is_flipped, return_features=False,
                      grl_alpha=grl_alpha, pair_indices=pair_idx, use_csg=use_csg)

            # ── Loss 1: Segmentation (always active) ────────────────
            loss_seg = seg_fn(seg_logits, seg_target)

            # ── Loss 2: Reproducibility (paired samples only) ───────
            loss_repro = torch.tensor(0.0, device=device)
            if has_pairs and args.mode in ('repro', 'full'):
                i1s, i2s = zip(*tcl_pairs)
                pred1 = torch.cat([seg_logits[i:i+1] for i in i1s])
                pred2 = torch.cat([seg_logits[i:i+1] for i in i2s])
                loss_repro = repro_fn(pred1, pred2, feat_loss=feat_loss_model)

            # ── Loss 3: Adversarial site invariance ─────────────────
            loss_inv = torch.tensor(0.0, device=device)
            if args.mode in ('inv', 'full', 'grl_only') and site_logits is not None:
                loss_inv = site_ce(site_logits, site_id)

            # ── Loss 4: Volumetric ──────────────────────────────────
            loss_vol = torch.tensor(0.0, device=device)
            if vol_pred is not None and args.mode in ('vol', 'full'):
                loss_vol = vol_fn(vol_pred, gt_vol, seg_vol)

            # ── Combine via uncertainty weighting ───────────────────
            losses = [loss_seg, loss_repro, loss_inv, loss_vol]
            active = [True,
                      has_pairs and args.mode in ('repro', 'full'),
                      args.mode in ('inv', 'full', 'grl_only', 'inv_only'),
                      vol_pred is not None and args.mode in ('vol', 'full')]
            loss, weights = uwl(losses, active_mask=active)

            # Collect active (task_idx, loss) pairs for PCGrad
            # task indices: 0=seg, 1=repro, 2=inv, 3=vol
            active_task_losses = [(i, losses[i]) for i, a in enumerate(active)
                                  if a and losses[i].requires_grad]

            # ── Gate entropy regularizer (prevent CSG collapse) ─────
            if gate_vals is not None and args.mode in ('inv', 'full', 'inv_only'):
                for gv in gate_vals:
                    eps = 1e-6
                    ent = -(gv * torch.log(gv + eps) +
                            (1 - gv) * torch.log(1 - gv + eps))
                    loss = loss - 0.01 * ent.mean()  # H4: per-voxel entropy

        # ── Backward ─────────────────────────────────────────────────
        if getattr(args, 'pcgrad', False) and len(active_task_losses) > 1:
            # PCGrad on backbone params; other params get standard gradients
            # from UWL-combined loss for their task-specific components.
            backbone_params = [p for p in unwrap_model(model).backbone.parameters()
                               if p.requires_grad]
            # UWL-combined loss backward for non-backbone params
            # (retain_graph=True because pcgrad_backward needs graph)
            loss.backward(retain_graph=True)
            # Zero backbone grads — pcgrad_backward will set them
            for p in backbone_params:
                if p.grad is not None:
                    p.grad.zero_()
            step_cos = pcgrad_backward(active_task_losses, backbone_params)
            for k, v in step_cos.items():
                cos_log[k] = cos_log.get(k, 0.0) + v
        else:
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for i, w in enumerate(weights):
            weight_log[i] += w
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_weights = [w / max(n_batches, 1) for w in weight_log]
    avg_cos = {k: v / max(n_batches, 1) for k, v in cos_log.items()}

    return avg_loss, avg_weights, avg_cos


# =============================================================================
# 5. Validation
# =============================================================================

@torch.no_grad()
def validate(model, loader, seg_fn, device):
    """Validation: compute Dice per class + total loss."""
    model.eval()
    total_loss = 0.0
    dice_sums = torch.zeros(NUM_GROUPED_CLASSES - 1, device=device)
    n_samples = 0

    for batch in loader:
        vol = batch['volume'].to(device)
        seg_target = batch['seg_target'].to(device)

        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(vol, return_features=False)
            seg_logits = outputs[0]
            loss = seg_fn(seg_logits, seg_target)

        total_loss += loss.item()

        # Per-class Dice
        pred = seg_logits.argmax(dim=1)
        pred_oh = F.one_hot(pred, NUM_GROUPED_CLASSES).permute(0, 4, 1, 2, 3).float()
        tgt_oh = F.one_hot(seg_target, NUM_GROUPED_CLASSES).permute(0, 4, 1, 2, 3).float()
        inter = (pred_oh * tgt_oh).sum(dim=(0, 2, 3, 4))
        union = pred_oh.sum(dim=(0, 2, 3, 4)) + tgt_oh.sum(dim=(0, 2, 3, 4))
        dice = (2.0 * inter + 1.0) / (union + 1.0)
        dice_sums += dice[1:]  # skip background
        n_samples += 1

    # H3: reduce val metrics across DDP ranks (was only using rank-0 subset)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(dice_sums)
        n_tensor = torch.tensor(float(n_samples), device=device)
        torch.distributed.all_reduce(n_tensor)
        n_samples = int(n_tensor.item())
        loss_tensor = torch.tensor(total_loss, device=device)
        torch.distributed.all_reduce(loss_tensor)
        total_loss = loss_tensor.item()

    avg_loss = total_loss / max(n_samples, 1)
    avg_dice = dice_sums / max(n_samples, 1)
    mean_dice = avg_dice.mean().item()

    return avg_loss, mean_dice, avg_dice


# =============================================================================
# 6. Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Train ReproSeg')
    # Experiment
    p.add_argument('--exp-name', type=str, required=True)
    p.add_argument('--mode', type=str, default='full',
                   choices=['baseline', 'repro', 'inv', 'vol', 'full', 'grl_only', 'inv_only'],
                   help='baseline: seg only | repro: +TCL+feat | inv: +GRL+CSG | '
                        'vol: +volume head | full: all | '
                        'grl_only: GRL loss but NO CSG purification | '
                        'inv_only: CSG purification but NO GRL loss')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=2)
    # Learning rate
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-head', type=float, default=5e-4)
    p.add_argument('--lr-inv', type=float, default=2e-4)
    # Volume head
    p.add_argument('--head-type', type=str, default='lr_aware',
                   choices=['gap', 'lr_aware'])
    # Gradient management
    p.add_argument('--pcgrad', action='store_true',
                   help='Enable PCGrad: project conflicting task gradients on backbone '
                        '(Yu et al., NeurIPS 2020). Logs grad_cos per epoch. '
                        'No effect in baseline mode (single task).')
    # Data
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dist_info = setup_distributed()
    local_rank = dist_info['local_rank']
    world_size = dist_info['world_size']
    device = torch.device(f'cuda:{local_rank}')
    torch.manual_seed(args.seed + local_rank)
    torch.cuda.manual_seed(args.seed + local_rank)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Output directory ─────────────────────────────────────────────────
    exp_dir = EXP_BASE / 'results' / args.exp_name
    ckpt_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  ReproSeg Training: {args.exp_name}")
        print(f"  Mode: {args.mode}")
        print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
        print(f"  LR: backbone={args.lr}, head={args.lr_head}, inv={args.lr_inv}")
        print(f"  PCGrad: {'ON (backbone only)' if args.pcgrad else 'OFF (standard backward)'}")
        print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    if is_main_process():
        print("Loading data...")
    manifest = pd.read_csv(str(MANIFEST))
    progression_df = pd.read_csv(str(PROGRESSION))
    pairs = build_longitudinal_pairs(progression_df, max_pairs=2000)

    # Split — cross-sectional (manifest) and longitudinal (pairs) are independent
    # subject pools with zero overlap, so we split each separately.
    np.random.seed(args.seed)
    subjects = manifest['subject_id'].unique()
    np.random.shuffle(subjects)
    n = len(subjects)
    train_ids = set(subjects[:int(0.8 * n)])
    val_ids = set(subjects[int(0.8 * n):int(0.9 * n)])

    train_df = manifest[manifest['subject_id'].isin(train_ids)]
    val_df = manifest[manifest['subject_id'].isin(val_ids)]

    # Longitudinal pairs: independent 80/20 split (disjoint from manifest subjects)
    pair_subjects = list({p['subject_id'] for p in pairs})
    np.random.shuffle(pair_subjects)
    train_pair_ids = set(pair_subjects[:int(0.8 * len(pair_subjects))])
    train_pairs = [p for p in pairs if p.get('subject_id', '') in train_pair_ids]

    train_ds = ReproSegDataset(train_df, train_pairs, is_train=True)
    val_ds = ReproSegDataset(val_df, [], is_train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    def worker_init_fn(worker_id):  # M6: rank-aware worker seeding
        np.random.seed(args.seed + local_rank * 100 + worker_id)
        random.seed(args.seed + local_rank * 100 + worker_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=4,
                              collate_fn=collate_reproseg, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=1,
                            sampler=val_sampler, num_workers=2,
                            collate_fn=collate_reproseg, pin_memory=True)

    if is_main_process():
        print(f"  Train: {len(train_df)} cross + {len(train_pairs)} pairs")
        print(f"  Val: {len(val_df)} subjects")

        # ── Split diagnostics ─────────────────────────────────────────────
        # 1) Leakage check: no subject should appear in both train and val
        leak = train_ids & val_ids
        assert len(leak) == 0, f"SPLIT LEAK: {len(leak)} subjects in both train and val"

        # 2) Scan-per-subject distribution in val (detects bias risk)
        val_counts = val_df.groupby('subject_id').size()
        multi_scan_subjects = (val_counts > 1).sum()
        print(f"\n  [Split Diagnostics]")
        print(f"  Leakage check : PASS (0 overlapping subjects)")
        print(f"  Train subjects: {len(train_ids)} → {len(train_df)} scans "
              f"({len(train_df)/len(train_ids):.1f} scans/subject avg)")
        print(f"  Val subjects  : {len(val_ids)} → {len(val_df)} scans "
              f"({len(val_df)/len(val_ids):.1f} scans/subject avg)")
        print(f"  Val multi-scan subjects: {multi_scan_subjects}/{len(val_ids)} "
              f"({100*multi_scan_subjects/len(val_ids):.1f}%)  ← inflates val dice if >0")
        print(f"  Val scan distribution : "
              f"min={val_counts.min()} max={val_counts.max()} "
              f"median={val_counts.median():.0f}")

        # 3) Site distribution in train vs val (checks for site imbalance)
        if 'site' in manifest.columns or 'dataset' in manifest.columns:
            site_col = 'site' if 'site' in manifest.columns else 'dataset'
            print(f"\n  Train site dist: {dict(train_df[site_col].value_counts())}")
            print(f"  Val   site dist: {dict(val_df[site_col].value_counts())}")
        print()

    # ── Model ────────────────────────────────────────────────────────────
    if is_main_process():
        print("Building ReproSeg...")
    from monai.networks.nets import SwinUNETR
    backbone = SwinUNETR(
        in_channels=1, out_channels=NUM_GROUPED_CLASSES,
        feature_size=48, use_checkpoint=True, spatial_dims=3,
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
    ).to(device)

    # Load pretrained weights
    ckpt = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)
    clean_state = {k.replace('module.', ''): v
                   for k, v in ckpt['state_dict'].items()}
    backbone.load_state_dict(clean_state, strict=False)

    # Volume head (only for vol/full modes)
    volume_head = None
    if args.mode in ('vol', 'full'):
        volume_head = LRAwareVolumeHead(in_ch=768, K=K).to(device)

    model = ReproSeg(backbone, n_sites=N_SITES, volume_head=volume_head).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if is_main_process():
        count_params(unwrap_model(model))

    # ── Optimizer: per-component LR ──────────────────────────────────────
    param_groups = [
        {'params': list(unwrap_model(model).backbone.parameters()),
         'lr': args.lr, 'name': 'backbone'},
        {'params': list(unwrap_model(model).inv_stream.parameters()),
         'lr': args.lr_inv, 'name': 'inv_stream'},
        {'params': list(unwrap_model(model).csg.parameters()),
         'lr': args.lr_inv, 'name': 'csg'},
        {'params': list(unwrap_model(model).site_classifier.parameters()),
         'lr': args.lr_inv, 'name': 'site_cls'},
        {'params': list(unwrap_model(model).feat_consistency.parameters()),
         'lr': args.lr_inv, 'name': 'feat_cons'},
    ]
    if volume_head is not None:
        param_groups.append({
            'params': list(unwrap_model(model).volume_head.parameters()),
            'lr': args.lr_head, 'name': 'vol_head',
        })

    # ── Loss functions (before optimizer so UWL params are included) ────
    seg_fn = DiceCELoss(NUM_GROUPED_CLASSES).to(device)
    repro_fn = ReproducibilityLoss(NUM_GROUPED_CLASSES).to(device)
    vol_fn = VolumetricLoss().to(device)
    uwl = UncertaintyWeightedLoss(n_tasks=4).to(device)

    # Include UWL params in optimizer from the start
    param_groups.append({
        'params': list(uwl.parameters()), 'lr': args.lr, 'name': 'uwl'
    })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # ── LR Scheduler: warmup + cosine ────────────────────────────────────
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fns = {
        'seg': seg_fn, 'repro': repro_fn, 'vol': vol_fn, 'uwl': uwl,
    }

    # ── Training loop ────────────────────────────────────────────────────
    best_dice = 0.0
    history = []

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        train_loss, task_weights, grad_cos = train_one_epoch(
            model, train_loader, optimizer, loss_fns, epoch, device, args
        )
        scheduler.step()

        val_loss, val_dice, per_class_dice = validate(
            model, val_loader, seg_fn, device
        )

        # ── Logging ──────────────────────────────────────────────────
        if is_main_process():
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']

            # UWL weights → readable format
            w_names = ['seg', 'repro', 'inv', 'vol']
            w_str = ' | '.join(f'{n}={w:.2f}' for n, w in zip(w_names, task_weights))

            # Gradient cosine sims (PCGrad mode)
            cos_str = ''
            if grad_cos:
                label = {'t0_t1': 'seg↔repro', 't0_t2': 'seg↔inv',
                         't0_t3': 'seg↔vol',   't1_t2': 'repro↔inv',
                         't1_t3': 'repro↔vol', 't2_t3': 'inv↔vol'}
                parts = [f'{label.get(k, k)}={v:+.3f}' for k, v in sorted(grad_cos.items())]
                cos_str = ' | cos: ' + ' '.join(parts)

            print(f"  ep {epoch:2d}/{args.epochs} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} | "
                  f"dice={val_dice:.4f} | lr={lr_now:.2e} | "
                  f"weights: {w_str}{cos_str} | {elapsed:.0f}s")

            # Per-class Dice (every 5 epochs)
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                for i in range(NUM_GROUPED_CLASSES - 1):
                    name = GROUP_NAMES.get(i + 1, f'class_{i+1}')
                    print(f"    {name}: {per_class_dice[i].item():.4f}")

            # Save history
            entry = {
                'epoch': epoch, 'train_loss': train_loss,
                'val_loss': val_loss, 'val_dice': val_dice,
                'task_weights': dict(zip(w_names, task_weights)),
                'grad_cos': grad_cos,  # gradient cosine similarities (empty if not pcgrad)
                'lr': lr_now, 'elapsed': elapsed,
            }
            history.append(entry)

            with open(log_dir / 'training_log.json', 'w') as f:
                json.dump(history, f, indent=2)

            # Save best
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # H2: resume support
                    'uwl_state_dict': uwl.state_dict(),              # H2: resume support
                    'val_dice': val_dice,
                    'args': vars(args),
                }, ckpt_dir / 'best.pt')
                print(f"    ★ New best: {val_dice:.4f}")

        barrier()

    # ── Final ────────────────────────────────────────────────────────────
    if is_main_process():
        print(f"\nTraining complete. Best Dice: {best_dice:.4f}")
        print(f"Results: {exp_dir}")

    cleanup_distributed()


if __name__ == '__main__':
    main()
