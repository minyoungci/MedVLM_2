"""ReproSeg V1: Reproducibility-Equivariant Dual-Stream Brain Segmentation.

Version: V1 (2026-04-01)
Total params: 75.7M (new components: 13.5M = 18%)

Architecture:
  INPUT [B, 1, 192, 224, 192]
      │
  [SwinViT Patch Embedding + Stage 0]
      │
      ├──→ Anatomy Stream (A)                Invariance Stream (I)
      │     SwinUNETR encoder (62.2M)         Lightweight CNN (5.6M, half-ch)
      │     (BrainSegFounder pretrained)      (random init)
      │     captures: anatomy, boundaries     captures: scanner bias, noise
      │                                       trained with GRL to predict site
      │     hs[0]: [B,  48, 96,112, 96]      inv[0]: [B,  24, 96,112, 96]
      │     hs[1]: [B,  96, 48, 56, 48]      inv[1]: [B,  48, 48, 56, 48]
      │     hs[2]: [B, 192, 24, 28, 24]      inv[2]: [B,  96, 24, 28, 24]
      │     hs[3]: [B, 384, 12, 14, 12]      inv[3]: [B, 192, 12, 14, 12]
      │     bottleneck: [B, 768, 6, 7, 6]
      │          │                                   │
      │     [Cross-Stream Gating at each scale]
      │     A_pure[i] = A[i] - σ(gate[i]) ⊙ I[i]
      │          │
      │     [Seg Decoder] → [B, 12, 192, 224, 192]
      │          │
      │     [Volume Head] → [B, 7]  (LR-aware)
      │
  Loss = L_dice_ce
       + λ_tcl  × L_tcl      (paired sessions, volume consistency)
       + λ_vol  × L_vol      (volume regression)
       + λ_cons × L_cons     (seg-volume consistency)
       + λ_inv  × L_inv      (adversarial site classification via GRL)
       + λ_feat × L_feat     (multi-scale feature consistency for paired scans)

Novel components (vs VASNet):
  1. Invariance Stream — lightweight encoder that models scanner/site effects
  2. Cross-Stream Gating (CSG) — subtracts scanner effects from anatomy features
  3. Multi-Scale Feature Consistency — enforces paired-scan consistency at ALL
     encoder scales, not just final volumes
  4. Adversarial Site Prediction — GRL on invariance stream features

Tensor shapes (192×224×192 input):
  Stage 0:  [B,  48,  96, 112,  96]  (stride 2 from input)
  Stage 1:  [B,  96,  48,  56,  48]  (stride 4)
  Stage 2:  [B, 192,  24,  28,  24]  (stride 8)
  Stage 3:  [B, 384,  12,  14,  12]  (stride 16)
  Bottleneck:[B, 768,   6,   7,   6]  (stride 32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# =============================================================================
# 1. Gradient Reversal Layer (for adversarial site prediction)
# =============================================================================

class GradientReversal(Function):
    """Reverses gradient during backward pass (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# =============================================================================
# 2. Invariance Stream — lightweight CNN encoder for scanner/site effects
# =============================================================================

class InvResBlock(nn.Module):
    """Residual block for invariance stream."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(ch, affine=True)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(ch, affine=True)

    def forward(self, x):
        residual = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.gelu(out + residual)


class InvarianceStream(nn.Module):
    """Lightweight CNN encoder that captures scanner/site-specific features.

    Produces features at 4 scales matching the anatomy stream (hs[0..3]).
    Does NOT need to match bottleneck — CSG only operates at hs scales.

    Default channels are HALF of anatomy stream to keep it lightweight (~5.6M).
    CSG gate handles the channel mismatch via 1x1 conv projection.
    """

    def __init__(self, in_ch=48, stage_channels=(24, 48, 96, 192)):
        super().__init__()
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i, out_ch in enumerate(stage_channels):
            ch_in = in_ch if i == 0 else stage_channels[i - 1]
            # Each stage: project channels + 2 res blocks
            stage = nn.Sequential(
                nn.Conv3d(ch_in, out_ch, 1, bias=False) if ch_in != out_ch else nn.Identity(),
                InvResBlock(out_ch),
                InvResBlock(out_ch),
            )
            self.stages.append(stage)
            # Downsample between stages (except last)
            if i < len(stage_channels) - 1:
                self.downs.append(
                    nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)
                )
            else:
                self.downs.append(nn.Identity())

    def forward(self, x):
        """Input: shared stem output [B, 48, 96, 112, 96].
        Returns: list of 4 feature maps matching anatomy stream scales.
        """
        features = []
        h = x
        for i, (stage, down) in enumerate(zip(self.stages, self.downs)):
            h = stage(h)
            features.append(h)
            if i < len(self.stages) - 1:
                h = down(h)
        return features  # [stage0, stage1, stage2, stage3]


# =============================================================================
# 3. Cross-Stream Gating (CSG) — subtract scanner effects from anatomy
# =============================================================================

class CrossStreamGate(nn.Module):
    """Learns a soft gate σ(W·[A;I]) that modulates how much of the invariance
    stream is subtracted from the anatomy stream.

    A_pure = A - σ(gate) ⊙ I

    This is ARCHITECTURAL — the gradient flows differently through each stream,
    not just a loss penalty.
    """

    def __init__(self, a_ch, i_ch=None):
        """
        Args:
            a_ch: anatomy stream channels
            i_ch: invariance stream channels (if None, same as a_ch)
        """
        super().__init__()
        if i_ch is None:
            i_ch = a_ch
        # Project invariance features to match anatomy channels if needed
        self.i_proj = nn.Conv3d(i_ch, a_ch, 1, bias=False) if i_ch != a_ch else nn.Identity()
        # Gate network: takes concatenated [A, I_proj] → sigmoid gate
        self.gate = nn.Sequential(
            nn.Conv3d(a_ch * 2, a_ch, 1, bias=False),
            nn.InstanceNorm3d(a_ch, affine=True),
            nn.GELU(),
            nn.Conv3d(a_ch, a_ch, 1),
            nn.Sigmoid(),
        )
        self.register_buffer('_target_entropy', torch.tensor(0.5))

    def forward(self, a_feat, i_feat):
        """
        Args:
            a_feat: anatomy stream feature [B, Ca, D, H, W]
            i_feat: invariance stream feature [B, Ci, D, H, W]
        Returns:
            a_pure: anatomy-pure feature [B, Ca, D, H, W]
            gate_val: gate values for monitoring/regularization
        """
        i_mapped = self.i_proj(i_feat)                                    # [B, Ca, D, H, W]
        i_unit = F.normalize(i_mapped, dim=1, eps=1e-6)                    # unit vector in channel space
        gate_val = self.gate(torch.cat([a_feat, i_mapped], dim=1))
        proj_coeff = (a_feat * i_unit).sum(dim=1, keepdim=True)            # scalar per voxel
        a_pure = a_feat - gate_val * proj_coeff * i_unit                   # remove only parallel component
        return a_pure, gate_val

    def gate_entropy_loss(self, gate_val):
        """Regularizer: prevent gate from collapsing to all-0 or all-1.
        Per-voxel binary entropy mean — prevents regional collapse even when
        global mean looks healthy. (H4 fix: was using entropy of mean, not mean of entropy)
        """
        eps = 1e-6
        ent = -(gate_val * torch.log(gate_val + eps) +
                (1 - gate_val) * torch.log(1 - gate_val + eps))
        return -ent.mean()


# =============================================================================
# 4. Site Classifier (on invariance stream, with GRL)
# =============================================================================

class SiteClassifier(nn.Module):
    """Predicts acquisition site from invariance stream bottleneck features.
    Applied AFTER gradient reversal — so the invariance stream is trained
    to capture site info, while the anatomy stream is trained to NOT capture it.
    """

    def __init__(self, in_ch=384, n_sites=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_sites),
        )

    def forward(self, inv_feat_stage3, alpha=1.0):
        """
        Args:
            inv_feat_stage3: invariance stream stage 3 feature [B, 384, 12, 14, 12]
            alpha: GRL strength (ramp up during training)
        Returns:
            site_logits: [B, n_sites]
        """
        # GRL: reverse gradient so anatomy stream learns to be site-invariant
        feat = grad_reverse(inv_feat_stage3, alpha)
        feat = self.pool(feat).flatten(1)  # [B, 384]
        return self.classifier(feat)


# =============================================================================
# 5. Multi-Scale Feature Consistency Loss
# =============================================================================

class MultiScaleFeatureConsistency(nn.Module):
    """Enforces feature-level consistency between paired scans at multiple
    encoder scales. Unlike TCL which only compares final volumes, this
    operates on intermediate representations.

    For stable subjects scanned twice: features should be similar at ALL scales.
    """

    def __init__(self, stage_channels=(48, 96, 192, 384), stage_weights=(0.1, 0.2, 0.3, 0.4)):
        super().__init__()
        self.stage_weights = stage_weights
        # Projection heads to reduce dimensionality before comparison
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(4),  # reduce spatial to 4×4×4
                nn.Flatten(),
                nn.Linear(ch * 64, 128),  # ch × 4³ → 128
                nn.LayerNorm(128),
            )
            for ch in stage_channels
        ])

    def forward(self, feats_1, feats_2):
        """
        Args:
            feats_1: list of 4 anatomy-pure features from scan 1
            feats_2: list of 4 anatomy-pure features from scan 2
        Returns:
            loss: weighted sum of cosine distance at each scale
        """
        total_loss = 0.0
        for i, (f1, f2, proj, w) in enumerate(
            zip(feats_1, feats_2, self.projectors, self.stage_weights)
        ):
            z1 = proj(f1)  # [B_pair, 128]
            z2 = proj(f2)  # [B_pair, 128]
            # SimSiam-style asymmetric stop-gradient to prevent representational collapse
            # (Chen & He, 2021 — each branch is pulled toward the other's detached target)
            loss_12 = (1.0 - F.cosine_similarity(z1, z2.detach(), dim=1)).mean()
            loss_21 = (1.0 - F.cosine_similarity(z2, z1.detach(), dim=1)).mean()
            total_loss += w * 0.5 * (loss_12 + loss_21)
        return total_loss


# =============================================================================
# 6. ReproSeg — Full Architecture
# =============================================================================

class ReproSeg(nn.Module):
    """Reproducibility-Equivariant Dual-Stream Brain Segmentation.

    Components:
      - Shared Stem (Conv3d 7×7×7)
      - Anatomy Stream (SwinUNETR encoder, pretrained)
      - Invariance Stream (lightweight CNN, random init)
      - Cross-Stream Gating at 4 scales
      - Seg Decoder (SwinUNETR decoder)
      - Volume Head (LR-aware)
      - Site Classifier (adversarial, on invariance stream)
    """

    def __init__(self, backbone, n_sites=4, volume_head=None):
        """
        Args:
            backbone: MONAI SwinUNETR instance (pretrained)
            n_sites: number of acquisition sites for adversarial training
            volume_head: optional VolumeHead module
        """
        super().__init__()

        # ── Shared Stem ──────────────────────────────────────────────────
        # Extract the first convolution from SwinUNETR's patch embedding
        # SwinUNETR.swinViT.patch_embed handles: Conv3d → LayerNorm
        # We DON'T replace it — instead we fork AFTER patch embedding
        self.backbone = backbone

        # ── Invariance Stream (lightweight: half channels of anatomy) ────
        i_channels = (24, 48, 96, 192)
        self.inv_stream = InvarianceStream(
            in_ch=48,  # input from hs[0]
            stage_channels=i_channels,
        )

        # ── Cross-Stream Gating (4 scales) ───────────────────────────────
        a_channels = (48, 96, 192, 384)
        self.csg = nn.ModuleList([
            CrossStreamGate(a_ch=ac, i_ch=ic) for ac, ic in zip(a_channels, i_channels)
        ])

        # ── Site Classifier (adversarial) ────────────────────────────────
        self.site_classifier = SiteClassifier(in_ch=192, n_sites=n_sites)  # matches inv stage3

        # ── Volume Head ──────────────────────────────────────────────────
        self.volume_head = volume_head

        # ── Multi-Scale Feature Consistency ──────────────────────────────
        self.feat_consistency = MultiScaleFeatureConsistency()

    def forward(self, x, is_flipped=None, return_features=False, grl_alpha=1.0,
                pair_indices=None, use_csg=True):
        """
        Args:
            x: input volume [B, 1, 192, 224, 192]
            is_flipped: list[bool] for LR-aware volume head
            return_features: if True, also return anatomy-pure features
            grl_alpha: GRL strength (ramp 0→1 over training)
            pair_indices: list of (i1, i2) tuples for feat_consistency.
                          Computed INSIDE forward to avoid DDP double-ready bug.

        Returns:
            seg_logits: [B, 12, 192, 224, 192]
            vol_pred:   [B, 7] or None
            seg_vol:    [B, 7] or None (stop-gradient)
            site_logits:[B, n_sites]
            feat_loss:  scalar tensor or None (multi-scale feature consistency)
            a_pure_feats: list of 4 tensors (only if return_features=True)
            gate_vals:  list of 4 tensors
        """
        # ── Anatomy Stream (SwinUNETR encoder) ───────────────────────────
        # swinViT returns features at 5 scales: hs[0..4]
        hs = self.backbone.swinViT(x, self.backbone.normalize)
        # hs[0]: [B,  48, 96,112, 96]  — stage 0
        # hs[1]: [B,  96, 48, 56, 48]  — stage 1
        # hs[2]: [B, 192, 24, 28, 24]  — stage 2
        # hs[3]: [B, 384, 12, 14, 12]  — stage 3
        # hs[4]: [B, 768,  6,  7,  6]  — stage 4 (for bottleneck)

        # ── Invariance Stream ────────────────────────────────────────────
        # Feed stage-0 anatomy features as input to invariance stream
        # (shared representation up to first SwinViT output)
        inv_feats = self.inv_stream(hs[0])
        # inv_feats[0..3] match hs[0..3] shapes

        # ── Cross-Stream Gating at 4 scales ──────────────────────────────
        a_pure = []
        gate_vals = []
        for i in range(4):
            if use_csg:
                purified, gv = self.csg[i](hs[i], inv_feats[i])
            else:
                purified, gv = hs[i], torch.zeros(hs[i].shape[0], 1, *hs[i].shape[2:],
                                                   device=hs[i].device)
            a_pure.append(purified)
            gate_vals.append(gv)

        # ── Bottleneck (unchanged — no CSG at bottleneck) ────────────────
        enc0       = self.backbone.encoder1(x)
        enc1       = self.backbone.encoder2(a_pure[0])  # use purified features
        enc2       = self.backbone.encoder3(a_pure[1])
        enc3       = self.backbone.encoder4(a_pure[2])
        bottleneck = self.backbone.encoder10(hs[4])     # bottleneck from original

        # ── Seg Decoder ──────────────────────────────────────────────────
        dec3       = self.backbone.decoder5(bottleneck, a_pure[3])  # purified skip
        dec2       = self.backbone.decoder4(dec3, enc3)
        dec1       = self.backbone.decoder3(dec2, enc2)
        dec0       = self.backbone.decoder2(dec1, enc1)
        out        = self.backbone.decoder1(dec0, enc0)
        seg_logits = self.backbone.out(out)  # [B, 12, D, H, W]

        # ── Site Classifier (on invariance stream) ───────────────────────
        site_logits = self.site_classifier(inv_feats[3], alpha=grl_alpha)

        # ── Volume Head ──────────────────────────────────────────────────
        vol_pred = None
        seg_vol = None
        if self.volume_head is not None:
            vol_pred = self.volume_head(bottleneck, is_flipped)

            # 7 dementia-relevant structures (class indices in 12-class seg)
            vol_classes = [1, 2, 3, 4, 5, 6, 7]  # hipp L/R, amyg L/R, ento L/R, vent
            with torch.no_grad():
                seg_soft = F.softmax(seg_logits.float(), dim=1)
                seg_vol = torch.stack(
                    [seg_soft[:, c].sum(dim=(1, 2, 3)) for c in vol_classes], dim=1
                ).detach()

        # ── Multi-Scale Feature Consistency (inside DDP forward) ─────────
        # Computed here (not outside model) to avoid DDP double-ready error.
        feat_loss = None
        if pair_indices is not None and len(pair_indices) > 0:
            i1s = [p[0] for p in pair_indices]
            i2s = [p[1] for p in pair_indices]
            feats1 = [f[i1s] for f in a_pure]
            feats2 = [f[i2s] for f in a_pure]
            feat_loss = self.feat_consistency(feats1, feats2)

        # ── Return ───────────────────────────────────────────────────────
        if return_features:
            return seg_logits, vol_pred, seg_vol, site_logits, feat_loss, a_pure, gate_vals
        return seg_logits, vol_pred, seg_vol, site_logits, feat_loss, gate_vals


# =============================================================================
# 7. Parameter count summary
# =============================================================================

def count_params(model):
    """Print parameter counts by component."""
    components = {
        'backbone (anatomy stream)': model.backbone,
        'invariance stream': model.inv_stream,
        'cross-stream gating': model.csg,
        'site classifier': model.site_classifier,
        'feat consistency projectors': model.feat_consistency,
    }
    if model.volume_head is not None:
        components['volume head'] = model.volume_head

    total = 0
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {n:,} params ({trainable:,} trainable)")
        total += n

    print(f"  TOTAL: {total:,} params")
    return total
