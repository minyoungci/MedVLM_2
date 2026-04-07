"""Generate ReproSeg V1 architecture diagram — publication quality."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.set_aspect('equal')
ax.axis('off')

# Colors
C_INPUT = '#E8E8E8'
C_STEM = '#B3D9FF'
C_ANAT = '#4A90D9'
C_ANAT_L = '#D6E8F7'
C_INV = '#F5A623'
C_INV_L = '#FDE8C8'
C_CSG = '#7B68EE'
C_CSG_L = '#E8E4FD'
C_DEC = '#5CB85C'
C_DEC_L = '#D4EDDA'
C_SITE = '#D9534F'
C_SITE_L = '#F5D0CE'
C_VOL = '#9B59B6'
C_VOL_L = '#EBD5F5'
C_FEAT = '#F0AD4E'
C_FEAT_L = '#FFF3CD'
C_LOSS = '#333333'
C_TEXT = '#222222'

def box(x, y, w, h, color, text, fontsize=8, bold=False, edgecolor=None):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=edgecolor or '#555555',
                          linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=C_TEXT, fontweight=weight, zorder=3,
            linespacing=1.4)

def arrow(x1, y1, x2, y2, color='#555555', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def arrow_curved(x1, y1, x2, y2, color='#555555', rad=0.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                               connectionstyle=f'arc3,rad={rad}'))

# ── Title ────────────────────────────────────────────────
ax.text(7, 21.5, 'ReproSeg V1', ha='center', va='center',
        fontsize=18, fontweight='bold', color=C_TEXT)
ax.text(7, 21.0, 'Reproducibility-Equivariant Dual-Stream Brain Segmentation',
        ha='center', va='center', fontsize=10, color='#666666')
ax.text(7, 20.65, '75.7M params  |  new components: 13.5M (18%)',
        ha='center', va='center', fontsize=8, color='#888888')

# ── Input ────────────────────────────────────────────────
box(4.5, 19.8, 5, 0.6, C_INPUT, 'T1w MRI Input  [B, 1, 192×224×192]', fontsize=9)
arrow(7, 19.8, 7, 19.3)

# ── Shared Stem ──────────────────────────────────────────
box(3.5, 18.6, 7, 0.6, C_STEM, 'SwinViT Patch Embedding + Stage 0  →  [B, 48, 96×112×96]', fontsize=8)
arrow(7, 18.6, 7, 18.2)

# ── Fork label ───────────────────────────────────────────
ax.text(7, 18.1, '◆ FORK', ha='center', va='center', fontsize=7, color='#999999')

# ── Anatomy Stream (left) ────────────────────────────────
ax.text(3.5, 17.8, 'Anatomy Stream', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C_ANAT)
ax.text(3.5, 17.45, '62.2M (pretrained)', ha='center', va='center',
        fontsize=7, color=C_ANAT)

anat_stages = [
    ('Stage 0: 48ch\n96×112×96', 16.7),
    ('Stage 1: 96ch\n48×56×48', 15.5),
    ('Stage 2: 192ch\n24×28×24', 14.3),
    ('Stage 3: 384ch\n12×14×12', 13.1),
]
for text, y in anat_stages:
    box(1.5, y, 4, 0.8, C_ANAT_L, text, fontsize=7)

# Bottleneck
box(1.5, 11.7, 4, 0.8, C_ANAT_L, 'Bottleneck: 768ch\n6×7×6', fontsize=7, bold=True)
for i in range(len(anat_stages)):
    if i < len(anat_stages) - 1:
        arrow(3.5, anat_stages[i][1], 3.5, anat_stages[i+1][1] + 0.8)
arrow(3.5, 13.1, 3.5, 12.5)

# ── Invariance Stream (right) ────────────────────────────
ax.text(10.5, 17.8, 'Invariance Stream', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C_INV)
ax.text(10.5, 17.45, '5.6M (random init, half-ch)', ha='center', va='center',
        fontsize=7, color=C_INV)

inv_stages = [
    ('Stage 0: 24ch\n96×112×96', 16.7),
    ('Stage 1: 48ch\n48×56×48', 15.5),
    ('Stage 2: 96ch\n24×28×24', 14.3),
    ('Stage 3: 192ch\n12×14×12', 13.1),
]
for text, y in inv_stages:
    box(8.5, y, 4, 0.8, C_INV_L, text, fontsize=7)

for i in range(len(inv_stages)):
    if i < len(inv_stages) - 1:
        arrow(10.5, inv_stages[i][1], 10.5, inv_stages[i+1][1] + 0.8)

# ── CSG modules (between streams) ────────────────────────
csg_ys = [16.85, 15.65, 14.45, 13.25]
for i, y in enumerate(csg_ys):
    # CSG diamond
    diamond = plt.Polygon([[7, y+0.25], [7.4, y], [7, y-0.25], [6.6, y]],
                          facecolor=C_CSG_L, edgecolor=C_CSG, linewidth=1.5, zorder=3)
    ax.add_patch(diamond)
    ax.text(7, y, 'CSG', ha='center', va='center', fontsize=6,
            fontweight='bold', color=C_CSG, zorder=4)
    # Arrows: inv → CSG, anat → CSG
    arrow(8.5, y, 7.45, y, color=C_INV, lw=1.0)
    arrow(5.5, y, 6.55, y, color=C_ANAT, lw=1.0)

# CSG formula
ax.text(7, 12.65, 'A_pure = A − σ(gate) · proj(I)', ha='center', va='center',
        fontsize=7, color=C_CSG, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_CSG, alpha=0.8))

# ── GRL + Site Classifier ────────────────────────────────
arrow(10.5, 13.1, 10.5, 12.5)
box(8.8, 11.7, 3.4, 0.7, C_SITE_L, 'GRL → Site Classifier\n4-class (NACC/OASIS/ADNI/AJU)',
    fontsize=6.5, edgecolor=C_SITE)

# ── Decoder ──────────────────────────────────────────────
arrow(3.5, 11.7, 3.5, 11.0)
# Purified skip arrows
for y in csg_ys:
    arrow_curved(6.55, y, 2.8, 10.6, color=C_CSG, rad=-0.15)

box(1.2, 9.5, 5.5, 1.0, C_DEC_L, 'Seg Decoder (SwinUNETR)\npurified skip connections from CSG',
    fontsize=8, edgecolor=C_DEC)

# ── Outputs ──────────────────────────────────────────────
arrow(3, 9.5, 3, 8.9)
box(1, 8.0, 4, 0.8, '#E8F5E9', 'Seg Output\n[B, 12, 192×224×192]', fontsize=8,
    bold=True, edgecolor=C_DEC)

arrow(5.5, 9.5, 8, 8.9)
box(6.5, 8.0, 3.5, 0.8, C_VOL_L, 'Volume Head (LR-aware)\n[B, 7]  (1.2M)',
    fontsize=7, edgecolor=C_VOL)

# ── Feature Consistency ──────────────────────────────────
box(1, 6.8, 9, 0.7, C_FEAT_L,
    'Multi-Scale Feature Consistency  |  4-scale cosine sim for paired scans  (5.9M)',
    fontsize=7, edgecolor=C_FEAT)
arrow(3, 8.0, 3, 7.55)

# ── Loss ─────────────────────────────────────────────────
box(0.5, 5.3, 13, 1.2, '#F8F8F8',
    'Loss = (1/2σ₁²)·L_seg + (1/2σ₂²)·L_repro + (1/2σ₃²)·L_inv + (1/2σ₄²)·L_vol + log(σ₁σ₂σ₃σ₄)\n'
    'Uncertainty-weighted (Kendall et al., CVPR 2018)  —  no manual λ tuning',
    fontsize=7.5, edgecolor='#CCCCCC')

# Loss component labels
loss_labels = [
    (2.5, 4.8, 'L_seg: Dice+CE', C_DEC),
    (5.5, 4.8, 'L_repro: TCL+FeatCons', C_FEAT),
    (8.5, 4.8, 'L_inv: Site CE+GRL', C_SITE),
    (11.5, 4.8, 'L_vol: MSE+Cons', C_VOL),
]
for x, y, text, color in loss_labels:
    ax.text(x, y, text, ha='center', va='center', fontsize=6.5,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=color, alpha=0.7))

# ── Legend ────────────────────────────────────────────────
legend_items = [
    (C_ANAT_L, C_ANAT, 'Anatomy (pretrained)'),
    (C_INV_L, C_INV, 'Invariance (new)'),
    (C_CSG_L, C_CSG, 'Cross-Stream Gating (new)'),
    (C_DEC_L, C_DEC, 'Decoder'),
    (C_SITE_L, C_SITE, 'Adversarial (new)'),
    (C_VOL_L, C_VOL, 'Volume Head (new)'),
    (C_FEAT_L, C_FEAT, 'Feature Consistency (new)'),
]
for i, (fc, ec, label) in enumerate(legend_items):
    y = 3.8 - i * 0.35
    rect = FancyBboxPatch((0.8, y-0.12), 0.5, 0.24, boxstyle="round,pad=0.05",
                          facecolor=fc, edgecolor=ec, linewidth=1)
    ax.add_patch(rect)
    ax.text(1.5, y, label, ha='left', va='center', fontsize=7, color=C_TEXT)

plt.tight_layout()
out = '/home/vlm/minyoung2/experiments/ReproSeg/figures/architecture/reproseg_v1_architecture.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')

# Also save PDF
out_pdf = out.replace('.png', '.pdf')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_pdf}')
