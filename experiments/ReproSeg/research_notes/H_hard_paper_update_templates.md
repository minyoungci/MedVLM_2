# H_hard Ablation — Paper Update Templates

**Status**: Pre-drafted 2026-04-19 during training.
**Purpose**: Drop-in LaTeX snippets for each of the three result scenarios so the paper can be updated immediately once `bootstrap_H_vs_H_hard.json` is available.

All three scenarios assume the H_hard checkpoint reaches a stable ICC value on the 270-pair held-out set.

---

## Where to apply

### Main table (inline in `main.tex`, around line 718)
Insert one new row between H and C in the ablation table:
```
A & SwinUNETR baseline              & 0.837 & —       \\
B & + Temporal consistency loss     & 0.800 & $-0.037$ \\
D & + Volume prediction head        & 0.836 & $-0.001$ \\
E & + Invariance stream + PCGrad    & 0.864 & $+0.027$ \\
F & + CSG $+$ GRL $+$ PCGrad        & 0.751 & $-0.086$ \\
G & + GRL only                      & 0.932 & $+0.095$ \\
H & + CSG only (learned gate)       & 0.932 & $+0.095$ \\
H$_\mathrm{hard}$ & + CSG with $g\equiv1$ (hard projection) & <VALUE> & <DELTA> \\
C & + CSG + GRL                     & 0.891 & $+0.054$ \\
```

### Ablation figure
Add one bar to `fig_ablation_progression.png` after H, labeled "H_hard (g=1)".

### Supplementary: per-structure ICC (Table S1)
Add one column for H_hard values, after H column.

### Discussion section
Update Sec 6.1 ("CSG and GRL likely address overlapping nuisance variation")
based on scenario.

---

## Scenario 1 — H_hard < H (Δ ≤ −0.02): learned gate is essential

### Interpretation (for Discussion Sec 6.1 insert)
> A further ablation (model H$_\mathrm{hard}$) replaced the learned gate with
> a hard projection ($g\equiv1$), removing the full invariance-direction
> component at every voxel. This degraded revisit ICC from 0.932 to
> <VALUE> ($\Delta = $<DELTA>, 95\% CI [<LO>, <HI>]), confirming that the
> learned gate is a load-bearing component of CSG: it lets the model
> preserve anatomical signal that happens to align with the
> invariance direction and apply the removal selectively where the
> invariance stream carries genuine scanner noise.

### Novelty claim (for Introduction contributions)
Add one sub-bullet under the CSG contribution:
> \item The learned per-voxel gate is essential rather than cosmetic:
>       a hard-projection variant (H$_\mathrm{hard}$) underperforms the
>       learned-gate variant by <DELTA> ICC points on the same held-out set.

### Table 1 row (optional, only if improvement is large)
Add a row to `table1_main_results.tex`:
```
Gating ablation & H$_\mathrm{hard}$ (hard projection) vs H & <VALUE> vs 0.932; $\Delta = $<DELTA> [<LO>, <HI>] & Learned gate is load-bearing \\
```

---

## Scenario 2 — H_hard ≈ H (|Δ| < 0.02): learned gate optional

### Honest reframing (replaces current Sec 6.1 opening)
> The most robust mechanistic observation in this study is not the existence
> of a formally proven ``invariance bottleneck'', but that a dual-stream
> architecture with any targeted removal of invariance-direction features
> stabilizes supervised volumetry. Three removal mechanisms reach nearly
> identical ICC on the held-out 270-pair set: (i) a learned per-voxel gate
> over orthogonal projection (H, CSG), (ii) gradient reversal against a
> site classifier (G, GRL), and (iii) a hard projection with
> $g\equiv1$ (H$_\mathrm{hard}$). Because the hard-projection variant
> matches the learned-gate variant within <DELTA> ICC points, the learned
> gate is not the source of the gain: the key ingredient is the
> invariance stream itself plus any mechanism that subtracts its
> contribution before the decoder.

### Contribution rewording (Introduction list)
Replace bullet 1 with:
> \item We show that a \emph{dual-stream} architecture — a 5.6M-parameter
>       invariance stream running in parallel with a SwinUNETR anatomy
>       stream — materially improves revisit consistency when combined with
>       any mechanism that suppresses the invariance stream's contribution
>       (learned gating, hard projection, or gradient reversal all reach
>       ICC $\approx$ 0.93 on the same held-out set).

### Title — consider changing
> Before: *Cross-Stream Gating Stabilizes Longitudinal Revisit Consistency*
> After: *Dual-Stream Invariance Subtraction Stabilizes Longitudinal
>        Revisit Consistency in Automated Brain Volumetry*

---

## Scenario 3 — H_hard > H (Δ ≥ +0.02): gate is harmful

### Honest reporting (new subsection in Results)
> Unexpectedly, replacing the learned gate with a hard projection
> ($g\equiv1$) improved ICC from 0.932 (H) to <VALUE> (H$_\mathrm{hard}$),
> $\Delta = $<DELTA> (95\% CI [<LO>, <HI>]). The learned gate therefore
> underperforms the simpler unconditional projection on this dataset.
> We interpret this as evidence that the gate entropy regulariser at
> $\lambda = 0.01$ systematically biases the gate toward intermediate
> values, whereas the anatomy stream actually benefits from the full
> invariance-direction subtraction. Future work should revisit the
> regulariser weight or remove the gate network entirely.

### Contribution rewording — simplify the method claim
> \item We show that a simple per-scale orthogonal projection from a
>       parallel invariance stream (5.6M parameters, no learned gate)
>       stabilizes supervised revisit consistency to ICC $= $ <VALUE> on the
>       270-pair held-out set, outperforming the learned-gate variant by
>       <DELTA> ICC points.

### Method section — simplify the CSG math
Remove the gate from Eqs 2-4, keeping only:
```
a_pure = a - c · î
```

---

## Checklist once H_hard ICC is known

1. [ ] Run `bash scripts/post_train_H_hard.sh 0` → produces `bootstrap_H_vs_H_hard.json`
2. [ ] Read observed Δ from the JSON
3. [ ] Pick scenario (1, 2, or 3) and apply the matching snippet
4. [ ] Update Table 2 inline in `main.tex` (line ~718) with the new H_hard row
5. [ ] Add per-structure column to Supp Table S1
6. [ ] Regenerate `fig_ablation_progression.png` (add H_hard bar)
7. [ ] Rebuild `main.pdf` and `supplementary.pdf` (two pdflatex passes)
8. [ ] Append result to `Final_check/04_notes/CONFIRMED_RESULTS.md` under a new
      "Section 8: Learned-gate ablation (H_hard)" subsection
9. [ ] Log the rerun in `Final_check/04_notes/RERUN_VERIFICATION_YYYY-MM-DD.md`
10. [ ] If scenario 2 or 3: update abstract and possibly paper title
