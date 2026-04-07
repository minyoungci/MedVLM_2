"""
generate_report.py
BrainCLIP 실험 결과 → 학습 곡선 figure + Markdown 보고서 생성

Usage:
    cd /home/vlm/minyoung2
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
        experiments/BrainCLIP/scripts/generate_report.py \
        --result_dir experiments/BrainCLIP/results \
        --output experiments/BrainCLIP/results/BrainCLIP_Report.md
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── 데이터 로더 ───────────────────────────────────────────────────────────────

def load_train_log(log_path: Path) -> pd.DataFrame | None:
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    return df

def load_retrieval(txt_path: Path) -> dict | None:
    """retrieval_test.txt 파싱 → {R@1_m2t, R@1_t2m, R@5_m2t, ...}"""
    if not txt_path.exists():
        return None
    result = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("R@"):
                parts = line.split()
                key = parts[0]  # e.g. R@1
                try:
                    result[f"{key}_mri2text"] = float(parts[1])
                    result[f"{key}_text2mri"] = float(parts[2])
                    result[f"{key}_random"]   = float(parts[3])
                except (IndexError, ValueError):
                    pass
            if line.startswith("split="):
                for token in line.split():
                    if token.startswith("N="):
                        result["N"] = int(token[2:])
    return result or None

def load_linear_probe(txt_path: Path) -> dict | None:
    """linear_probe_results.txt 파싱 → {brainclip/backbone: {acc, bacc, auroc}}"""
    if not txt_path.exists():
        return None
    result = {}
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] not in ("Mode",):
                mode, acc, bacc, auroc = parts
                try:
                    result[mode] = {
                        "accuracy": float(acc),
                        "balanced_accuracy": float(bacc),
                        "auroc": float(auroc),
                    }
                except ValueError:
                    pass
    return result or None

def load_pipeline_state(result_dir: Path) -> dict:
    state_file = result_dir / "pipeline_state.json"
    if state_file.exists():
        return json.load(open(state_file))
    return {}

# ── Figure 생성 ───────────────────────────────────────────────────────────────

COLORS = {"exp01": "#2196F3", "exp02": "#FF5722"}
EXP_LABELS = {
    "exp01_baseline":    "EXP01 (w/ diagnosis)",
    "exp02_no_diag_text": "EXP02 (no diagnosis)",
}

def fig_learning_curves(logs: dict, fig_path: Path):
    """Loss, R@1, Temperature, Grad Norm 4-panel 학습 곡선"""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("BrainCLIP — Learning Curves", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    titles    = ["Contrastive Loss", "In-Batch R@1 (MRI→Text)", "Temperature τ", "Gradient Norm"]
    train_cols = ["loss", "recall_at_1_mri2text", "temperature", "grad_norm"]
    val_cols   = ["loss", "recall_at_1_mri2text", "temperature", None]

    for ax, title, tc, vc in zip(axes, titles, train_cols, val_cols):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

        for exp_name, df in logs.items():
            if df is None:
                continue
            color = COLORS.get(exp_name.split("_")[0] + "_" + exp_name.split("_")[1]
                               if "_" in exp_name else exp_name, "#888")
            # epoch-level rows (step==-1)
            epoch_df = df[df["step"] == -1].copy()
            if epoch_df.empty:
                # fallback: step-level 데이터로 rolling mean
                epoch_df = df.copy()

            label_base = EXP_LABELS.get(exp_name, exp_name)

            # train
            tr = epoch_df[epoch_df["split"] == "train"]
            if not tr.empty and tc in tr.columns:
                vals = pd.to_numeric(tr[tc], errors="coerce").dropna()
                if not vals.empty:
                    ax.plot(range(len(vals)), vals.values,
                            color=color, label=f"{label_base} train", linewidth=2)

            # val
            if vc is not None:
                vl = epoch_df[epoch_df["split"] == "val"]
                if not vl.empty and vc in vl.columns:
                    vals = pd.to_numeric(vl[vc], errors="coerce").dropna()
                    if not vals.empty:
                        ax.plot(range(len(vals)), vals.values,
                                color=color, linestyle="--",
                                label=f"{label_base} val", linewidth=2, alpha=0.7)

        ax.legend(fontsize=8, loc="best")

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def fig_retrieval_comparison(retrievals: dict, fig_path: Path):
    """Recall@K bar chart (MRI→Text)"""
    ks = ["R@1", "R@5", "R@10"]
    exp_names = list(retrievals.keys())
    n_exp = len(exp_names)
    if n_exp == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("BrainCLIP — Retrieval Recall@K", fontsize=13, fontweight="bold")

    for ax_idx, direction in enumerate(["mri2text", "text2mri"]):
        ax = axes[ax_idx]
        ax.set_title(f"{'MRI → Text' if direction == 'mri2text' else 'Text → MRI'}", fontsize=11)
        ax.set_xlabel("K")
        ax.set_ylabel("Recall@K")
        ax.grid(True, alpha=0.3, axis="y")

        x = np.arange(len(ks))
        width = 0.25
        offsets = np.linspace(-(n_exp - 1) * width / 2, (n_exp - 1) * width / 2, n_exp)

        for i, (exp_name, ret) in enumerate(retrievals.items()):
            if ret is None:
                continue
            label = EXP_LABELS.get(exp_name, exp_name)
            color = list(COLORS.values())[i % len(COLORS)]
            vals = [ret.get(f"{k}_{direction}", 0.0) for k in ks]
            rand = [ret.get(f"{k}_random",     0.0) for k in ks]
            bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # random baseline (첫 exp 기준)
        first_ret = next(iter(retrievals.values()))
        if first_ret:
            rand_vals = [first_ret.get(f"{k}_random", 0) for k in ks]
            ax.plot(x, rand_vals, "k--", marker="x", label="Random baseline", linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(ks)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def fig_linear_probe(probes: dict, fig_path: Path):
    """Linear probe BAcc/AUROC grouped bar"""
    if not probes:
        return

    metrics = ["balanced_accuracy", "auroc"]
    metric_labels = ["Balanced Accuracy", "AUROC (macro OvR)"]
    # 모든 (exp, mode) 조합
    entries = []
    for exp_name, modes in probes.items():
        if modes is None:
            continue
        for mode, vals in modes.items():
            label = f"{EXP_LABELS.get(exp_name, exp_name)}\n({mode})"
            entries.append((label, vals))

    if not entries:
        return

    n = len(entries)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("BrainCLIP — Linear Probe (CN/MCI/AD)", fontsize=13, fontweight="bold")

    x = np.arange(n)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        ax.set_title(mlabel, fontsize=11)
        ax.set_ylabel(mlabel)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(1/3, color="gray", linestyle=":", linewidth=1, label="Chance (1/3)")

        vals = [entry[1].get(metric, 0.0) for entry in entries]
        bars = ax.bar(x, vals, color=colors, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in entries], fontsize=8, rotation=15, ha="right")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Markdown 보고서 ───────────────────────────────────────────────────────────

def fmt_retrieval_table(retrievals: dict) -> str:
    if not retrievals:
        return "_결과 없음_\n"
    ks = ["R@1", "R@5", "R@10"]
    lines = ["| EXP | Direction | " + " | ".join(ks) + " | vs Random R@1 |",
             "|-----|-----------|" + "|".join([":---:"] * len(ks)) + "|:---:|"]
    for exp_name, ret in retrievals.items():
        if ret is None:
            continue
        label = EXP_LABELS.get(exp_name, exp_name)
        rand1 = ret.get("R@1_random", 0)
        for direction, dir_label in [("mri2text", "MRI→Text"), ("text2mri", "Text→MRI")]:
            vals = [f"{ret.get(f'{k}_{direction}', 0):.4f}" for k in ks]
            r1 = ret.get("R@1_mri2text" if direction == "mri2text" else "R@1_text2mri", 0)
            vs_rand = f"+{r1/rand1:.1f}×" if rand1 > 0 else "—"
            lines.append(f"| {label} | {dir_label} | " + " | ".join(vals) + f" | {vs_rand} |")
    return "\n".join(lines) + "\n"

def fmt_probe_table(probes: dict) -> str:
    if not probes:
        return "_결과 없음_\n"
    lines = ["| EXP | Mode | Accuracy | BAcc | AUROC | vs BrainIAC-only |",
             "|-----|------|:--------:|:----:|:-----:|:----------------:|"]
    backbone_bacc = {}
    for exp_name, modes in probes.items():
        if modes and "backbone" in modes:
            backbone_bacc[exp_name] = modes["backbone"]["balanced_accuracy"]

    for exp_name, modes in probes.items():
        if modes is None:
            continue
        label = EXP_LABELS.get(exp_name, exp_name)
        for mode, vals in modes.items():
            acc   = vals.get("accuracy", 0)
            bacc  = vals.get("balanced_accuracy", 0)
            auroc = vals.get("auroc", 0)
            if mode == "backbone":
                vs = "— (baseline)"
            else:
                base = backbone_bacc.get(exp_name, None)
                vs = f"{bacc - base:+.4f}" if base else "—"
            lines.append(f"| {label} | {mode} | {acc:.4f} | {bacc:.4f} | {auroc:.4f} | {vs} |")
    return "\n".join(lines) + "\n"

def analyze_issues(logs: dict, retrievals: dict, probes: dict) -> tuple[list, list]:
    """관찰된 문제점과 개선 제안 도출"""
    issues = []
    improvements = []

    # Retrieval 분석
    for exp_name, ret in retrievals.items():
        if ret is None:
            issues.append(f"**{EXP_LABELS.get(exp_name, exp_name)}**: 평가 결과 없음")
            continue
        r1 = ret.get("R@1_mri2text", 0)
        rand = ret.get("R@1_random", 0)
        if r1 < rand * 2:
            issues.append(
                f"**{EXP_LABELS.get(exp_name, exp_name)}**: "
                f"R@1 ({r1:.4f})이 random baseline ({rand:.4f})의 2배 미만 — "
                "contrastive alignment이 거의 학습되지 않음"
            )
            improvements.append(
                "배치 크기 증가 (32→64) 또는 학습률 조정 검토. "
                "1,472 페어로 contrastive learning은 한계 — "
                "hard negative mining 또는 multi-crop augmentation 고려"
            )
        elif r1 < 0.05:
            issues.append(
                f"**{EXP_LABELS.get(exp_name, exp_name)}**: "
                f"R@1 {r1:.4f} — 성공 기준(5%) 미달"
            )
            improvements.append("Temperature schedule 또는 projection dim (128→256) 확대 검토")

    # Linear probe 비교
    for exp_name, modes in probes.items():
        if modes is None:
            continue
        backbone = modes.get("backbone", {})
        brainclip = modes.get("brainclip", {})
        if backbone and brainclip:
            delta_bacc = brainclip.get("balanced_accuracy", 0) - backbone.get("balanced_accuracy", 0)
            if delta_bacc < 0:
                issues.append(
                    f"**{EXP_LABELS.get(exp_name, exp_name)}**: "
                    f"BrainCLIP linear probe BAcc ({brainclip.get('balanced_accuracy',0):.4f}) < "
                    f"BrainIAC-only ({backbone.get('balanced_accuracy',0):.4f}) — "
                    f"contrastive 학습이 분류 표현을 오히려 저하시킴"
                )
                improvements.append(
                    "projection head 이전 표현으로 linear probe 재평가 검토 "
                    "(128-dim projection이 클래스 정보를 squeeze할 수 있음). "
                    "MRI encoder partial fine-tune (mri_fine_tune_layers=2) 실험 권장"
                )
            elif delta_bacc > 0.02:
                improvements.append(
                    f"**{EXP_LABELS.get(exp_name, exp_name)}** linear probe 개선 확인 "
                    f"(ΔBAcc={delta_bacc:+.4f}). EXP 03: MRI fine-tune 실험으로 상한 확인 권장"
                )

    # Learning curve 분석
    for exp_name, df in logs.items():
        if df is None:
            continue
        epoch_df = df[(df["step"] == -1) & (df["split"] == "train")]
        if epoch_df.empty:
            continue
        losses = pd.to_numeric(epoch_df["loss"], errors="coerce").dropna().values
        if len(losses) > 5:
            last5_mean = losses[-5:].mean()
            first5_mean = losses[:5].mean()
            if last5_mean > first5_mean * 0.98:
                issues.append(
                    f"**{EXP_LABELS.get(exp_name, exp_name)}**: "
                    "학습 곡선이 수렴하지 않음 (loss 감소 미미). "
                    "warmup 비율 또는 lr 재조정 필요"
                )
            if "grad_norm" in epoch_df.columns:
                gnorms = pd.to_numeric(epoch_df["grad_norm"], errors="coerce").dropna().values
                if len(gnorms) > 0 and gnorms.mean() < 0.01:
                    issues.append(
                        f"**{EXP_LABELS.get(exp_name, exp_name)}**: "
                        f"평균 gradient norm {gnorms.mean():.4f} — vanishing gradient 의심"
                    )

    # EXP01 vs EXP02 비교
    r_01 = retrievals.get("exp01_baseline", {}) or {}
    r_02 = retrievals.get("exp02_no_diag_text", {}) or {}
    if r_01 and r_02:
        diff = r_01.get("R@1_mri2text", 0) - r_02.get("R@1_mri2text", 0)
        if diff > 0.02:
            improvements.append(
                f"EXP01 R@1 > EXP02 R@1 by {diff:.4f}: 진단 레이블이 retrieval에 강한 신호 확인. "
                "논문 contribution 명확화: EXP02 결과를 '진단 독립 MRI 표현' 지표로 강조"
            )
        else:
            improvements.append(
                "EXP01/EXP02 R@1 차이 미미: CDR/GDS 등 임상 지표만으로도 MRI alignment 가능성 — "
                "긍정적 신호. EXP02 linear probe가 EXP01보다 높다면 강력한 contribution"
            )

    if not issues:
        issues.append("주요 문제 없음 — 결과 검토 후 EXP 03 (MRI partial fine-tune) 진행 권장")
    if not improvements:
        improvements.append("현재 결과에서 구체적 개선 방향 도출 불가 — 추가 에폭 학습 필요")

    return issues, improvements


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=Path,
                        default=Path("experiments/BrainCLIP/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/BrainCLIP/results/BrainCLIP_Report.md"))
    args = parser.parse_args()

    rdir = args.result_dir
    fig_dir = rdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 수집 ────────────────────────────────────────────────────────────
    exps = {
        "exp01_baseline":     rdir / "exp01_baseline",
        "exp02_no_diag_text": rdir / "exp02_no_diag_text",
    }

    logs = {}
    retrievals = {}
    probes = {}

    for exp_name, exp_dir in exps.items():
        logs[exp_name]       = load_train_log(exp_dir / "logs" / "train_log.csv")
        retrievals[exp_name] = load_retrieval(exp_dir / "logs" / "retrieval_test.txt")
        probes[exp_name]     = load_linear_probe(exp_dir / "logs" / "linear_probe_results.txt")

    state = load_pipeline_state(rdir)

    # ── Figure 생성 ────────────────────────────────────────────────────────────
    any_log = any(v is not None for v in logs.values())
    any_ret = any(v is not None for v in retrievals.values())
    any_probe = any(v is not None for v in probes.values())

    fig_lc   = fig_dir / "learning_curves.png"
    fig_ret  = fig_dir / "retrieval_comparison.png"
    fig_prob = fig_dir / "linear_probe.png"

    if any_log:
        fig_learning_curves(logs, fig_lc)
        print(f"[report] learning_curves.png 생성")
    if any_ret:
        fig_retrieval_comparison(retrievals, fig_ret)
        print(f"[report] retrieval_comparison.png 생성")
    if any_probe:
        fig_linear_probe(probes, fig_prob)
        print(f"[report] linear_probe.png 생성")

    # ── 문제점/개선점 분석 ─────────────────────────────────────────────────────
    issues, improvements = analyze_issues(logs, retrievals, probes)

    # ── Markdown 작성 ──────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    pipeline_phase = state.get("phase", "unknown")

    md = f"""# BrainCLIP 실험 보고서

**생성 일시**: {now}
**파이프라인 상태**: `{pipeline_phase}`

---

## 1. 실험 개요

| 항목 | EXP01 baseline | EXP02 no_diag_text |
|------|:--------------:|:------------------:|
| 텍스트 | 진단 포함 (age/CDR/GDS/**diag**) | 진단 **제외** (age/CDR/GDS만) |
| 학습 목적 | alignment 품질 확인 | MRI 독립 표현 학습 |
| Linear probe 해석 | label alignment 측정 | **진단 독립 MRI 표현** 측정 |
| 배치 크기 | 32 | 32 |
| Epochs | 30 | 30 |
| MRI encoder | BrainIAC frozen (768→128 proj) | BrainIAC frozen |
| Text encoder | PubMedBERT top-2 layers | PubMedBERT top-2 layers |

---

## 2. 학습 곡선

"""
    if any_log:
        md += f"![Learning Curves](figures/learning_curves.png)\n\n"
        # epoch summary table
        md += "### Epoch Summary (Last Epoch)\n\n"
        md += "| EXP | Train Loss | Val Loss | Val R@1 | Temp τ | Grad Norm |\n"
        md += "|-----|:----------:|:--------:|:-------:|:------:|:---------:|\n"
        for exp_name, df in logs.items():
            if df is None:
                md += f"| {EXP_LABELS.get(exp_name, exp_name)} | — | — | — | — | — |\n"
                continue
            epoch_df = df[df["step"] == -1]
            tr = epoch_df[epoch_df["split"] == "train"]
            vl = epoch_df[epoch_df["split"] == "val"]
            tr_loss = f"{pd.to_numeric(tr['loss'], errors='coerce').iloc[-1]:.4f}" if not tr.empty else "—"
            vl_loss = f"{pd.to_numeric(vl['loss'], errors='coerce').iloc[-1]:.4f}" if not vl.empty else "—"
            vl_r1   = f"{pd.to_numeric(vl['recall_at_1_mri2text'], errors='coerce').iloc[-1]:.4f}" if not vl.empty else "—"
            tau     = f"{pd.to_numeric(vl['temperature'], errors='coerce').iloc[-1]:.4f}" if not vl.empty else "—"
            gnorm   = f"{pd.to_numeric(tr['grad_norm'], errors='coerce').iloc[-1]:.4f}" if (not tr.empty and 'grad_norm' in tr.columns) else "—"
            label   = EXP_LABELS.get(exp_name, exp_name)
            md += f"| {label} | {tr_loss} | {vl_loss} | {vl_r1} | {tau} | {gnorm} |\n"
        md += "\n"
    else:
        md += "_학습 로그 없음 (학습 미완료)_\n\n"

    md += """---

## 3. Retrieval 성능 (Recall@K)

"""
    md += fmt_retrieval_table(retrievals)
    if any_ret:
        md += f"\n![Retrieval Comparison](figures/retrieval_comparison.png)\n"
    md += "\n"

    md += """---

## 4. Linear Probe (CN/MCI/AD)

"""
    md += fmt_probe_table(probes)
    if any_probe:
        md += f"\n![Linear Probe](figures/linear_probe.png)\n"

    md += """
> ⚠ **해석 주의**:
> - EXP01 BrainCLIP mode: "진단 label ↔ MRI alignment" 품질 측정
> - EXP02 BrainCLIP mode: "진단 독립 MRI 표현 학습" 품질 측정 — 논문 핵심 claim
> - backbone mode (BrainIAC-only): contrastive 학습 전 baseline

---

## 5. 문제점

"""
    for i, issue in enumerate(issues, 1):
        md += f"{i}. {issue}\n"

    md += """
---

## 6. 개선 방향

"""
    for i, imp in enumerate(improvements, 1):
        md += f"{i}. {imp}\n"

    md += f"""
---

## 7. 다음 실험 제안

| EXP | 설명 | 우선순위 | 예상 소요 |
|-----|------|:--------:|:--------:|
| EXP03 | MRI fine-tune (mri_fine_tune_layers=2) | 🔴 High | ~3h |
| EXP04 | Batch size 64 (in-batch negatives 2× 증가) | 🟡 Medium | ~3h |
| EXP05 | AJU zero-shot retrieval 평가 (학습 없음) | 🟡 Medium | ~30min |
| EXP06 | projection dim 128→256 + hidden 512 | 🔵 Low | ~3h |

---

## 8. 파이프라인 로그 위치

```
results/pipeline.log         — 전체 파이프라인 실행 로그
results/pipeline_state.json  — 현재 phase 상태
results/exp01_baseline/logs/train_log.csv
results/exp02_no_diag_text/logs/train_log.csv
results/BrainCLIP_Report.md  — 이 보고서
```

_보고서 자동 생성 — generate_report.py_
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"[report] 보고서 저장: {args.output}")
    print(f"[report] 문제점 {len(issues)}개, 개선점 {len(improvements)}개")


if __name__ == "__main__":
    main()
