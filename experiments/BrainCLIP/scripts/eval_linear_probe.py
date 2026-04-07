"""
eval_linear_probe.py
BrainCLIP MRI embedding → CN/MCI/AD linear probe 분류

BrainIAC-only baseline (projection head 없이 raw 768-dim) vs BrainCLIP projected 128-dim 비교

⚠ 해석 주의: clinical text에 "Clinical diagnosis: ..." 포함됨.
  BrainCLIP linear probe 성능은 "MRI 구조로부터 독립적으로 학습된 표현"이 아니라
  "텍스트 내 진단 레이블과 MRI 표현의 alignment 품질"을 반영함.
  논문 claim 시: "label-aligned representation learning" 으로 명시 필요.
  ablation 권장: diagnosis 제거한 텍스트(age/CDR만) 버전 별도 실험.

Usage:
    cd /home/vlm/minyoung2
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
        experiments/BrainCLIP/scripts/eval_linear_probe.py \
        --ckpt experiments/BrainCLIP/results/exp01_baseline/checkpoints/brainclip_best.pt \
        --mode both
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataset_brainclip import BrainCLIPDataset, collate_fn
from model_brainclip import BrainCLIP

DIAGNOSIS_LABELS = {
    "cognitively normal": 0,
    "CN": 0,
    "mild cognitive impairment": 1,
    "MCI": 1,
    "Alzheimer's dementia": 2,
    "AD": 2,
    "Alzheimer's dementia with other conditions": 2,
    "Dementia": 2,
    "dementia": 2,
}
LABEL_NAMES = ["CN", "MCI", "AD"]


@torch.no_grad()
def collect_features(model: BrainCLIP, loader: DataLoader, device: torch.device,
                     mode: str = "brainclip"):
    """
    mode='brainclip'  → BrainCLIP MRI projection head 출력 [B, 128]
    mode='backbone'   → BrainIAC raw global_embedding [B, 768] (projection 없이)
    """
    feats  = []
    labels = []

    for batch in loader:
        mri = batch["mri"].to(device).to(torch.bfloat16)
        diag = batch["diagnosis"]

        if mode == "brainclip":
            f = model.encode_mri(mri).float().cpu()
        else:
            # raw BrainIAC without projection
            raw_model = model.mri_encoder
            if mri.dim() == 5:
                f = raw_model.backbone(mri).global_embedding.float().cpu()
            else:
                f = mri.float().cpu()

        feats.append(f)

        for d in diag:
            lbl = DIAGNOSIS_LABELS.get(str(d).strip(), -1)
            labels.append(lbl)

    X = torch.cat(feats, dim=0).numpy()
    y = np.array(labels)

    # 라벨 없는 샘플 제거
    valid = y >= 0
    return X[valid], y[valid]


def run_linear_probe(X_train, y_train, X_test, y_test) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, roc_auc_score
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_tr, y_train)
    y_pred  = clf.predict(X_te)
    y_prob  = clf.predict_proba(X_te)

    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    # AUROC: one-vs-rest, macro average
    try:
        auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    return {"accuracy": acc, "balanced_accuracy": bacc, "auroc": auroc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--texts_csv", type=Path,
                        default=Path("experiments/BrainCLIP/data/clinical_texts.csv"))
    parser.add_argument("--embedding_cache_dir", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["brainclip", "backbone", "both"],
                        help="brainclip=projected 128d / backbone=raw 768d BrainIAC")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[linear_probe] device={device}  ckpt={args.ckpt}")

    model = BrainCLIP().to(device).to(torch.bfloat16)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    def make_loader(split):
        ds = BrainCLIPDataset(
            texts_csv=args.texts_csv,
            split=split,
            embedding_cache_dir=args.embedding_cache_dir,
            augment=False,
        )
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=collate_fn, drop_last=False,
        )

    train_loader = make_loader("train")
    test_loader  = make_loader("test")

    modes = ["brainclip", "backbone"] if args.mode == "both" else [args.mode]
    results = {}

    for mode in modes:
        print(f"\n[{mode}] feature 수집 중...")
        X_train, y_train = collect_features(model, train_loader, device, mode)
        X_test,  y_test  = collect_features(model, test_loader,  device, mode)
        print(f"  train: {X_train.shape}  labels={np.bincount(y_train)}")
        print(f"  test:  {X_test.shape}   labels={np.bincount(y_test)}")

        res = run_linear_probe(X_train, y_train, X_test, y_test)
        results[mode] = res
        print(f"  Accuracy:          {res['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {res['balanced_accuracy']:.4f}")
        print(f"  AUROC (macro OvR): {res['auroc']:.4f}")

    print(f"\n{'─'*55}")
    print(f"  {'Mode':<14}  {'Acc':>8}  {'BAcc':>8}  {'AUROC':>8}")
    for mode, res in results.items():
        print(f"  {mode:<14}  {res['accuracy']:>8.4f}  {res['balanced_accuracy']:>8.4f}  {res['auroc']:>8.4f}")
    if len(results) == 2:
        delta_bacc  = results["brainclip"]["balanced_accuracy"] - results["backbone"]["balanced_accuracy"]
        delta_auroc = results["brainclip"]["auroc"] - results["backbone"]["auroc"]
        print(f"  {'Δ (clip-base)':<14}  {'':>8}  {delta_bacc:>+8.4f}  {delta_auroc:>+8.4f}")
    print(f"{'─'*55}")

    # 저장
    out_dir = args.ckpt.parent.parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "linear_probe_results.txt"
    with open(out_path, "w") as f:
        f.write(f"{'Mode':<14}  {'Acc':>8}  {'BAcc':>8}  {'AUROC':>8}\n")
        for mode, res in results.items():
            f.write(f"{mode:<14}  {res['accuracy']:>8.4f}  {res['balanced_accuracy']:>8.4f}  {res['auroc']:>8.4f}\n")
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
