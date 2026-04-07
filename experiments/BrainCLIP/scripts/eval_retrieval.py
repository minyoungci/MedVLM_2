"""
eval_retrieval.py
BrainCLIP Retrieval 평가 — Recall@K (MRI→Text, Text→MRI)

Usage:
    cd /home/vlm/minyoung2
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
        experiments/BrainCLIP/scripts/eval_retrieval.py \
        --ckpt experiments/BrainCLIP/results/exp01_baseline/checkpoints/brainclip_best.pt \
        --split test \
        --topk 1 5 10
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

MANIFEST  = Path("/home/vlm/data/metadata/v4_manifest.csv")


@torch.no_grad()
def build_embeddings(model: BrainCLIP, loader: DataLoader, device: torch.device):
    """전체 split의 MRI/text embedding 수집"""
    mri_embs  = []
    text_embs = []
    sids      = []
    diags     = []

    for batch in loader:
        mri = batch["mri"].to(device).to(torch.bfloat16)
        texts = batch["text"]

        m = model.encode_mri(mri)
        t = model.encode_text(texts)

        mri_embs.append(m.float().cpu())
        text_embs.append(t.float().cpu())
        sids.extend(batch["subject_id"])
        diags.extend(batch["diagnosis"])

    return (
        torch.cat(mri_embs,  dim=0),   # [N, D]
        torch.cat(text_embs, dim=0),   # [N, D]
        sids,
        diags,
    )


def recall_at_k(query: torch.Tensor, gallery: torch.Tensor, ks: list[int]) -> dict:
    """
    query:   [N, D] L2-normalized
    gallery: [N, D] L2-normalized  (query[i]의 positive = gallery[i])
    반환: {k: recall@k_value}
    """
    sim = query @ gallery.T  # [N, N]
    N = sim.size(0)
    labels = torch.arange(N)

    results = {}
    for k in ks:
        topk_idx = sim.topk(k, dim=1).indices  # [N, k]
        hit = (topk_idx == labels.unsqueeze(1)).any(dim=1).float()
        results[f"R@{k}"] = hit.mean().item()
    return results


def random_baseline(N: int, ks: list[int]) -> dict:
    return {f"R@{k}": k / N for k in ks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   type=Path, required=True)
    parser.add_argument("--texts_csv", type=Path,
                        default=Path("experiments/BrainCLIP/data/clinical_texts.csv"))
    parser.add_argument("--embedding_cache_dir", type=Path, default=None)
    parser.add_argument("--split",  type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--topk",   type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  ckpt={args.ckpt}")

    # 모델 로드 (bf16 — B200 환경)
    model = BrainCLIP().to(device).to(torch.bfloat16)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    # 데이터
    ds = BrainCLIPDataset(
        texts_csv=args.texts_csv,
        split=args.split,
        embedding_cache_dir=args.embedding_cache_dir,
        augment=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"[eval] {args.split} set: {len(ds)} subjects")

    mri_emb, text_emb, sids, diags = build_embeddings(model, loader, device)
    N = len(sids)
    print(f"[eval] embedding 수집 완료: {N} pairs  shape={mri_emb.shape}")

    ks = sorted(args.topk)

    # MRI→Text
    m2t = recall_at_k(mri_emb, text_emb, ks)
    # Text→MRI
    t2m = recall_at_k(text_emb, mri_emb, ks)
    # Random baseline
    rand = random_baseline(N, ks)

    print(f"\n{'─'*50}")
    print(f"  N={N}  split={args.split}")
    print(f"{'─'*50}")
    print(f"  {'Metric':<12}  {'MRI→Text':>10}  {'Text→MRI':>10}  {'Random':>10}")
    for k in ks:
        key = f"R@{k}"
        print(f"  {key:<12}  {m2t[key]:>10.4f}  {t2m[key]:>10.4f}  {rand[key]:>10.4f}")
    print(f"{'─'*50}")

    # 진단군별 Recall@1 분석
    diag_set = sorted(set(diags))
    if len(diag_set) <= 10:
        print(f"\n진단군별 MRI→Text Recall@1:")
        for dg in diag_set:
            idx = [i for i, d in enumerate(diags) if d == dg]
            if len(idx) < 2:
                continue
            sub_query   = mri_emb[idx]
            sub_gallery = text_emb[idx]
            # 전체 gallery 대비 local recall
            sim_sub = (sub_query @ text_emb.T)  # [n, N]
            hits = (sim_sub.topk(1, dim=1).indices.squeeze(1) == torch.tensor(idx)).float()
            r1 = hits.mean().item()
            print(f"  {dg:<40}  n={len(idx):4d}  R@1={r1:.4f}")

    # 저장
    out_dir = args.ckpt.parent.parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"retrieval_{args.split}.txt"
    with open(result_path, "w") as f:
        f.write(f"split={args.split}  N={N}\n")
        f.write(f"{'Metric':<12}  {'MRI->Text':>10}  {'Text->MRI':>10}  {'Random':>10}\n")
        for k in ks:
            key = f"R@{k}"
            f.write(f"{key:<12}  {m2t[key]:>10.4f}  {t2m[key]:>10.4f}  {rand[key]:>10.4f}\n")
    print(f"\n결과 저장: {result_path}")


if __name__ == "__main__":
    main()
