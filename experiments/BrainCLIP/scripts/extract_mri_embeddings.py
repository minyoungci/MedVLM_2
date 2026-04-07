"""
extract_mri_embeddings.py
BrainIAC frozen encoder로 MRI embedding을 오프라인 사전 추출 → .npy 파일로 저장

사전 추출 이점:
  - 학습 시 GPU에서 BrainIAC forward pass 스킵 → 배치당 ~4× 속도 향상
  - BrainIAC를 완전 freeze할 때만 유효 (fine_tune_layers > 0이면 사용 불가)

Usage:
    cd /home/vlm/minyoung2
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
        experiments/BrainCLIP/scripts/extract_mri_embeddings.py \
        --output_dir experiments/BrainCLIP/data/mri_embeddings \
        --split all \
        --batch_size 4
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/vlm/minyoung/model-claude/src")
from model.encoder_3d import BrainIACEncoder  # type: ignore

BRAINIAC_CKPT = Path("/home/vlm/minyoung/pretrain/brainiac/BrainIAC.ckpt")
V4_ROOT       = Path("/home/vlm/data/preprocessed_v4/cross_sectional")
MANIFEST      = Path("/home/vlm/data/metadata/v4_manifest.csv")


class RawMRIDataset(Dataset):
    """Manifest 기반 raw NIfTI 로더 (embedding 추출 전용)"""

    def __init__(self, manifest_csv: Path, splits: list[str]):
        manifest = pd.read_csv(manifest_csv)
        if splits != ["all"]:
            manifest = manifest[manifest["split"].isin(splits)]
        self.records = manifest.reset_index(drop=True)
        print(f"[RawMRIDataset] {len(self.records)} subjects ({splits})")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records.iloc[idx]
        nii_path = V4_ROOT / row["dataset"].upper() / row["subject_id"] / "native_t1w.nii.gz"
        img = nib.load(str(nii_path))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        return {
            "subject_id": row["subject_id"],
            "volume": torch.from_numpy(vol).unsqueeze(0),  # [1, D, H, W]
            "nii_path": str(nii_path),
        }


def collate_fn(batch: list) -> dict:
    ok = [b for b in batch if b["volume"] is not None]
    if not ok:
        return {}
    return {
        "subject_id": [b["subject_id"] for b in ok],
        "volume":     torch.stack([b["volume"] for b in ok]),
        "nii_path":   [b["nii_path"] for b in ok],
    }


@torch.no_grad()
def extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}")

    # BrainIAC 로드
    # V4 볼륨 크기: 192×224×192, patch_size=16 → 2016 패치
    # BrainIAC는 96³(216 패치)로 사전학습 → pos_embed interpolation 필수
    encoder = BrainIACEncoder(
        checkpoint_path=str(BRAINIAC_CKPT),
        img_size=(192, 224, 192),
        patch_size=(16, 16, 16),
        interpolate_pos_embed=True,
    ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("[extract] BrainIAC loaded and frozen")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    dataset = RawMRIDataset(MANIFEST, splits)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    errors = 0

    for batch in loader:
        if not batch:
            continue

        sids    = batch["subject_id"]
        volumes = batch["volume"].to(device)

        # 이미 캐싱된 subject 건너뜀
        to_process = [(i, sid) for i, sid in enumerate(sids)
                      if not (output_dir / f"{sid}.npy").exists()]

        if not to_process:
            skipped += len(sids)
            continue

        idxs  = [i for i, _ in to_process]
        sub_v = volumes[idxs]

        try:
            out = encoder(sub_v)
            embs = out.global_embedding.cpu().numpy()  # [B', 768]
        except Exception as e:
            print(f"  ERROR batch starting {sids[0]}: {e}")
            errors += len(idxs)
            continue

        for j, (_, sid) in enumerate(to_process):
            np.save(output_dir / f"{sid}.npy", embs[j])

        done += len(to_process)
        if done % 50 == 0 or done <= len(to_process):
            print(f"  {done}/{len(dataset)} saved  (skipped={skipped}, errors={errors})")

    print(f"\n완료: saved={done}, skipped={skipped}, errors={errors}")
    print(f"출력 디렉토리: {output_dir}")

    # 검증
    npy_files = list(output_dir.glob("*.npy"))
    if npy_files:
        sample = np.load(npy_files[0])
        print(f"샘플 embedding shape: {sample.shape}  dtype: {sample.dtype}")
        assert sample.shape == (768,), f"Expected (768,), got {sample.shape}"
        print("shape 검증 통과")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path,
                        default=Path("experiments/BrainCLIP/data/mri_embeddings"))
    parser.add_argument("--split",       type=str, default="all",
                        choices=["all", "train", "val", "test"])
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
