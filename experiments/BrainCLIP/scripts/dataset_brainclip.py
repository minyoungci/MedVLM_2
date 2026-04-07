"""
dataset_brainclip.py
BrainCLIP Dataset — MRI (3D T1w) + Clinical Text 페어 로더

MRI: BrainIAC frozen encoder로 사전 추출된 embedding 또는 raw NIfTI 직접 로드
Text: clinical_texts.csv의 자연어 문자열
"""

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── 경로 상수 ──────────────────────────────────────────────────────────────
V4_ROOT  = Path("/home/vlm/data/preprocessed_v4/cross_sectional")
MANIFEST = Path("/home/vlm/data/metadata/v4_manifest.csv")


class BrainCLIPDataset(Dataset):
    """
    MRI (T1w NIfTI or cached embedding) + Clinical Text 페어 데이터셋.

    Args:
        texts_csv: build_clinical_text.py 출력 CSV (subject_id, clinical_text, diagnosis)
        manifest_csv: v4_manifest.csv
        split: "train" | "val" | "test"
        embedding_cache_dir: 미리 추출된 BrainIAC embedding .npy 디렉토리 (없으면 NIfTI 로드)
        augment: L-R flip + intensity jitter (train 전용)
        max_text_len: tokenizer max length
    """

    def __init__(
        self,
        texts_csv: Path,
        manifest_csv: Path = MANIFEST,
        split: str = "train",
        embedding_cache_dir: Optional[Path] = None,
        augment: bool = False,
        max_text_len: int = 128,
    ):
        self.augment = augment
        self.max_text_len = max_text_len
        self.embedding_cache_dir = embedding_cache_dir

        manifest = pd.read_csv(manifest_csv)
        manifest = manifest[manifest["split"] == split]

        texts = pd.read_csv(texts_csv)
        # manifest × texts 내적
        merged = manifest.merge(texts, on="subject_id", how="inner")
        # 결측 텍스트 제거
        merged = merged[merged["clinical_text"].str.len() > 30].reset_index(drop=True)

        self.records = merged
        print(f"[BrainCLIPDataset] split={split}  N={len(self.records)}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_mri(self, row: pd.Series) -> torch.Tensor:
        """NIfTI → float32 tensor [1, D, H, W]"""
        dataset = row["dataset_x"] if "dataset_x" in row.index else row["dataset"]
        sid = row["subject_id"]
        nii_path = V4_ROOT / dataset.upper() / sid / "native_t1w.nii.gz"

        img = nib.load(str(nii_path))
        vol = np.asarray(img.dataobj, dtype=np.float32)  # (192, 224, 192)
        return torch.from_numpy(vol).unsqueeze(0)  # [1, D, H, W]

    def _load_embedding(self, row: pd.Series) -> Optional[torch.Tensor]:
        if self.embedding_cache_dir is None:
            return None
        sid = row["subject_id"]
        emb_path = self.embedding_cache_dir / f"{sid}.npy"
        if emb_path.exists():
            return torch.from_numpy(np.load(emb_path))
        return None

    def _augment(self, vol: torch.Tensor) -> torch.Tensor:
        """L-R flip (50%) + intensity jitter"""
        if torch.rand(1).item() > 0.5:
            vol = vol.flip(dims=[1])  # D축 (RAS의 R→L)
        scale = 0.9 + 0.2 * torch.rand(1).item()
        noise = torch.randn_like(vol) * 0.01
        return vol * scale + noise

    def __getitem__(self, idx: int) -> dict:
        row = self.records.iloc[idx]

        # MRI: embedding 캐시 우선, 없으면 NIfTI
        emb = self._load_embedding(row)
        if emb is not None:
            mri_data = emb  # [768] pre-extracted
            use_raw = False
        else:
            mri_data = self._load_mri(row)  # [1, 192, 224, 192]
            if self.augment:
                mri_data = self._augment(mri_data)
            use_raw = True

        return {
            "subject_id": row["subject_id"],
            "dataset": str(row.get("dataset_x", row.get("dataset", ""))),
            "diagnosis": str(row.get("diagnosis_x", row.get("diagnosis", "unknown"))),
            "mri": mri_data,
            "text": row["clinical_text"],
            "use_raw_mri": use_raw,
        }


def collate_fn(batch: list) -> dict:
    """가변 길이 텍스트를 포함한 배치 collate"""
    # use_raw_mri: 배치 내 모든 샘플이 동일한 경로(캐시 or NIfTI)를 써야 함
    # embedding_cache_dir 설정이 전체에 동일하게 적용되므로 일관성 보장됨.
    # 혼재가 발생하면 첫 번째 샘플 값 대신 majority vote 사용.
    use_raw_flags = [b["use_raw_mri"] for b in batch]
    use_raw = sum(use_raw_flags) > len(use_raw_flags) // 2  # majority vote
    return {
        "subject_id":  [b["subject_id"] for b in batch],
        "dataset":     [b["dataset"]    for b in batch],
        "diagnosis":   [b["diagnosis"]  for b in batch],
        "mri":         torch.stack([b["mri"] for b in batch]),
        "text":        [b["text"] for b in batch],
        "use_raw_mri": use_raw,
    }
