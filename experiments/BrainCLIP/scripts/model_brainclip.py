"""
model_brainclip.py
BrainCLIP 모델 정의

- MRI 인코더: BrainIAC (frozen) + projection head
- 텍스트 인코더: PubMedBERT (top-N layers fine-tune) + projection head
- 손실: NT-Xent bidirectional contrastive (SimCLR)
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# BrainIAC 경로
BRAINIAC_CKPT = Path("/home/vlm/minyoung/pretrain/brainiac/BrainIAC.ckpt")


# ── Projection Head ────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """2-layer MLP projection: in_dim → hidden → out_dim (L2-normalized)"""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── MRI Encoder (BrainIAC wrapper) ────────────────────────────────────────
class MRIEncoder(nn.Module):
    """
    BrainIAC 3D ViT encoder + projection head.

    Args:
        freeze_backbone: BrainIAC 전체 동결 여부
        fine_tune_layers: 동결 해제할 마지막 N개 블록 (freeze_backbone=True일 때 무시)
    """

    def __init__(
        self,
        ckpt_path: Path = BRAINIAC_CKPT,
        embed_dim: int = 768,
        proj_dim: int = 128,
        freeze_backbone: bool = True,
        fine_tune_layers: int = 0,
    ):
        super().__init__()

        # BrainIAC 로드 (minyoung 프로젝트의 encoder_3d 재사용)
        import sys
        sys.path.insert(0, "/home/vlm/minyoung/model-claude/src")
        from model.encoder_3d import BrainIACEncoder  # type: ignore

        # V4 볼륨: 192×224×192, patch_size=16 → 2016 패치
        # BrainIAC 사전학습은 96³(216 패치) → pos_embed trilinear interpolation 필수
        self.backbone = BrainIACEncoder(
            checkpoint_path=str(ckpt_path),
            img_size=(192, 224, 192),
            patch_size=(16, 16, 16),
            interpolate_pos_embed=True,
            gradient_checkpointing=True,  # 2016 패치 → 메모리 절약
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # 마지막 N 블록 해제 (MONAI ViT blocks: self.backbone.backbone.blocks)
            if fine_tune_layers > 0:
                blocks = list(self.backbone.backbone.blocks)[-fine_tune_layers:]
                for blk in blocks:
                    for p in blk.parameters():
                        p.requires_grad = True
                print(f"[MRIEncoder] fine-tuning last {fine_tune_layers} blocks")
        else:
            print("[MRIEncoder] full backbone fine-tune")

        self.proj = ProjectionHead(embed_dim, 256, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 192, 224, 192] 또는 [B, 768] (pre-extracted embedding)
        returns: [B, proj_dim] L2-normalized
        """
        if x.dim() == 5:  # raw MRI [B, 1, D, H, W]
            feat = self.backbone(x).global_embedding  # Encoder3DOutput → [B, 768]
        else:
            feat = x  # pre-extracted embedding [B, 768]
        return self.proj(feat)


# ── Text Encoder (PubMedBERT wrapper) ─────────────────────────────────────
class TextEncoder(nn.Module):
    """
    PubMedBERT + projection head.

    Args:
        model_name: HuggingFace 모델 이름
        fine_tune_layers: 마지막 N 레이어 fine-tune (0 = projection만)
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        proj_dim: int = 128,
        fine_tune_layers: int = 2,
        max_length: int = 128,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        embed_dim = self.bert.config.hidden_size  # 768

        # 전체 동결 후 마지막 N 레이어만 해제
        for p in self.bert.parameters():
            p.requires_grad = False

        if fine_tune_layers > 0:
            layers = self.bert.encoder.layer[-fine_tune_layers:]
            for layer in layers:
                for p in layer.parameters():
                    p.requires_grad = True
            # pooler도 해제
            for p in self.bert.pooler.parameters():
                p.requires_grad = True
            print(f"[TextEncoder] fine-tuning last {fine_tune_layers} BERT layers")

        self.proj = ProjectionHead(embed_dim, 256, proj_dim)

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        enc = self.tokenize(texts, device)
        out = self.bert(**enc)
        cls_feat = out.last_hidden_state[:, 0, :]  # [CLS] token [B, 768]
        return self.proj(cls_feat)  # [B, proj_dim]


# ── NT-Xent Loss ──────────────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    SimCLR-style bidirectional contrastive loss.
    Expects L2-normalized embeddings.

    learnable=True  → temperature는 학습 가능한 파라미터 (기본값 0.07)
    learnable=False → temperature 고정 (EXP04: τ=0.02로 고정해 병목 검증)
    """

    def __init__(self, init_temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        log_t = torch.tensor(math.log(init_temperature))
        if learnable:
            self.log_temp = nn.Parameter(log_t)
        else:
            self.register_buffer("log_temp", log_t)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.01, 0.5)

    def forward(
        self, mri_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        mri_emb:  [B, D] L2-normalized
        text_emb: [B, D] L2-normalized
        returns: scalar loss, metrics dict
        """
        B = mri_emb.size(0)
        tau = self.temperature

        # similarity matrix [2B, 2B]
        emb = torch.cat([mri_emb, text_emb], dim=0)  # [2B, D]
        sim = emb @ emb.T / tau  # [2B, 2B]

        # mask out self-similarity
        mask = torch.eye(2 * B, device=sim.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # positive pairs: (i, i+B) and (i+B, i)
        labels = torch.arange(B, device=sim.device)
        labels = torch.cat([labels + B, labels])  # [2B]

        loss = F.cross_entropy(sim, labels)

        # metrics
        with torch.no_grad():
            pred = sim.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            # Recall@1 for MRI→Text direction
            mri_sim = (mri_emb @ text_emb.T) / tau  # [B, B]
            r1 = (mri_sim.argmax(dim=1) == torch.arange(B, device=sim.device)).float().mean().item()

        return loss, {
            "loss": loss.item(),
            "temperature": tau.item(),
            "in_batch_acc": acc,
            "recall_at_1_mri2text": r1,
        }


# ── Full BrainCLIP Model ───────────────────────────────────────────────────
class BrainCLIP(nn.Module):
    def __init__(
        self,
        mri_freeze_backbone: bool = True,
        mri_fine_tune_layers: int = 0,
        text_fine_tune_layers: int = 2,
        proj_dim: int = 128,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.mri_encoder  = MRIEncoder(
            freeze_backbone=mri_freeze_backbone,
            fine_tune_layers=mri_fine_tune_layers,
            proj_dim=proj_dim,
        )
        self.text_encoder = TextEncoder(
            fine_tune_layers=text_fine_tune_layers,
            proj_dim=proj_dim,
        )
        self.loss_fn = NTXentLoss(init_temperature=temperature, learnable=learnable_temperature)

    def forward(
        self,
        mri: torch.Tensor,
        texts: list[str],
    ) -> tuple[torch.Tensor, dict]:
        mri_emb  = self.mri_encoder(mri)
        text_emb = self.text_encoder(texts)
        loss, metrics = self.loss_fn(mri_emb, text_emb)
        return loss, metrics

    @torch.no_grad()
    def encode_mri(self, mri: torch.Tensor) -> torch.Tensor:
        return self.mri_encoder(mri)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self.text_encoder(texts)
