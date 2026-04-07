"""
train_brainclip.py
BrainCLIP 학습 스크립트

Usage (single GPU):
    cd /home/vlm/minyoung2
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv run python \
        experiments/BrainCLIP/scripts/train_brainclip.py \
        --config experiments/BrainCLIP/configs/exp01_baseline.toml

Usage (DDP, 2 GPU):
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    torchrun --nproc_per_node=2 \
        experiments/BrainCLIP/scripts/train_brainclip.py \
        --config experiments/BrainCLIP/configs/exp01_baseline.toml
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# 프로젝트 경로
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore

from dataset_brainclip import BrainCLIPDataset, collate_fn
from model_brainclip import BrainCLIP

# ── Param groups (differential LR) ───────────────────────────────────────────

def build_param_groups(model: nn.Module, lr: float, mri_lr: float) -> list:
    """
    MRI backbone fine-tune params: mri_lr (catastrophic forgetting 방지)
    그 외 (projection heads, BERT layers, temperature): lr

    DDP 래핑 후에도 id() 기반으로 정확히 분리됨.
    """
    raw = model.module if hasattr(model, "module") else model
    mri_backbone_ids = {
        id(p) for p in raw.mri_encoder.backbone.parameters() if p.requires_grad
    }
    mri_params   = [p for p in model.parameters() if p.requires_grad and id(p) in mri_backbone_ids]
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in mri_backbone_ids]

    groups = [{"params": other_params, "lr": lr}]
    if mri_params:
        groups.append({"params": mri_params, "lr": mri_lr})
        if is_main():
            print(f"[optimizer] param groups: other={len(other_params)} (lr={lr:.1e}), "
                  f"mri_backbone={len(mri_params)} (lr={mri_lr:.1e})")
    return groups


# ── DDP helpers ───────────────────────────────────────────────────────────────

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main() -> bool:
    return rank() == 0

def setup_ddp():
    """torchrun 환경에서 DDP 초기화"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


# ── LR Schedule ──────────────────────────────────────────────────────────────

def build_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay"""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ── Logging ───────────────────────────────────────────────────────────────────

class MetricLogger:
    """수동 로그 (wandb 없음)"""

    # [Fix #6] grad_norm, lr 컬럼 추가
    HEADER = "epoch,step,split,loss,temperature,in_batch_acc,recall_at_1_mri2text,grad_norm,lr\n"

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if is_main():
            with open(self.log_path, "w") as f:
                f.write(self.HEADER)

    def log(self, epoch: int, step: int, split: str, metrics: dict,
            gnorm: float | None = None, lr: float | None = None):
        if not is_main():
            return
        gnorm_str = f"{gnorm:.4f}" if gnorm is not None else ""
        lr_str    = f"{lr:.2e}"    if lr    is not None else ""
        row = (
            f"{epoch},{step},{split},"
            f"{metrics.get('loss', 0.0):.4f},"
            f"{metrics.get('temperature', 0.0):.4f},"
            f"{metrics.get('in_batch_acc', 0.0):.4f},"
            f"{metrics.get('recall_at_1_mri2text', 0.0):.4f},"
            f"{gnorm_str},{lr_str}\n"
        )
        with open(self.log_path, "a") as f:
            f.write(row)

    def log_epoch(self, epoch: int, split: str, avg: dict, lr: float | None = None):
        """Epoch-level 집계 기록 (step=-1 convention)"""
        self.log(epoch, -1, split, avg, gnorm=avg.get("grad_norm"), lr=lr)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_ckpt(model: nn.Module, optimizer, scheduler, epoch: int, step: int,
              metrics: dict, ckpt_dir: Path, tag: str = "last"):
    if not is_main():
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    state = {
        "epoch": epoch,
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
    }
    path = ckpt_dir / f"brainclip_{tag}.pt"
    torch.save(state, path)
    print(f"  [ckpt] saved → {path}")


def load_ckpt(model: nn.Module, optimizer, scheduler, path: Path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"  [ckpt] resumed from {path} (epoch={state['epoch']}, step={state['step']})")
    return state["epoch"], state["step"]


# ── Train / Val loop ──────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    cfg: dict,
    epoch: int,
    logger: MetricLogger,
    split: str = "train",
):
    training = split == "train"
    model.train(training)
    total_loss  = 0.0
    total_acc   = 0.0
    total_r1    = 0.0
    total_gnorm = 0.0
    n_batches   = 0
    # val step 번호는 train loader 길이 기준 offset 없이 epoch×1000 으로 구분
    step_offset = epoch * len(loader)
    last_metrics: dict = {}

    for i, batch in enumerate(loader):
        # [Fix #1] zero_grad는 forward 이전에
        if training:
            optimizer.zero_grad(set_to_none=True)

        mri   = batch["mri"].to(device, non_blocking=True).to(torch.bfloat16)
        texts = batch["text"]

        # [Fix #3] bf16 환경: GradScaler 제거, autocast만 사용
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            loss, metrics = model(mri, texts)

        if training:
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.get("clip_grad_norm", 1.0),
            ).item()
            optimizer.step()
            scheduler.step()
            total_gnorm += gnorm
        else:
            gnorm = 0.0

        total_loss += metrics["loss"]
        total_acc  += metrics["in_batch_acc"]
        total_r1   += metrics["recall_at_1_mri2text"]
        n_batches  += 1
        last_metrics = metrics

        if is_main() and i % cfg.get("log_every", 20) == 0:
            lr_now = scheduler.get_last_lr()[0] if training else 0.0
            print(
                f"  [{split}] ep{epoch} step{step_offset+i:05d}  "
                f"loss={metrics['loss']:.4f}  acc={metrics['in_batch_acc']:.3f}  "
                f"R@1={metrics['recall_at_1_mri2text']:.3f}  "
                f"τ={metrics['temperature']:.4f}  "
                + (f"gnorm={gnorm:.3f}  " if training else "")
                + f"lr={lr_now:.2e}"
            )
            logger.log(epoch, step_offset + i, split, metrics,
                       gnorm=gnorm if training else None,
                       lr=lr_now if training else None)

    # [Fix #2] last_metrics 안전하게 참조 (빈 loader 방어)
    avg = {
        "loss":                  total_loss  / max(n_batches, 1),
        "in_batch_acc":          total_acc   / max(n_batches, 1),
        "recall_at_1_mri2text":  total_r1    / max(n_batches, 1),
        "temperature":           last_metrics.get("temperature", 0.0),
        "grad_norm":             total_gnorm / max(n_batches, 1) if training else 0.0,
    }
    return avg


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None,
                        help="체크포인트 경로로 재시작")
    args = parser.parse_args()

    # DDP 초기화 (single GPU면 환경변수 없으므로 그냥 진행)
    use_ddp = "LOCAL_RANK" in os.environ
    local_rank = 0
    if use_ddp:
        local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Config 로드
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    if is_main():
        print(f"[train] config={args.config}")
        print(f"        device={device}  world_size={world_size()}")

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    texts_csv = Path(data_cfg["texts_csv"])
    emb_dir   = Path(data_cfg["embedding_cache_dir"]) if data_cfg.get("embedding_cache_dir") else None

    train_ds = BrainCLIPDataset(
        texts_csv=texts_csv,
        split="train",
        embedding_cache_dir=emb_dir,
        augment=True,
    )
    val_ds = BrainCLIPDataset(
        texts_csv=texts_csv,
        split="val",
        embedding_cache_dir=emb_dir,
        augment=False,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    train_loader  = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=True,
    )

    # ── 모델 ──────────────────────────────────────────────────────────────────
    model = BrainCLIP(
        mri_freeze_backbone=model_cfg.get("mri_freeze_backbone", True),
        mri_fine_tune_layers=model_cfg.get("mri_fine_tune_layers", 0),
        text_fine_tune_layers=model_cfg.get("text_fine_tune_layers", 2),
        proj_dim=model_cfg.get("proj_dim", 128),
        temperature=model_cfg.get("temperature", 0.07),
        learnable_temperature=model_cfg.get("learnable_temperature", True),
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if is_main():
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[model] trainable={n_train:,}  total={n_total:,}")

    mri_lr = train_cfg.get("mri_lr", train_cfg["lr"])  # 미설정이면 lr과 동일
    param_groups = build_param_groups(model, lr=train_cfg["lr"], mri_lr=mri_lr)
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.98),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    total_steps  = train_cfg["epochs"] * len(train_loader)
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_lr_lambda(warmup_steps, total_steps)
    )
    # bf16은 GradScaler 불필요 (dynamic range 충분). B200 환경에서 scaler 제거.

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and args.resume.exists():
        start_epoch, _ = load_ckpt(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1

    # ── 로거 / 체크포인트 경로 ─────────────────────────────────────────────────
    exp_name  = cfg.get("exp_name", "exp01")
    ckpt_dir  = Path(train_cfg.get("ckpt_dir", f"experiments/BrainCLIP/results/{exp_name}/checkpoints"))
    log_path  = Path(train_cfg.get("log_dir",  f"experiments/BrainCLIP/results/{exp_name}/logs")) / "train_log.csv"
    logger    = MetricLogger(log_path)

    best_val_r1   = -1.0   # best 기준: val R@1 (val_loss는 plateau로 구분 불가)
    best_val_loss = float("inf")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, train_cfg["epochs"]):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_metrics = run_epoch(
            model, train_loader, optimizer, scheduler,
            device, train_cfg, epoch, logger, split="train",
        )

        val_metrics = run_epoch(
            model, val_loader, optimizer, scheduler,
            device, train_cfg, epoch, logger, split="val",
        )

        if is_main():
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            # [Fix #6] Epoch-level summary를 CSV에도 기록
            logger.log_epoch(epoch, "train", train_metrics, lr=lr_now)
            logger.log_epoch(epoch, "val",   val_metrics)

            # [Fix #8] lr, gnorm을 epoch summary에 포함
            print(
                f"[epoch {epoch:03d}]  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_R@1={val_metrics['recall_at_1_mri2text']:.4f}  "
                f"τ={val_metrics['temperature']:.4f}  "
                f"gnorm={train_metrics['grad_norm']:.3f}  "
                f"lr={lr_now:.2e}  "
                f"({elapsed:.0f}s)"
            )

            save_ckpt(model, optimizer, scheduler, epoch, -1, val_metrics, ckpt_dir, tag="last")

            # best 기준: val R@1 우선 (val_loss는 batch 크기에 따라 plateau로 구분 불가)
            # R@1 동점 시 val_loss로 tie-break
            cur_r1   = val_metrics["recall_at_1_mri2text"]
            cur_loss = val_metrics["loss"]
            is_best  = (cur_r1 > best_val_r1) or (cur_r1 == best_val_r1 and cur_loss < best_val_loss)
            if is_best:
                best_val_r1   = cur_r1
                best_val_loss = cur_loss
                save_ckpt(model, optimizer, scheduler, epoch, -1, val_metrics, ckpt_dir, tag="best")
                print(
                    f"  → new best  val_R@1={best_val_r1:.4f}  "
                    f"val_loss={best_val_loss:.4f}"
                )

    cleanup_ddp()
    if is_main():
        print("\n학습 완료.")
        print(f"  Best val_R@1={best_val_r1:.4f}  val_loss={best_val_loss:.4f}")
        print(f"  체크포인트: {ckpt_dir}")


if __name__ == "__main__":
    main()
