#!/usr/bin/env python
"""Training entry point.

Usage:
  /home/madrin/venvs/lyden/bin/python scripts/train.py --config configs/nano.yaml \
      [--max-steps N] [--val-batches 20]

Multi-GPU (torchrun; ranks auto-detect via env):
  torchrun --standalone --nproc-per-node=2 scripts/train.py \
      --config configs/100m-kaggle-ddp.yaml
Config note: with N ranks the per-rank batch is per-rank, so use a config
whose (batch_size x grad_accum) equals the single-GPU tokens/step divided by N
(see configs/100m-kaggle-ddp.yaml).

The config's `data.dataset` selects the stream:
  "tinystories" / "fineweb_mix" -> pre-tokenized shards under
      {data_dir}/shards/{train,val} (run scripts/prepare_data.py first)
  "synthetic"                   -> deterministic pseudo-random stream

Format (console backend): step | loss | val | lr | grad_norm | tok/s | vram
plus MoE routing stats every log_every steps and stability warnings.
Logging/validation/checkpointing happen on rank 0 only.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lyden.config import LydenConfig, load_config  # noqa: E402
from lyden.data.dataloader import PackedBatcher, ShardDataset, SyntheticDataset  # noqa: E402
from lyden.model.model import LydenModel  # noqa: E402
from lyden.train import Trainer  # noqa: E402


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_batchers(cfg: LydenConfig, rank: int = 0, world_size: int = 1):
    train_ds: ShardDataset | SyntheticDataset
    val_ds: ShardDataset | SyntheticDataset
    if cfg.data.dataset == "synthetic":
        train_ds = SyntheticDataset(cfg.model.vocab_size, seed=cfg.seed, tokens_total=5_000_000)
        val_ds = SyntheticDataset(cfg.model.vocab_size, seed=cfg.seed + 1, tokens_total=1_000_000)
    else:
        shards = Path(cfg.data.data_dir).expanduser() / "shards"
        train_ds = ShardDataset(shards, "train")
        val_ds = ShardDataset(shards, "val")
    train_batcher = PackedBatcher(
        train_ds,
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
        shuffle=True,
        rank=rank,
        world_size=world_size,
    )
    val_batcher = PackedBatcher(
        val_ds,
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
        shuffle=False,
        rank=rank,
        world_size=world_size,
    )
    return train_batcher, val_batcher


def fmt_moe(stats: dict[str, Any], layer: str) -> str:
    load = stats.get(f"moe_load@{layer}")
    ent = stats.get(f"moe_entropy@{layer}")
    if not isinstance(load, torch.Tensor) or not isinstance(ent, torch.Tensor):
        return ""
    top = float(load.max())
    return f" load_top={top:.2f} ent={float(ent):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="absolute step bound for THIS process (chunked runs resume "
        "automatically; the scheduler is anchored to cfg total_steps)",
    )
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=600)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    lead = rank == 0
    device = os.environ.get("LYDEN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    backend = os.environ.get("LYDEN_DIST_BACKEND", "nccl")
    if world > 1:
        if device == "cuda":
            if local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"
            # else: fewer GPUs than ranks (e.g. a multi-rank smoke on one
            # device) — ranks share device 0 and must use a backend that
            # allows that (gloo, via LYDEN_DIST_BACKEND).
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world)
        print(f"[dist] rank {rank}/{world} on {device} via {backend}", flush=True)
    set_all_seeds(cfg.seed)
    train_batcher, val_batcher = build_batchers(cfg, rank=rank, world_size=world)
    model = LydenModel(cfg.model).to(device)
    if world > 1:
        # Belt & braces: identical per-rank init even if initialization is not
        # perfectly deterministic across ranks.
        for tensor in model.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)
    trainer = Trainer(
        cfg,
        model,
        train_batcher,
        val_batcher=val_batcher,
        device=device,
        dist_rank=rank,
        dist_world=world,
        dist_backend=backend,
    )
    total_steps = cfg.training.total_steps
    bound = total_steps if args.max_steps is None else min(args.max_steps, total_steps)
    log_every = cfg.training.log_every
    ckpt_every = max(1, min(cfg.training.checkpoint_every, total_steps // 10))
    if lead:
        print(
            f"[train] config={args.config} steps={total_steps} warmup={cfg.training.warmup_steps} "
            f"seq={cfg.training.seq_len} bs={cfg.training.batch_size} "
            f"accum={cfg.training.grad_accum} amp={cfg.training.amp} world={world}"
        )
    # Auto-resume: continue from the latest checkpoint under the run's dir.
    ckpts_dir = Path(cfg.training.out_dir) / cfg.run_name / "ckpts"
    if ckpts_dir.exists():
        candidates = sorted(ckpts_dir.glob("ckpt_*.pt"))
        if candidates:
            latest = candidates[-1]
            trainer.load_checkpoint(latest)
            if lead:
                print(f"[resume] {latest} (global_step={trainer.global_step})")

    t_start = time.perf_counter()
    val_loss = float("nan")
    val_every = max(1, args.val_every)
    info: dict[str, Any] | None = None
    while trainer.global_step < bound:
        info = trainer.step()
        if info["step"] % log_every == 0 or info["step"] == bound:
            tok_s = info["tokens_per_s"]
            if lead:
                print(
                    f"step {info['step']:>6d} | loss {info['loss']:.4f} | val {val_loss:.4f} | "
                    f"lr {info['lr']:.2e} | gn {info['grad_norm']:.2f} | {tok_s:,.0f} tok/s | "
                    f"{info['vram_gb']:.2f} GiB{fmt_moe(info['stats'], '3')}"
                )
                for warn in info["warnings"]:
                    print(f"  [warn] {warn}")
        if info["step"] % val_every == 0 or info["step"] == total_steps:
            if lead:
                val_loss = trainer.validate(args.val_batches)
                print(f"  [val] step {info['step']}: loss {val_loss:.4f}")
        if info["step"] % ckpt_every == 0 or info["step"] == bound:
            if lead:
                path = trainer.save_checkpoint()
                print(f"  [ckpt] step {info['step']} -> {path}")
    if lead:
        dt = time.perf_counter() - t_start
        tail = f" | loss {info['loss']:.4f} | val {val_loss:.4f}" if info else ""
        print(
            f"[done] reached step {trainer.global_step} in {dt / 60:.1f} min (this process){tail}"
        )
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
