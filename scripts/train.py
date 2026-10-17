#!/usr/bin/env python
"""Training entry point.

Usage:
  /home/madrin/venvs/lyden/bin/python scripts/train.py --config configs/nano.yaml \
      [--max-steps N] [--val-batches 20]

The config's `data.dataset` selects the stream:
  "tinystories" / "fineweb_mix" -> pre-tokenized shards under
      {data_dir}/shards/{train,val} (run scripts/prepare_data.py first)
  "synthetic"                   -> deterministic pseudo-random stream

Format (console backend): step | loss | val | lr | grad_norm | tok/s | vram
plus MoE routing stats every log_every steps and stability warnings.
"""

from __future__ import annotations

import argparse
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


def build_batchers(cfg: LydenConfig):
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
    )
    val_batcher = PackedBatcher(
        val_ds,
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
        shuffle=False,
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
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.seed)
    device = "cuda"
    train_batcher, val_batcher = build_batchers(cfg)
    model = LydenModel(cfg.model).to(device)
    trainer = Trainer(
        cfg,
        model,
        train_batcher,
        val_batcher=val_batcher,
        device=device,
    )
    total_steps = args.max_steps if args.max_steps is not None else cfg.training.total_steps
    trainer.total_steps = min(cfg.training.total_steps, total_steps)
    log_every = cfg.training.log_every
    ckpt_every = max(1, min(cfg.training.checkpoint_every, total_steps // 4 or 1))
    print(
        f"[train] config={args.config} steps={total_steps} warmup={cfg.training.warmup_steps} "
        f"seq={cfg.training.seq_len} bs={cfg.training.batch_size} accum={cfg.training.grad_accum} "
        f"amp={cfg.training.amp}"
    )
    t_start = time.perf_counter()
    val_loss = float("nan")
    for _ in range(total_steps):
        info = trainer.step()
        if info["step"] % log_every == 0 or info["step"] == total_steps:
            tok_s = info["tokens_per_s"]
            print(
                f"step {info['step']:>6d} | loss {info['loss']:.4f} | val {val_loss:.4f} | "
                f"lr {info['lr']:.2e} | gn {info['grad_norm']:.2f} | {tok_s:,.0f} tok/s | "
                f"{info['vram_gb']:.2f} GiB{fmt_moe(info['stats'], '3')}"
            )
            for warn in info["warnings"]:
                print(f"  [warn] {warn}")
        if info["step"] % max(1, total_steps // 4) == 0 or info["step"] == total_steps:
            val_loss = trainer.validate(args.val_batches)
            print(f"  [val] step {info['step']}: loss {val_loss:.4f}")
        if info["step"] % ckpt_every == 0 or info["step"] == total_steps:
            path = trainer.save_checkpoint()
            print(f"  [ckpt] step {info['step']} -> {path}")
    dt = time.perf_counter() - t_start
    print(
        f"[done] {int(total_steps)} steps in {dt/60:.1f} min | final loss {info['loss']:.4f} "
        f"| val {val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
