#!/usr/bin/env python
"""M12 context-warming phase configs: 1024 -> 32K in five windows.

Emits one YAML per phase (A..E) from the M11 base config into ``configs/``.
``rope_base`` stays 10000.0 - the warming phases themselves teach the model
the new position range. Per-phase (batch, seq, accum) keeps per-rank
optimizer tokens within a 16K..524K budget so T4 16 GiB VRAM with
activation checkpointing holds and optimizer dynamics stay sane.

    python scripts/m12_warmup_configs.py --entry-step 9600
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from lyden.config import load_config

BASE_CONFIG = Path("configs/m11-100m-continue-ddp.yaml")
# Fits one T4x2 session: ~9.4 h train at ~5K tok/s/rank + ~1.5 h wheels/prep,
# under the 12 h Kaggle cap. Edge saves at `bound` make sessions chainable.
WINDOWS_AND_STEPS = [(2048, 750), (4096, 450), (8192, 220), (16384, 70), (32768, 130)]
PHASE_NAMES = ["a", "b", "c", "d", "e"]
CKPT_EVERY = 200


def phase_edges(entry_step: int = 9600) -> list[int]:
    """Absolute global step each phase ends on (ckpt names continue from there)."""
    edges: list[int] = []
    step = entry_step
    for _, total_steps in WINDOWS_AND_STEPS:
        step += total_steps
        edges.append(step)
    return edges


def build_phase_configs(entry_step: int = 9600) -> list[dict]:
    """The five phase dicts derived from the M11 base (plain nested dicts)."""
    base = load_config(BASE_CONFIG)
    raw = dataclasses.asdict(base)
    configs: list[dict] = []
    for name, (window, total_steps) in zip(PHASE_NAMES, WINDOWS_AND_STEPS, strict=False):
        cfg = _phase_cfg(raw, name, window, total_steps)
        configs.append(cfg)
    return configs


def write_phase_configs(out_dir: Path = Path("configs")) -> list[Path]:
    paths: list[Path] = []
    for name, cfg in zip(PHASE_NAMES, build_phase_configs(), strict=False):
        path = out_dir / f"m12-phase-{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def _phase_cfg(raw: dict, name: str, window: int, total_steps: int) -> dict:
    import copy

    cfg = copy.deepcopy(raw)
    cfg["run_name"] = f"100m-m12-{name}"
    cfg["training"]["seq_len"] = window
    cfg["training"]["total_steps"] = total_steps
    cfg["training"]["checkpoint_every"] = CKPT_EVERY
    cfg["training"]["reset_scheduler"] = True
    cfg["training"]["reset_moe_bias"] = True
    batch, accum = _data_geometry(window)
    cfg["training"]["batch_size"] = batch
    cfg["training"]["grad_accum"] = accum
    cfg["data"]["data_dir"] = "/kaggle/working/lyden-data-m12"
    cfg["data"]["tokenizer_dir"] = "/kaggle/working/lyden-data/tokenizer"
    return cfg


def _data_geometry(window: int) -> tuple[int, int]:
    """(micro_batch, grad_accum) so per-rank tokens/step stay in budget."""
    if window <= 4096:
        return 2, 8
    return 1, 16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-step", type=int, default=9600)
    parser.add_argument("--out-dir", default="configs")
    args = parser.parse_args()
    for path in write_phase_configs(Path(args.out_dir)):
        print(f"wrote {path}")
    print("edges:", phase_edges(args.entry_step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
