#!/usr/bin/env python
"""M11 evaluation: story ppl + fact-fix recall + long-context + router health.

Runs against the continue-trained checkpoint (written into eval_report.json
under ``runs/<run_name>/``). The balance block reports the already-computed
per-layer load/entropy snapshot from one collect_stats forward on the val
stream, to compare against the M11 success bar (load_top <= 0.5,
entropy >= 1.95).

Usage (Kaggle v3 notebook eval cell):
  python scripts/m11_eval.py --config configs/m11-100m-continue-ddp.yaml \
      --ckpt runs/100m-m11/ckpts/ckpt_0009600.pt \
      --eval-seeds 40 --out runs/100m-m11/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch

from lyden.config import load_config
from lyden.data.clm_sessions import heldout_sessions
from lyden.data.dataloader import PackedBatcher, ShardDataset
from lyden.data.tokenizer import LydenTokenizer
from lyden.eval import fact_fix_recall, long_context_ppl, perplexity, sample_text
from lyden.model.model import LydenModel

EVAL_SEED0 = 50_000


def router_snapshot(model: LydenModel, val_cfg, device) -> dict:
    """One collect_stats forward over a val batch; per-layer load/entropy."""
    val_ds = ShardDataset(Path(val_cfg.data.data_dir).expanduser() / "shards", "val")
    batcher = PackedBatcher(
        val_ds,
        seq_len=val_cfg.training.seq_len,
        batch_size=2,
        seed=val_cfg.seed,
        shuffle=False,
    )
    model.eval()
    x, _ = batcher.next_batch()
    with torch.no_grad():
        _, stats = model(x.to(device), collect_stats=True)
    out: dict[str, dict[str, float]] = {}
    for key, value in stats.items():
        if key.startswith("moe_load@"):
            layer = key.split("@")[1]
            out[layer] = {"load_top": round(float(value.max()), 4), "mean": round(float(value.mean()), 4)}
        if key.startswith("moe_entropy@"):
            layer = key.split("@")[1]
            out.setdefault(layer, {})["entropy"] = round(float(value), 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--eval-seeds", type=int, default=40)
    parser.add_argument("--ppl-batches", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_dir = Path(
        args.tokenizer_dir if args.tokenizer_dir else cfg.data.tokenizer_dir
    ).expanduser()
    tokenizer = LydenTokenizer.load(tokenizer_dir)

    model = LydenModel(cfg.model).to(device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    step = int(payload["step"])

    report: dict[str, object] = {"step": step}
    ppl = perplexity(model, cfg, n_batches=args.ppl_batches, device=device)
    report["story_ppl"] = {
        "loss": round(ppl["loss"], 4),
        "ppl": round(ppl["ppl"], 4),
        "loss_tokens": ppl["loss_tokens"],
    }
    print(f"[ppl] {report['story_ppl']}")

    sessions = heldout_sessions(EVAL_SEED0, args.eval_seeds)
    recall = fact_fix_recall(model, tokenizer, sessions, device=device)
    report["fact_fix_recall"] = {
        "recall": round(recall["recall"], 4),
        "matches": recall["matches"],
        "n": recall["n"],
        "seeds": f"{EVAL_SEED0}..{EVAL_SEED0 + args.eval_seeds - 1}",
    }
    print(f"[recall] {report['fact_fix_recall']}")

    if device == "cuda":
        lc = long_context_ppl(model, cfg, lengths=(512, 1024), n_batches=2, device=device)
        report["long_context"] = {
            k: {"loss": round(v["loss"], 4), "ppl": round(v["ppl"], 4)} for k, v in lc.items()
        }
        report["router_health"] = router_snapshot(model, cfg, device)

        prompt, _target = sessions[2]
        text = sample_text(
            model, tokenizer, prompt, n_new=60, temperature=0.0, seed=0
        )
        report["recall_sample"] = {"prompt_tail": prompt.splitlines()[-1], "generated": text}

    out_path = Path(
        args.out if args.out else Path(cfg.training.out_dir) / cfg.run_name / "eval_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
