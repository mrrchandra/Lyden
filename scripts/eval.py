#!/usr/bin/env python
"""Evaluation CLI (M10): val ppl, long-context probe, sample generation.

Usage:
  /home/madrin/venvs/lyden/bin/python scripts/eval.py \
      --config configs/nano.yaml --ckpt runs/nano/ckpts/ckpt_0005000.pt \
      --tasks ppl,longctx,samples --val-batches 40 --max-samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from lyden.config import load_config  # noqa: E402
from lyden.data.tokenizer import LydenTokenizer  # noqa: E402
from lyden.eval import long_context_ppl, perplexity, sample_text  # noqa: E402
from lyden.model.model import LydenModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tasks", default="ppl")
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--lengths", default="512,1024,2048")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--prompt", default="Once upon a time, there was a little dragon named")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    model = LydenModel(cfg.model).to(device).eval()
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    tok = LydenTokenizer.load(cfg.data.tokenizer_dir)
    print(f"[load] {args.ckpt} (step {payload.get('step', '?')})")

    tasks = args.tasks.split(",")
    report: dict[str, object] = {}
    if "ppl" in tasks:
        res = perplexity(model, cfg, n_batches=args.val_batches, device=device)
        report["ppl"] = res
        print(
            f"[ppl] loss {res['loss']:.4f} | ppl {res['ppl']:.2f}"
            f" | tokens {res['loss_tokens']:.0f}"
        )
    if "longctx" in tasks:
        lengths = tuple(int(v) for v in args.lengths.split(",") if v)
        lc: dict[str, dict[str, float]] = long_context_ppl(
            model, cfg, lengths=lengths, n_batches=2, device=device
        )
        report["longctx"] = lc
        for sl, r in lc.items():
            print(f"[longctx {sl:>5}] loss {r['loss']:.4f} | ppl {r['ppl']:.2f}")
    if "samples" in tasks:
        samples = []
        for i in range(args.max_samples):
            text = sample_text(
                model,
                tok,
                args.prompt + f" {i + 1}",
                n_new=160,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=cfg.seed + i,
            )
            samples.append(text)
            print(f"[sample {i}]\n{text}\n")
        report["samples"] = samples

    out_dir = Path(cfg.training.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[saved] {out_dir / 'eval_report.json'}")


if __name__ == "__main__":
    main()
