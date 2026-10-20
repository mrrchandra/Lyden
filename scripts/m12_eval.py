#!/usr/bin/env python
"""M12 evaluation: story ppl + long-context ppl + distance recall + CoT recall.

Extends scripts/m11_eval.py for the 32K context-warming run:

- long_context_ppl at {512, 2048, 8192, 16384, 32768} - graceful scaling, no collapse;
- 1-hop fact-fix recall on held-out LONG sessions at 8K/16K/32K distance (T=0);
- 2-hop CoT recall on held-out cot_session probes (trace then answer; soft bar);
- router health snapshot from one collect_stats val forward (M11 bar).

Usage (Kaggle notebook eval cell):
  python scripts/m12_eval.py --config configs/m12-phase-e.yaml \
      --ckpt runs/100m-m12-e/ckpts/ckpt_00012600.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch

from lyden.config import load_config
from lyden.data.dataloader import PackedBatcher, ShardDataset
from lyden.data.m12_sessions import cot_session, long_fact_session, n_filler_for_distance
from lyden.data.tokenizer import LydenTokenizer
from lyden.eval import fact_fix_recall, long_context_ppl, perplexity, sample_text
from lyden.model.model import LydenModel

LONG_SEED0 = 60_000  # held-out long-probe seeds (M11 train seeds never go here)
COT_SEED0 = 90_000  # held-out CoT probe seeds
PROBE_TOK = 24


def router_snapshot(model: LydenModel, val_cfg, device) -> dict:
    """One collect_stats forward over a val batch; per-layer load/entropy."""
    val_ds = ShardDataset(Path(val_cfg.data.data_dir).expanduser() / "shards", "val")
    batcher = PackedBatcher(
        val_ds,
        seq_len=min(2048, val_cfg.training.seq_len),
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
            out[layer] = {
                "load_top": round(float(value.max()), 4),
                "mean": round(float(value.mean()), 4),
            }
        if key.startswith("moe_entropy@"):
            layer = key.split("@")[1]
            out.setdefault(layer, {})["entropy"] = round(float(value), 4)
    return out


def long_probe_session(seed: int, filler_n: int) -> tuple[str, str]:
    """Held-out long fact session cut to (prompt, expected answer)."""
    session = long_fact_session(random.Random(seed), n_filler=filler_n)
    lines = [ln for ln in session.split("\n") if ln.strip()]
    target = lines[-2]
    prompt = "\n".join(lines[1:-2]) + "\n"
    return prompt, target


def cot_probe(seed: int) -> tuple[str, str, str]:
    """Held-out 2-hop probe: (prompt, full trace+answer target, bridge fact)."""
    session = cot_session(random.Random(seed))
    lines = [ln for ln in session.split("\n") if ln.strip()]
    answer = lines[-2]
    prompt = "\n".join(lines[1:-2]) + "\n"
    # the bridge clause, e.g. "via the boathouse", is the scored surface
    via = answer.split("via ", 1)[1]
    return prompt, answer, via


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--probe-seeds", type=int, default=20)
    parser.add_argument("--cot-probe-seeds", type=int, default=20)
    parser.add_argument("--ppl-batches", type=int, default=12)
    parser.add_argument("--lc-batches", type=int, default=1)
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

    # long-context ppl across the warming ladder (capped at this phase's window)
    lc_lengths: tuple[int, ...] = (512, 2048, 8192, 16384, 32768)
    lc_lengths = tuple(x for x in lc_lengths if x <= cfg.training.seq_len)
    lc = long_context_ppl(model, cfg, lengths=lc_lengths, n_batches=args.lc_batches, device=device)
    report["long_context"] = {
        k: {"loss": round(v["loss"], 4), "ppl": round(v["ppl"], 4)} for k, v in lc.items()
    }
    print(f"[long_context] {report['long_context']}")

    # 1-hop recall at calibration distances (held-out seeds)
    calib = [n_filler_for_distance(d, tokenizer, random.Random(10_000 + d)) for d in (8192,)]
    n8k = calib[0]
    results: dict[str, dict] = {}
    per_distance = {8192: n8k, 16384: 2 * n8k, 32768: 4 * n8k}
    for distance, n_fill in per_distance.items():
        if distance > cfg.training.seq_len:
            results[str(distance)] = {"skipped": True}
            continue
        seed_hi = LONG_SEED0 + args.probe_seeds
        probes = [long_probe_session(s, n_fill) for s in range(LONG_SEED0, seed_hi)]
        recall = fact_fix_recall(model, tokenizer, probes, device=device)
        results[str(distance)] = {
            "recall": round(recall["recall"], 4),
            "matches": recall["matches"],
            "n": recall["n"],
            "seeds": f"{LONG_SEED0}..{LONG_SEED0 + args.probe_seeds - 1}",
        }
        print(f"[recall@{distance}] {results[str(distance)]}")
    report["long_recall"] = results

    # 2-hop CoT probes (soft bar)
    if cfg.training.seq_len >= 4096:
        cot_probes = [cot_probe(s) for s in range(COT_SEED0, COT_SEED0 + args.cot_probe_seeds)]
        scored = [(p, a) for p, a, _via in cot_probes]
        recall = fact_fix_recall(model, tokenizer, scored, device=device, n_new=64)
        report["cot_recall"] = {
            "recall": round(recall["recall"], 4),
            "matches": recall["matches"],
            "n": recall["n"],
            "seeds": f"{COT_SEED0}..{COT_SEED0 + args.cot_probe_seeds - 1}",
        }
        print(f"[cot_recall] {report['cot_recall']}")
        prompt, answer, via = cot_probes[0]
        text = sample_text(model, tokenizer, prompt, n_new=80, temperature=0.0, seed=0)
        report["cot_sample"] = {
            "prompt_tail": prompt.splitlines()[-1],
            "generated": text,
            "target": answer,
        }

    if device == "cuda":
        report["router_health"] = router_snapshot(model, cfg, device)

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
