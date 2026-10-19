#!/usr/bin/env python
"""Exact per-component parameter counts for any Lyden config.

Usage: uv run scripts/param_count.py [configs/250m.yaml]
Defaults to the 250m config when no path is given.

Groups (split-LR plan, spec §4): "fast" = routed experts + shared experts +
router + attention branch gates (base LR); everything else is trunk (0.1x).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyden.config import load_config  # noqa: E402
from lyden.model.model import LydenModel  # noqa: E402

FAST_MARKERS = ("routed", "shared", "router", "mixer.gate")


def group_of(name: str) -> str:
    if name.startswith("blocks."):
        parts = name.split(".")
        layer = parts[1]
        subsys = parts[2] if len(parts) > 2 else ""
        kind = "mamba" if blocks_kind.get(layer) == "m" else "attention"
        if subsys == "mixer":
            return f"blocks[{layer}].mixer ({kind})"
        if subsys == "ffn":
            return f"blocks[{layer}].ffn ({'dense ffn' if kind == 'mamba' else 'moe'})"
        return f"blocks[{layer}] norms"
    if name.startswith("emb."):
        return "embedding"
    if name.startswith("head."):
        return "lm head"
    if name.startswith("final_norm"):
        return "final norm"
    return name


blocks_kind: dict[str, str] = {}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "250m.yaml"
    cfg = load_config(path)
    model = LydenModel(cfg.model)
    global blocks_kind
    blocks_kind = {str(i): kind for i, kind in enumerate(cfg.model.layer_kinds())}

    def group_of2(subs: int) -> str:
        del subs
        return ""

    counts: Counter[str] = Counter()
    fast: Counter[str] = Counter()
    for name, p in model.named_parameters():
        g = group_of(name)
        counts[g] += p.numel()
        if any(m in name for m in FAST_MARKERS):
            fast[g] += p.numel()
    total = sum(counts.values())
    width = max((len(k) for k in counts), default=8)
    print(
        f"config: {path.stem}  (dim={cfg.model.dim}, layers={cfg.model.n_layers}, "
        f"pattern={cfg.model.layer_pattern!r}, vocab={cfg.model.vocab_size})"
    )
    for g, n in counts.most_common():
        print(f"  {g:{width}s} {n:>11,d}")
    print(f"  {'TOTAL':{width}s} {total:>11,d}")
    print(f"  fast-LR params: {sum(fast.values()):,} | trunk: {total - sum(fast.values()):,}")


if __name__ == "__main__":
    main()
