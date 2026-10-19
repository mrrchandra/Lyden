#!/usr/bin/env python
"""M11 data prep: pack a 70/30-like story+CLM-session mixture into shards.

One synthetic fact-fix session every ``--ratio`` plain documents; stories
stream from the HF hub exactly like scripts/prepare_data.py. Session seeds
are partitioned so training, val-ppl, and eval-recall never share facts:

  train sessions: seeds 0..          (train split)
  val sessions   : seeds 20_000..80_000   (val split)
  eval recall    : seeds 50_000..50_0xx   (scripts/m11_eval.py — disjoint start;
                    val split uses 20_000..49_999 + heldout eval reuse 50_000+)

Usage (WSL/Kaggle):
  python scripts/m11_prepare_sessions.py --tokenizer-dir ~/lyden-data/tokenizer \
      --data-dir ~/lyden-data-m11 --max-train-tokens 70_000_000 --max-val-tokens 700_000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from lyden.data.clm_sessions import mixed_documents, synthetic_fact_session
from lyden.data.dataset import ShardIndex, pretokenize
from lyden.data.hub import list_shard_paths, stream_texts
from lyden.data.tokenizer import LydenTokenizer

CORPUS = "roneneldan/TinyStories"
TRAIN_SEED_ROW = 0
VAL_SEED_ROW = 20_000


def resolve_paths(split: str) -> list[str]:
    shards = list_shard_paths(CORPUS)
    if split not in shards:
        raise FileNotFoundError(f"split '{split}' missing in {CORPUS}: {sorted(shards)}")
    return shards[split]


def framed(doc: str) -> str:
    return f"<|bos|>{doc}<|eos|>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="~/lyden-data-m11")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--max-train-tokens", type=int, default=70_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=700_000)
    parser.add_argument(
        "--ratio", type=int, default=2, help="stories per session; 2 measures ~27% session tokens"
    )
    parser.add_argument("--tokens-per-shard", type=int, default=10_000_000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    tokenizer = LydenTokenizer.load(Path(args.tokenizer_dir).expanduser())
    print(f"[tokenizer] vocab={tokenizer.vocab_size}")

    def run(split: str, seed_row: int, max_tokens: int) -> ShardIndex:
        rows = resolve_paths("train" if split == "train" else "val")
        rnd = random.Random(seed_row)

        def sessions() -> str:
            return synthetic_fact_session(rnd)

        lazy = lambda: mixed_documents(stream_texts(rows), sessions, ratio=args.ratio)  # noqa: E731
        t0 = time.time()
        index = pretokenize(
            {split: lazy},
            lambda docs: tokenizer.encode_batch([framed(d) for d in docs]),
            out_dir=data_dir / "shards",
            tokens_per_shard=args.tokens_per_shard,
            max_tokens=max_tokens,
        )[split]
        print(
            f"[{split}] {index.tokens_total:,} tokens, {sum(index.docs_per_shard):,} docs "
            f"in {time.time() - t0:.0f}s"
        )
        return index

    train_index = run("train", TRAIN_SEED_ROW, args.max_train_tokens)
    val_index = run("val", VAL_SEED_ROW, args.max_val_tokens)
    summary = {
        "dataset": "tinystories+clm-sessions",
        "data_dir": str(data_dir),
        "tokenizer_dir": str(args.tokenizer_dir),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_index.tokens_total,
        "val_tokens": val_index.tokens_total,
        "shards_dir": str(data_dir / "shards"),
    }
    (data_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
