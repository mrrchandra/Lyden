#!/usr/bin/env python
"""M12 data prep for the 32K context-warming run.

Mixture per split (mirrors scripts/m11_prepare_sessions.py wiring):

- TinyStories streamed from the HF hub (plain docs, zero padding).
- Every ``--stories-per-special`` plain stories, one SPECIAL doc from a
  rotating roster: [2-hop CoT trace, long fact session @8K, @16K, @32K,
  short fact session]. Fillers are calibrated against the real tokenizer
  so the fact -> question token distance matches the target (+/-10%).
- Split seeds start far apart (0 vs 20_000) exactly like M11 so train/val
  facts are disjoint.

Usage (Kaggle/WSL):
  python scripts/m12_prepare_data.py --tokenizer-dir /kaggle/working/lyden-data/tokenizer \
      --data-dir /kaggle/working/lyden-data-m12 \
      --max-train-tokens 260_000_000 --max-val-tokens 2_000_000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

sys.path.insert(0, "src")

from lyden.data.clm_sessions import synthetic_fact_session
from lyden.data.dataset import ShardIndex, pretokenize
from lyden.data.hub import list_shard_paths, stream_texts
from lyden.data.m12_sessions import cot_session, long_fact_session, n_filler_for_distance
from lyden.data.tokenizer import LydenTokenizer

CORPUS = "roneneldan/TinyStories"
TRAIN_SEED_ROW = 0
VAL_SEED_ROW = 20_000
DISTANCES = (8192, 16384, 32768)


def resolve_paths(split: str) -> list[str]:
    shards = list_shard_paths(CORPUS)
    if split not in shards:
        raise FileNotFoundError(f"split '{split}' missing in {CORPUS}: {sorted(shards)}")
    return shards[split]


def framed(doc: str) -> str:
    return f"<|bos|>{doc}<|eos|>"


def m12_document_stream(
    rows: list[str],
    *,
    tokenizer: LydenTokenizer,
    seed_row: int,
    stories_per_special: int,
    distances: tuple[int, ...] = DISTANCES,
) -> tuple[Iterator[str], Callable[[], dict[str, object]]]:
    """Yield the mixed story/special document stream until stories dry up.

    After each ``stories_per_special`` stories the roster emits, in order:
    one 2-hop CoT trace, one long fact session per distance, one short
    M11-style fact session - then the block repeats.

    Returns (stream, stats_fn); stats_fn exposes emitted counters.
    """
    rnd = random.Random(seed_row)
    fillers = {
        d: n_filler_for_distance(d, tokenizer, random.Random(10_000 + d)) for d in distances
    }
    specials: list[Callable[[], str]] = [
        lambda: cot_session(rnd),
        *(
            lambda d=d, n=fillers[d]: long_fact_session(rnd, n_filler=n) for d in distances
        ),
        lambda: synthetic_fact_session(rnd),
    ]
    stories = stream_texts(rows)
    stats: dict[str, int] = {"stories": 0, "specials": 0}

    def gen() -> Iterator[str]:
        while True:
            for _ in range(stories_per_special):
                story = next(stories, None)
                if story is None:
                    return
                stats["stories"] += 1
                yield story
            stats["specials"] += 1
            yield specials[(stats["specials"] - 1) % len(specials)]()

    filler_counts = {int(d): int(n) for d, n in fillers.items()}

    def stats_fn() -> dict[str, object]:
        return {**stats, "filler_counts": filler_counts}

    return gen(), stats_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="~/lyden-data-m12")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--max-train-tokens", type=int, default=260_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=2_000_000)
    parser.add_argument("--stories-per-special", type=int, default=100)
    parser.add_argument("--tokens-per-shard", type=int, default=10_000_000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    tokenizer = LydenTokenizer.load(Path(args.tokenizer_dir).expanduser())
    print(f"[tokenizer] vocab={tokenizer.vocab_size}")

    def run(split: str, seed_row: int, max_tokens: int) -> ShardIndex:
        rows = resolve_paths("train" if split == "train" else "val")
        lazy = m12_document_stream(
            rows,
            tokenizer=tokenizer,
            seed_row=seed_row,
            stories_per_special=args.stories_per_special,
        )
        stream, stats_fn = lazy
        t0 = time.time()
        index = pretokenize(
            {split: lambda: stream},
            lambda docs: tokenizer.encode_batch([framed(d) for d in docs]),
            out_dir=data_dir / "shards",
            tokens_per_shard=args.tokens_per_shard,
            max_tokens=max_tokens,
        )[split]
        stats = stats_fn()
        print(
            f"[{split}] {index.tokens_total:,} tokens, {sum(index.docs_per_shard):,} docs "
            f"({stats['specials']} specials) in {time.time() - t0:.0f}s"
        )
        return index

    train_index = run("train", TRAIN_SEED_ROW, args.max_train_tokens)
    val_index = run("val", VAL_SEED_ROW, args.max_val_tokens)
    summary = {
        "dataset": "tinystories+m12-sessions",
        "data_dir": str(data_dir),
        "tokenizer_dir": str(args.tokenizer_dir),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_index.tokens_total,
        "val_tokens": val_index.tokens_total,
        "distances": list(DISTANCES),
        "shards_dir": str(data_dir / "shards"),
    }
    (data_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
