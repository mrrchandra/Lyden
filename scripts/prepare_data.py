"""Prepare datasets: train tokenizer, pretokenize packed shards.

Usage (inside WSL):
  uv run --project . python scripts/prepare_data.py \
      --dataset tinystories --data-dir ~/lyden-data \
      --tokenizer-sample 30000 --max-train-tokens 200_000_000

Corpora stream from the HuggingFace hub as parquet shards (direct pyarrow
reads; the ``datasets`` library is NOT used because its streaming layer is
broken against current pyarrow). The tokenizer trains on the first
``--tokenizer-sample`` train documents; the train split is consumed up to
``--max-train-tokens``; the validation split (capped by ``--max-val-tokens``
when given) becomes the val shards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, "src")

from lyden.data.dataset import ShardIndex, pretokenize
from lyden.data.hub import list_shard_paths, stream_texts
from lyden.data.tokenizer import LydenTokenizer

#: dataset -> hub repo id
CORPORA: dict[str, str] = {
    "tinystories": "roneneldan/TinyStories",
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
}


def resolve_corpus_paths(dataset: str, split: str) -> list[str]:
    """Hub parquet shard paths for one split of one known corpus."""
    repo = CORPORA[dataset]
    shards = list_shard_paths(repo)
    if split not in shards:
        raise FileNotFoundError(
            f"split '{split}' not present in {repo}; discovered: {sorted(shards)}"
        )
    return shards[split]


def framed(doc: str) -> str:
    """Frame a document with explicit special markers before encoding."""
    return f"<|bos|>{doc}<|eos|>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(CORPORA))
    parser.add_argument("--data-dir", default="~/lyden-data")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--tokenizer-sample", type=int, default=30_000)
    parser.add_argument("--max-train-tokens", type=int, default=200_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=None)
    parser.add_argument("--tokens-per-shard", type=int, default=100_000_000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    tokenizer_dir = Path(
        args.tokenizer_dir if args.tokenizer_dir else data_dir / "tokenizer"
    )

    # ---------------------------------------------------------------- tokenizer
    if (tokenizer_dir / "tokenizer.json").exists():
        print(f"[tokenizer] reusing existing {tokenizer_dir}")
        tokenizer = LydenTokenizer.load(tokenizer_dir)
    else:
        print(f"[tokenizer] streaming {args.tokenizer_sample} docs for BPE training...")
        sample = list(
            islice(
                stream_texts(resolve_corpus_paths(args.dataset, "train")),
                args.tokenizer_sample,
            )
        )
        t0 = time.time()
        tokenizer = LydenTokenizer.train(
            (framed(doc) for doc in sample), args.vocab_size, tokenizer_dir
        )
        print(
            f"[tokenizer] trained vocab={tokenizer.vocab_size} "
            f"in {time.time() - t0:.1f}s -> {tokenizer_dir}"
        )

    def encode_batch(docs: list[str]) -> list[list[int]]:  # noqa: F811
        return tokenizer.encode_batch([framed(d) for d in docs])

    # ---------------------------------------------------------------- shards
    def run(split_name: str) -> ShardIndex:
        paths = resolve_corpus_paths(args.dataset, split_name)
        label = "train" if split_name == "train" else "val"
        max_tokens = args.max_train_tokens if label == "train" else args.max_val_tokens
        stream_fn = lambda: stream_texts(paths)  # noqa: E731
        t0 = time.time()
        indexes = pretokenize(
            {label: stream_fn},
            encode_batch,
            out_dir=data_dir / "shards",
            tokens_per_shard=args.tokens_per_shard,
            max_tokens=max_tokens,
        )
        index = indexes[label]
        dt = time.time() - t0
        print(
            f"[{label}] {index.tokens_total:,} tokens, {sum(index.docs_per_shard):,} docs, "
            f"{len(index.shard_paths)} shards in {dt:.0f}s "
            f"({index.tokens_total / max(dt, 1e-6) / 1e3:.0f}K tok/s)"
        )
        return index

    train_index = run("train")
    val_index = run("val")

    summary = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_index.tokens_total,
        "val_tokens": val_index.tokens_total,
        "shards_dir": str(data_dir / "shards"),
    }
    (data_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
