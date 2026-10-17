"""Tests for packing, shards, window reads, doc scanning, and the batcher."""

from __future__ import annotations

from collections.abc import Iterator  # noqa: F401 - used by prepare_data later
from pathlib import Path

import numpy as np
import pytest
import torch

from lyden.data.dataloader import PackedBatcher, ShardDataset, SyntheticDataset
from lyden.data.dataset import (
    MY_BOS_ID,
    MY_EOS_ID,
    pretokenize,
    read_window,
)

#: Deterministic toy "tokenization": one id per char, framed with bos/eos.
BOS, EOS = MY_BOS_ID, MY_EOS_ID
#: Distinct-but-equal-length docs so toy streams are non-periodic (doc = 18 ids).
DOC = "story {i:04d} doggo"
DOC_ID_LEN = 2 + len(DOC.format(i=0))  # bos + body ids + eos


def toy_encode(text: str) -> list[int]:
    body = [10 + ord(c) for c in text]
    assert max(body) < 65_535
    return [BOS, *body, EOS]


def toy_encode_batch(batch: list[str]) -> list[list[int]]:
    return [toy_encode(text) for text in batch]


def test_pretokenize_and_manifest(tmp_path: Path) -> None:
    n_docs = 10
    indexes = pretokenize(
        {"train": lambda: iter(DOC.format(i=i) for i in range(n_docs))},
        toy_encode_batch,
        out_dir=tmp_path,
        tokens_per_shard=30,  # forces multiple small shards
        batch_size=4,  # exercises chunked consumption
    )
    index = indexes["train"]
    per_doc = DOC_ID_LEN
    assert index.tokens_total == n_docs * per_doc
    assert sum(index.tokens_per_shard) == index.tokens_total
    assert sum(index.docs_per_shard) == n_docs
    assert len(index.shard_paths) > 1  # several shards were written
    for path, count in zip(index.shard_paths, index.tokens_per_shard, strict=False):
        assert path.stat().st_size == count * 2  # uint16

    stream = read_window(index, 0, index.tokens_total)
    assert (stream == BOS).sum() == n_docs
    assert (stream == EOS).sum() == n_docs
    assert read_window(index, 7, 9).tolist() == stream[7:16].tolist()


def test_read_window_across_shard_boundary(tmp_path: Path) -> None:
    indexes = pretokenize(
        {"train": lambda: iter(DOC.format(i=i) for i in range(6))},
        toy_encode_batch,
        out_dir=tmp_path,
        tokens_per_shard=10,
    )
    index = indexes["train"]
    stream = read_window(index, 0, index.tokens_total)
    cut = index.tokens_per_shard[0]
    for start in range(max(0, cut - 5), cut + 2):
        piece = read_window(index, start, 4)
        assert piece.tolist() == stream[start : start + 4].tolist()


def test_read_window_bounds(tmp_path: Path) -> None:
    indexes = pretokenize(
        {"train": lambda: iter(DOC.format(i=i) for i in range(3))},
        toy_encode_batch,
        out_dir=tmp_path,
        tokens_per_shard=10_000,
    )
    index = indexes["train"]
    with pytest.raises(ValueError, match="invalid for stream"):
        read_window(index, index.tokens_total - 2, 10)
    with pytest.raises(ValueError):
        read_window(index, index.tokens_total, 1)


def _prepare(tmp_path: Path, docs: int, split: str = "train") -> ShardDataset:
    pretokenize(
        {split: lambda: iter(DOC.format(i=i) for i in range(docs))},
        toy_encode_batch,
        out_dir=tmp_path,
        tokens_per_shard=100_000,
    )
    return ShardDataset(tmp_path, split)


def test_batcher_shapes_shift_determinism(tmp_path: Path) -> None:
    ds = _prepare(tmp_path, docs=2000)
    batcher = PackedBatcher(ds, seq_len=8, batch_size=4, seed=7, shuffle=True)
    x, y = batcher.next_batch()
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])  # target = next token within each row

    batcher = PackedBatcher(ds, seq_len=8, batch_size=4, seed=7, shuffle=True)
    first = batcher.next_batch()[0]
    second = batcher.next_batch()[0]
    assert not torch.equal(first, second)  # consecutive batches differ


def test_val_batches_tile_the_stream(tmp_path: Path) -> None:
    ds = _prepare(tmp_path, docs=1000, split="val")
    batch, slen = 2, 5
    batcher = PackedBatcher(ds, seq_len=slen, batch_size=batch, seed=0, shuffle=False)
    x1, _ = batcher.next_batch()
    x2, _ = batcher.next_batch()
    # span = b*(l+1); row i of batch k starts at stream offset k*span + i*(l+1)
    direct = ds.read(0, batch * (slen + 1) * 3)
    span = batch * (slen + 1)
    assert x1[1].tolist() == direct[(slen + 1) : (slen + 1) + slen].tolist()  # row 1 of batch 0
    assert x2[0].tolist() == direct[span : span + slen].tolist()  # row 0 of batch 1
    assert x2[1].tolist() == direct[span + (slen + 1) : span + (slen + 1) + slen].tolist()


def test_val_walk_wraps(tmp_path: Path) -> None:
    # stream of 4 docs * per_doc ids; 3 exact batches of span 20, then wraps
    ds = _prepare(tmp_path, docs=4, split="val")
    batcher = PackedBatcher(ds, seq_len=4, batch_size=4, seed=0, shuffle=False)
    span = 4 * 5
    assert len(ds) >= 3 * span and len(ds) < 4 * span
    x1, _ = batcher.next_batch()
    batcher.next_batch()
    batcher.next_batch()
    x_wrapped, _ = batcher.next_batch()
    assert torch.equal(x_wrapped, x1)


def test_synthetic_deterministic_and_reserved() -> None:
    a = SyntheticDataset(vocab_size=1024, seed=3, tokens_total=50_000)
    b = SyntheticDataset(vocab_size=1024, seed=3, tokens_total=50_000)
    assert np.array_equal(a.read(100, 50), b.read(100, 50))
    window = a.read(0, 1000)
    assert window.min() >= SyntheticDataset.N_RESERVED_SPECIALS
    assert window.max() < 1024


def test_synthetic_bounds() -> None:
    ds = SyntheticDataset(vocab_size=1024, seed=1, tokens_total=1000)
    with pytest.raises(ValueError, match="exceeds"):
        ds.read(900, 200)


def test_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ShardDataset(tmp_path, "train")
