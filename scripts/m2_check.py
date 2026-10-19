#!/usr/bin/env python3
"""M2 verification battery: real-data tokenizer + shards + batcher, with evidence.

Run inside WSL: uv run --project . python scripts/m2_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

DATA_DIR = Path.home() / "lyden-data"

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        PASS.append(name)
        print(f"[PASS] {name:>32}  {detail}")
    except Exception as exc:  # noqa: BLE001
        FAIL.append(name)
        print(f"[FAIL] {name:>32}  {type(exc).__name__}: {exc}")


def _tokenizer_loaded():
    from lyden.data import LydenTokenizer

    tok = LydenTokenizer.load(DATA_DIR / "tokenizer")
    assert tok.vocab_size >= 32_000, tok.vocab_size
    return f"vocab={tok.vocab_size}"


def _tokenizer_roundtrip():
    from lyden.data import LydenTokenizer

    tok = LydenTokenizer.load(DATA_DIR / "tokenizer")
    text = "Once upon a time there was a little fish named Fin. <|edit|>WRITE new fact<|/edit|>"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert tok.decode(ids) == f"<|bos|>{text}<|eos|>"
    specials = {3, 4}
    assert specials.issubset(set(ids)), "edit markers lost"
    stripped = tok.decode(ids, skip_special=True)
    plain = text.replace("<|edit|>", "").replace("<|/edit|>", "")
    assert stripped == plain, f"strip mismatch: {stripped!r}"
    return f"{len(ids)} ids, markers kept"


def _tokenizer_compression():
    from lyden.data import LydenTokenizer

    tok = LydenTokenizer.load(DATA_DIR / "tokenizer")
    english = (
        "Once upon a time, there was a little mouse. "
        "He wanted to go for a walk and see the world with his friends."
    )
    hinglish = (
        "Yaar, kal mujhe bahut maza aaya jab humne ghar pe cricket khela. "
        "Fir chai pi ke story sunayi."
    )
    benglish = "Bhalobashi bolen toh ami kal bazaar jabo, notun boi kinbo ar mishti anbo bari te."
    r1 = len(english) / len(tok.encode(english))
    r2 = len(hinglish) / len(tok.encode(hinglish))
    r3 = len(benglish) / len(tok.encode(benglish))
    assert r1 > 3.0, f"english compression too low: {r1:.2f}"
    return f"chars/token: EN={r1:.2f} HI={r2:.2f} BN={r3:.2f} (romanized, untrained mix)"


def _shards_train():
    from lyden.data import ShardDataset

    ds = ShardDataset(DATA_DIR / "shards", "train")
    assert ds.index.tokens_total > 150_000_000
    assert ds.index.dtype == "uint16"
    shards = len(ds.index.shard_paths)
    return (
        f"train {ds.index.tokens_total:,} tok / "
        f"{sum(ds.index.docs_per_shard):,} docs / {shards} shards"
    )


def _shards_val():
    from lyden.data import ShardDataset

    ds = ShardDataset(DATA_DIR / "shards", "val")
    assert ds.index.tokens_total > 1_000_000
    return f"val {ds.index.tokens_total:,} tok / {sum(ds.index.docs_per_shard):,} docs"


def _doc_integrity():
    from lyden.data.dataset import MY_BOS_ID, MY_EOS_ID, ShardIndex, iter_documents

    index = ShardIndex.load(DATA_DIR / "shards", "val")
    docs = list(
        d
        for _, d in zip(range(500), iter_documents(index), strict=False)
        if len(d) and int(d[0]) == MY_BOS_ID
    )
    assert len(docs) == 500, f"got {len(docs)} framed docs"
    assert all(d[-1] == MY_EOS_ID for d in docs)
    return "500/500 docs bos...eos framed"


def _batcher_real():
    import torch

    from lyden.data import PackedBatcher, ShardDataset

    ds = ShardDataset(DATA_DIR / "shards", "train")
    b = PackedBatcher(ds, seq_len=512, batch_size=8, seed=1337, shuffle=True)
    x, y = b.next_batch()
    assert x.shape == (8, 512) and y.shape == (8, 512)
    assert x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])
    b2 = PackedBatcher(ds, seq_len=512, batch_size=8, seed=1337, shuffle=True)
    assert torch.equal(x, b2.next_batch()[0])
    return "8x512 int64, shifted targets, reproducible"


def _decode_real_window():
    from lyden.data import LydenTokenizer, PackedBatcher, ShardDataset

    tok = LydenTokenizer.load(DATA_DIR / "tokenizer")
    ds = ShardDataset(DATA_DIR / "shards", "val")
    b = PackedBatcher(ds, seq_len=256, batch_size=1, seed=42, shuffle=False)
    x, _ = b.next_batch()
    text = tok.decode(x[0].tolist())
    assert "Once upon a time" in text or "Once" in text, text[:120]
    return f"start: {text[:60]!r}"


def _throughput_hint():
    """Cheap factual note: shard read + batch materialize cost."""
    import time

    from lyden.data import PackedBatcher, ShardDataset

    ds = ShardDataset(DATA_DIR / "shards", "train")
    b = PackedBatcher(ds, seq_len=2048, batch_size=8, seed=1, shuffle=True)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        b.next_batch()
    dt = time.perf_counter() - t0
    tok = n * 8 * 2048 / dt
    return f"{tok / 1e6:.0f}M tok/s batcher IO (excl. model)"


if __name__ == "__main__":
    print(f"data dir: {DATA_DIR}")
    print("-" * 78)
    checks = [
        ("tokenizer 32K loaded", _tokenizer_loaded),
        ("tokenizer round-trip+markers", _tokenizer_roundtrip),
        ("compression 3 languages", _tokenizer_compression),
        ("train shards valid", _shards_train),
        ("val shards valid", _shards_val),
        ("doc framing integrity", _doc_integrity),
        ("batcher real data", _batcher_real),
        ("decode real window", _decode_real_window),
        ("batcher IO throughput", _throughput_hint),
    ]
    for name, fn in checks:
        check(name, fn)
    print("-" * 78)
    print(f"TOTAL: {len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
