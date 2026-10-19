"""Batcher over packed shards or a synthetic token stream.

All batches are padding-free: fixed-length windows over a continuous packed
token stream, with inputs/targets shifted by one for next-token prediction.
Two backends:

* :class:`ShardDataset` reads pre-tokenized binary shards (real data).
* :class:`SyntheticDataset` materializes a deterministic pseudo-random token
  stream (used for CI/smoke runs without any download).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from lyden.data.dataset import ShardIndex, read_window


class ShardDataset:
    """Windows over a pre-tokenized packed shard split."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        base = Path(data_dir).expanduser()
        if not (base / split / "manifest.json").exists():
            raise FileNotFoundError(
                f"Dataset split {split} not prepared under {base}; "
                "run scripts/prepare_data.py first"
            )
        self.index = ShardIndex.load(base, split)

    def __len__(self) -> int:
        return self.index.tokens_total

    def read(self, start: int, length: int) -> np.ndarray:
        """Return ``length`` int64 tokens starting at a global position."""
        return read_window(self.index, start, length)


class SyntheticDataset:
    """Deterministic pseudo-random token stream (no download required).

    A virtual stream of ``tokens_total`` ids in ``[n_special, vocab_size)`` is
    materialized once from the seed; reads are slices of it, so identical
    (seed, vocab, length) triples reproduce identical data forever.
    """

    N_RESERVED_SPECIALS = 7

    def __init__(
        self,
        vocab_size: int,
        seed: int = 1337,
        tokens_total: int = 10_000_000,
    ) -> None:
        if vocab_size <= self.N_RESERVED_SPECIALS:
            raise ValueError(f"vocab_size must exceed {self.N_RESERVED_SPECIALS} reserved specials")
        rng = np.random.default_rng(seed)
        self._stream = rng.integers(
            self.N_RESERVED_SPECIALS, vocab_size, size=tokens_total, dtype=np.int64
        )

    def __len__(self) -> int:
        return self._stream.size

    def read(self, start: int, length: int) -> np.ndarray:
        if start + length > self._stream.size:
            raise ValueError(f"read [{start}:{start + length}) exceeds virtual stream")
        return self._stream[start : start + length]


class PackedBatcher:
    """Serves padding-free (input, target) batches of shape (B, L).

    ``shuffle=True`` (training) samples random stream starts per batch with the
    seeded RNG; ``shuffle=False`` (validation) walks the stream sequentially in
    fixed strides, deterministically wrapping near the end.

    Distributed use: with ``world_size > 1``, each rank's train sampler gets a
    rank-shifted RNG (statistically independent windows across ranks) and the
    validation walk is strided so ranks cover disjoint spans.
    """

    def __init__(
        self,
        dataset: ShardDataset | SyntheticDataset,
        seq_len: int = 512,
        batch_size: int = 32,
        seed: int = 1337,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.dataset = dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rank = int(rank)
        self._world_size = int(world_size)
        # Prime spread decorrelates per-rank samplers while keeping world_size=1
        # baseline identical to the historical rng (seed + 0).
        self._rng = np.random.default_rng(seed + 8191 * self._rank)
        self._sequential_offset = self._rank * batch_size * (seq_len + 1)

    def state_dict(self) -> dict[str, object]:
        """Stream position + RNG state (checkpoint-resume support)."""
        bg = self._rng.bit_generator
        return {
            "bg_state": dict(bg.state),
            "sequential_offset": self._sequential_offset,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._rng.bit_generator.state = dict(state["bg_state"])
        self._sequential_offset = int(state["sequential_offset"])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input, target)`` int64 LongTensors of shape (B, L).

        A batch reads ``B * (L + 1)`` consecutive tokens, reshaped row-major to
        (B, L+1); the target is the input shifted by one position within each
        row (standard next-token packing, no padding anywhere).
        """
        span = self.batch_size * (self.seq_len + 1)
        if self.shuffle:
            start = int(self._rng.integers(0, len(self.dataset) - span))
        else:
            if self._sequential_offset + span > len(self.dataset):
                self._sequential_offset = self._rank * span
            start = self._sequential_offset
            self._sequential_offset += span * self._world_size
        chunk = torch.from_numpy(self.dataset.read(start, span)).view(
            self.batch_size, self.seq_len + 1
        )
        x = chunk[:, :-1].contiguous()
        y = chunk[:, 1:].contiguous()
        return x, y
