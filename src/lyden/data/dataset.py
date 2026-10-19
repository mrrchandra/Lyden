"""Packed token-shard store for pre-tokenized datasets.

Documents are pre-tokenized (bos, text, eos) and concatenated into a single
token stream per split, written to fixed-size binary shards (uint16 by
default; the 32K vocab fits). A per-split ``manifest.json`` records the shard
layout so the dataloader can serve random training windows or sequential
validation windows with zero padding and zero per-step tokenization cost.

Document boundaries inside a shard remain recoverable because ``<|bos|>`` is a
reserved id (1); ``iter_documents`` scans for it with numpy.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np

#: Reserved ids from LydenTokenizer.SPECIAL_TOKENS ordering.
MY_BOS_ID = 1
MY_EOS_ID = 2

#: numpy dtype names accepted for shard storage.
_DTYPE_NAMES = ("uint16", "uint32")


@dataclass(frozen=True)
class ShardIndex:
    """Validated view over one split's shards."""

    shard_paths: tuple[Path, ...]
    tokens_per_shard: tuple[int, ...]
    docs_per_shard: tuple[int, ...]
    tokens_total: int
    dtype: str  # "uint16" or "uint32"

    @classmethod
    def load(cls, data_dir: str | Path, split: str) -> ShardIndex:
        """Load and validate the manifest for ``split`` under ``data_dir``."""
        base = Path(data_dir).expanduser() / split
        manifest_path = base / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest at {manifest_path}; run scripts/prepare_data.py first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = cls(
            shard_paths=tuple(base / name for name in manifest["shards"]),
            tokens_per_shard=tuple(manifest["tokens_per_shard"]),
            docs_per_shard=tuple(manifest["docs_per_shard"]),
            tokens_total=int(manifest["tokens_total"]),
            dtype=manifest.get("dtype", "uint16"),
        )
        index.validate(on_disk=True)
        return index

    def validate(self, on_disk: bool) -> None:
        """Fail loudly if the index is internally inconsistent (or broken on disk)."""
        n = len(self.shard_paths)
        if not (n == len(self.tokens_per_shard) == len(self.docs_per_shard)):
            raise ValueError(f"shard count mismatch: {n} paths vs counts")
        if sum(self.tokens_per_shard) != self.tokens_total:
            raise ValueError("tokens_per_shard does not sum to tokens_total")
        if self.dtype not in _DTYPE_NAMES:
            raise ValueError(f"dtype must be one of {_DTYPE_NAMES}, got {self.dtype}")
        if on_disk:
            itemsize = np.dtype(self.dtype).itemsize
            for path, count in zip(self.shard_paths, self.tokens_per_shard, strict=False):
                if not path.exists():
                    raise FileNotFoundError(f"missing shard file: {path}")
                if path.stat().st_size != count * itemsize:
                    raise ValueError(
                        f"shard {path.name} size {path.stat().st_size} bytes != "
                        f"{count * itemsize} expected from manifest"
                    )

    @property
    def shard_offsets(self) -> tuple[int, ...]:
        """Global token offset at which each shard starts."""
        offsets = [0]
        for count in self.tokens_per_shard:
            offsets.append(offsets[-1] + count)
        return tuple(offsets[:-1])

    def locate(self, global_pos: int) -> tuple[Path, int]:
        """Map a global stream position to ``(shard_path, local_offset)``."""
        if not 0 <= global_pos < self.tokens_total:
            raise IndexError(f"position {global_pos} outside stream of {self.tokens_total} tokens")
        for path, offset, count in zip(
            self.shard_paths, self.shard_offsets, self.tokens_per_shard, strict=False
        ):
            if offset <= global_pos < offset + count:
                return path, global_pos - offset
        raise AssertionError("unreachable: position valid but no shard matched")

    def open_shard(self, path: Path) -> np.ndarray:
        """Memory-map a shard binary as an array of its declared dtype."""
        return np.memmap(path, dtype=np.dtype(self.dtype), mode="r")


class ShardWriter:
    """Accumulates token ids and flushes fixed-size binary shards for one split."""

    def __init__(self, token_type: str = "uint16") -> None:
        if token_type not in _DTYPE_NAMES:
            raise ValueError(f"token_type must be one of {_DTYPE_NAMES}, got {token_type}")
        self.token_type = token_type
        self.max_value = int(np.iinfo(np.dtype(token_type)).max)
        self.tokens_per_shard: list[int] = []
        self.docs_per_shard: list[int] = []
        self.tokens_total = 0
        self._buffer: list[int] = []
        self._shard_idx = 0
        self._docs_in_current_shard = 0
        self._dir: Path | None = None

    def start(self, out_dir: Path) -> None:
        """Create the split directory and begin accepting documents."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self._dir = out_dir

    def add_document(self, ids: list[int]) -> None:
        """Append one pre-tokenized document (must already include bos/eos)."""
        assert self._dir is not None, "call start() before add_document()"
        if not ids:
            return
        if max(ids) > self.max_value:
            raise ValueError(
                f"token id {max(ids)} exceeds {self.token_type} maximum {self.max_value}"
            )
        self._buffer.extend(ids)
        self.tokens_total += len(ids)
        self._docs_in_current_shard += 1

    def maybe_flush(self, tokens_per_shard: int) -> None:
        """Flush if the buffer has reached ``tokens_per_shard``."""
        if len(self._buffer) >= tokens_per_shard:
            self._flush()

    def finalize(self) -> None:
        """Flush any remaining tokens; call once after the last document."""
        if self._buffer:
            self._flush()

    def manifest(self) -> dict:
        """Build the manifest dict matching :meth:`ShardIndex.load` expectations."""
        return {
            "shards": [f"shard_{i:04d}.bin" for i in range(len(self.tokens_per_shard))],
            "tokens_per_shard": list(self.tokens_per_shard),
            "docs_per_shard": list(self.docs_per_shard),
            "tokens_total": self.tokens_total,
            "dtype": self.token_type,
        }

    def _flush(self) -> None:
        assert self._dir is not None
        path = self._dir / f"shard_{self._shard_idx:04d}.bin"
        np.asarray(self._buffer, dtype=self.token_type).tofile(path)
        assert path.stat().st_size == len(self._buffer) * np.dtype(self.token_type).itemsize
        self.tokens_per_shard.append(len(self._buffer))
        self.docs_per_shard.append(self._docs_in_current_shard)
        self._shard_idx += 1
        self._docs_in_current_shard = 0
        self._buffer = []


def pretokenize(
    splits: dict[str, Callable[[], Iterator[str]]],
    encode_batch_fn: Callable[[list[str]], list[list[int]]],
    out_dir: str | Path,
    tokens_per_shard: int = 100_000_000,
    token_type: str = "uint16",
    batch_size: int = 512,
    max_tokens: int | None = None,
) -> dict[str, ShardIndex]:
    """Tokenize lazily-supplied document iterators into packed shards per split.

    Args:
        splits: mapping of split name to a callable returning document iterator
            (lazy so a failing split does not consume the others).
        encode_batch_fn: ``[text, ...] -> list[list[int]]`` batched encoder; each
            document's ids must already be framed with bos/eos. Batched
            encoding keeps the Rust tokenizer threading hot.
        out_dir: parent directory; each split gets ``out_dir/<split>/``.
        tokens_per_shard: target shard size in tokens (last shard short).
        token_type: storage dtype ("uint16" or "uint32").
        batch_size: documents per encoder batch.
        max_tokens: stop consuming the split after this many tokens (None = all).

    A batch whose ids exceed the storage dtype aborts that split with
    a warning rather than corrupting the stream.
    """
    out_dir = Path(out_dir).expanduser()
    indexes: dict[str, ShardIndex] = {}
    for split, lazy_docs in splits.items():
        split_dir = out_dir / split
        writer = ShardWriter(token_type=token_type)
        writer.start(split_dir)
        aborted = False
        doc_iter = lazy_docs()
        while True:
            if max_tokens is not None and writer.tokens_total >= max_tokens:
                break
            chunk = list(islice(doc_iter, batch_size))
            if not chunk:
                break
            try:
                batch = encode_batch_fn(chunk)
            except ValueError as exc:
                warnings.warn(f"split '{split}' aborted: {exc}", stacklevel=2)
                aborted = True
                break
            for ids in batch:
                writer.add_document(ids)
                writer.maybe_flush(tokens_per_shard)
        if aborted:
            for stale in split_dir.glob("shard_*.bin"):
                stale.unlink()
            continue
        writer.finalize()
        (split_dir / "manifest.json").write_text(
            json.dumps(writer.manifest(), indent=2), encoding="utf-8"
        )
        indexes[split] = ShardIndex.load(out_dir, split)
    return indexes


def read_window(
    index: ShardIndex,
    global_start: int,
    length: int,
) -> np.ndarray:
    """Read ``length`` tokens starting at a global stream position.

    Returns a contiguous int64 array (ready for ``torch.from_numpy``). Reads
    may cross shard boundaries; pieces are assembled in stream order.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    end = global_start + length
    if not 0 <= global_start < end <= index.tokens_total:
        raise ValueError(
            f"window [{global_start}, {end}) invalid for stream of {index.tokens_total}"
        )
    pieces: list[np.ndarray] = []
    pos = global_start
    while pos < end:
        path, local = index.locate(pos)
        shard_no = index.shard_paths.index(path)
        take = min(index.tokens_per_shard[shard_no] - local, end - pos)
        arr = index.open_shard(path)
        pieces.append(np.asarray(arr[local : local + take], dtype=np.int64))
        pos += take
    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def iter_documents(
    index: ShardIndex,
    bos_id: int = MY_BOS_ID,
) -> Iterator[np.ndarray]:
    """Yield document token arrays (bos..., eos) via bos-boundary scanning.

    Shards may split documents at flush boundaries; the first span of a shard
    (before its first bos) is the tail of a document from the previous shard
    and is yielded as a standalone fragment.
    """
    for path in index.shard_paths:
        toks = np.asarray(index.open_shard(path))
        starts = np.flatnonzero(toks == bos_id)
        if starts.size == 0:
            if toks.size:
                yield toks
            continue
        edges = [*starts.tolist(), toks.size]
        if edges[0]:
            yield toks[: edges[0]]
        for begin, end in zip(edges[:-1], edges[1:], strict=False):
            yield toks[begin:end]
