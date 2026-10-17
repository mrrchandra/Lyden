"""Streaming text corpora from HuggingFace-hub parquet shards.

Bypasses the ``datasets`` library's streaming module (broken against current
pyarrow TF data paths) and reads parquet shards directly through
``pyarrow.parquet`` + fsspec, which supports HTTP range reads per row group —
only the needed data is fetched.

Public contract deliberately hides row iteration specifics behind
``iter_row_batches`` so unit tests can drive it with local parquet files.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

import pyarrow.parquet as pq

_SPLIT_PATTERN = re.compile(r"[-/](?P<split>train|val|validation|test|dev)[-/_.]")


def list_shard_paths(repo: str, repo_type: str = "dataset") -> dict[str, list[str]]:
    """Discover parquet shards in a repo, keyed by inferred split name.

    ``repo`` must expose ``.parquet`` files with conventional names such as
    ``data/train-00000-of-00004-<sha>.parquet``. The split token is taken from
    the filename prefix (train/validation/test...). Splits get sorted URLs so
    consumption order is stable.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    names = sorted(
        f
        for f in api.list_repo_files(repo, repo_type=repo_type)
        if f.endswith(".parquet")
    )
    shards: dict[str, list[str]] = {}
    for name in names:
        m = _SPLIT_PATTERN.search(name)
        split = m.group("split") if m else "train"
        split = {"validation": "val"}.get(split, split)
        shards.setdefault(split, []).append(f"hf://{repo_type}s/{repo}/{name}")
    if not shards:
        raise FileNotFoundError(f"No .parquet shards found in hub repo {repo!r}")
    return shards


def iter_row_batches(
    paths: Sequence[str],
    column: str,
    batch_rows: int = 2048,
) -> Iterator[Any]:
    """Yield pyarrow batches of ``column`` across parquet paths in order.

    Each path may be a local file or any fsspec URL (hub ``hf://`` supported
    natively; downloads are row-group ranges, not whole files).
    """
    for path in paths:
        pf = pq.ParquetFile(path)
        if column not in pf.schema_arrow.names:
            raise KeyError(
                f"column {column!r} missing in {path}; columns={list(pf.schema_arrow.names)}"
            )
        yield from pf.iter_batches(batch_size=batch_rows, columns=[column])


def stream_texts(
    paths: Sequence[str],
    column: str = "text",
    batch_rows: int = 2048,
    strip: bool = True,
    skip_empty: bool = True,
) -> Iterator[str]:
    """Stream text documents from the given parquet paths."""

    def to_py(batch: Any) -> list[str]:
        return [batch.column(0)[i].as_py() for i in range(batch.num_rows)]

    for batch in iter_row_batches(paths, column, batch_rows):
        for value in to_py(batch):
            if value is None:
                if skip_empty:
                    continue
                text = ""
            elif not isinstance(value, str):
                text = value.decode("utf-8", errors="ignore")
            else:
                text = value
            if strip:
                text = text.strip()
            if not text and skip_empty:
                continue
            yield text
