#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import traceback

try:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(
        "hf://datasets/roneneldan/TinyStories/data/train-00000-of-00004-2d5a1467fff1081b.parquet"
    )
    print("row groups:", pf.num_row_groups, "| rows:", pf.metadata.num_rows)
    batch = next(pf.iter_batches(batch_size=2, use_threads=False))
    col = batch.column(batch.schema.get_field_index("text"))
    print("row0:", repr(col[0].as_py()[:80]))
    print("FSSPEC RANGE READ: PASS")
except Exception:
    traceback.print_exc()
    print("FSSPEC RANGE READ: FAIL")
EOF
