#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import traceback

print("--- versions ---")
import pyarrow, huggingface_hub, datasets
print("pyarrow", pyarrow.__version__, "| hub", huggingface_hub.__version__, "| datasets", datasets.__version__)

print("--- direct hub parquet read test ---")
try:
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    files = sorted(
        f for f in api.list_repo_files("roneneldan/TinyStories", repo_type="dataset")
        if f.endswith(".parquet")
    )
    print("parquet files:", len(files), "| first:", files[0])
    local = hf_hub_download("roneneldan/TinyStories", files[0], repo_type="dataset")
    pf = pq.ParquetFile(local)
    print("schema:", pf.schema_arrow.names, "| rows:", pf.metadata.num_rows)
    batch = next(pf.iter_batches(batch_size=4))
    col = batch.column(batch.schema.get_field_index("story"))
    for i, v in enumerate(col):
        print("row", i, "len", len(v.as_py() or ""), repr((v.as_py() or "")[:80]))
    print("DIRECT PARQUET READ: PASS")
except Exception:
    traceback.print_exc()
    print("DIRECT PARQUET READ: FAIL")
EOF
