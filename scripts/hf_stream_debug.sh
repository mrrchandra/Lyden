#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import traceback
try:
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=False)
    print("streaming object ok")
    it = iter(ds)
    for i in range(3):
        s = next(it)
        print(i, type(s), list(s.keys())[:5] if isinstance(s, dict) else s)
        print(repr(s.get("story", ""))[:120])
except Exception:
    traceback.print_exc()
EOF
