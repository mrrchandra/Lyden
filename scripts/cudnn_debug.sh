#!/usr/bin/env bash
set -euo pipefail
SITE=/home/madrin/venvs/lyden/lib/python3.12/site-packages
echo "--- nvidia/cudnn tree ---"
find "$SITE/nvidia/cudnn" -type f 2>/dev/null | head -20 || echo "cudnn dir missing entirely"
echo "--- RECORD lib entries (first 5) ---"
grep -m 5 "lib/" "$SITE"/nvidia_cudnn_cu12-9.10.2.21.dist-info/RECORD 2>/dev/null || echo "no dist-info RECORD or no lib entries"
echo "--- uv cache for cudnn ---"
ls ~/.cache/uv/wheels-v6 2>/dev/null | head
find ~/.cache/uv -iname "*cudnn*" -maxdepth 4 -type d 2>/dev/null | head -5
