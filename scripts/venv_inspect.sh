#!/usr/bin/env bash
set -euo pipefail
"$HOME/.local/bin/uv" pip list --python /home/madrin/venvs/lyden/bin/python 2>/dev/null |
  grep -Ei "torch|cudnn|cublas|nvidia" || true
echo "--- cudnn .so files ---"
ls /home/madrin/venvs/lyden/lib/python3.12/site-packages/nvidia/cudnn/lib/ 2>/dev/null || echo "NO cudnn/lib"
ls /home/madrin/venvs/lyden/lib/python3.12/site-packages/nvidia/ 2>/dev/null || echo "NO nvidia dir"
