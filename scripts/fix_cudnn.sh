#!/usr/bin/env bash
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=/home/madrin/venvs/lyden
UV="$HOME/.local/bin/uv"
PROJECT=/mnt/c/Users/Acer/Desktop/Projs/Lyden
SITE=/home/madrin/venvs/lyden/lib/python3.12/site-packages

# Purge the ghost install (metadata present, payload missing).
rm -rf "$SITE"/nvidia_cudnn_cu12-*.dist-info "$SITE/nvidia/cudnn"

"$UV" sync --project "$PROJECT" --extra dev

echo "--- verify ---"
"$UV" run --project "$PROJECT" python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda")
    torch.cuda.synchronize()
    print("matmul ok:", (x @ x).shape)
EOF
