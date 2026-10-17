#!/usr/bin/env bash
# Environment check + (re)install script for Lyden inside WSL.
# Usage: wsl -e bash -lc "bash /mnt/c/Users/Acer/Desktop/Projs/Lyden/scripts/env_setup.sh"
set -euo pipefail

PROJECT=/mnt/c/Users/Acer/Desktop/Projs/Lyden
VENV=/home/madrin/venvs/lyden
UV="$HOME/.local/bin/uv"

export UV_PROJECT_ENVIRONMENT="$VENV"

if ! command -v "$UV" >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

"$UV" sync --project "$PROJECT" --extra dev

echo "--- python/torch check ---"
"$VENV/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(8, 8, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul ok:", y.shape, y.dtype)
EOF
