#!/usr/bin/env bash
# Cleanly uninstall the broken torch/cuda stack via uv pip, then rebuild with uv sync.
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=/home/madrin/venvs/lyden
UV="$HOME/.local/bin/uv"
PROJECT=/mnt/c/Users/Acer/Desktop/Projs/Lyden

PKGS="torch triton nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 \
nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core \
nvidia-cutlass-dsl-libs-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 \
nvidia-nvtx-cu12 torch-c-dlpack-ext"

# Uninstall best-effort (ignore failures for packages that no longer exist).
for p in $PKGS; do
  "$UV" pip uninstall --python /home/madrin/venvs/lyden/bin/python "$p" >/dev/null 2>&1 || true
done

echo "--- residual dist-info/dirs ---"
find /home/madrin/venvs/lyden/lib/python3.12/site-packages -maxdepth 1 \
  \( -iname "*nvidia*" -o -iname "torch*" -o -iname "triton*" -o -iname "functorch*" \) || true

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
