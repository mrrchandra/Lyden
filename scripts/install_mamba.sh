#!/usr/bin/env bash
# Install mamba-ssm + causal-conv1d prebuilt wheels matched to torch 2.8 / cu12 / cp312.
set -euo pipefail
VENV=/home/madrin/venvs/lyden
UV="$HOME/.local/bin/uv"

"$UV" pip install --python "$VENV/bin/python" \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

echo "--- import + CUDA kernel check ---"
"$VENV/bin/python" - <<'EOF'
import torch
from mamba_ssm import Mamba2
from causal_conv1d import causal_conv1d_fn

m = Mamba2(d_model=128, d_state=32, d_conv=4, expand=2).cuda()
x = torch.randn(2, 64, 128, device="cuda", requires_grad=True)
y = m(x)
y.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
print("mamba_ssm OK: forward+backward on CUDA, out shape", tuple(y.shape))

q = torch.randn(1, 64, 16, device="cuda", dtype=torch.float16)
k = torch.randn(1, 16, 4, device="cuda", dtype=torch.float16)
v = torch.randn(1, 16, 4, device="cuda", dtype=torch.float16)
w = torch.randn(16, 1, 3, 4, device="cuda", dtype=torch.float16)
o = causal_conv1d_fn(q, w, bias=None, activation=None)
print("causal_conv1d OK:", tuple(o.shape))
EOF
