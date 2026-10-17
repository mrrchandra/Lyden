#!/usr/bin/env bash
set -euo pipefail
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import torch
from mamba_ssm import Mamba2
from causal_conv1d import causal_conv1d_fn

m = Mamba2(d_model=128, d_state=32, d_conv=4, expand=2).cuda()
x = torch.randn(2, 64, 128, device="cuda", requires_grad=True)
y = m(x)
y.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
print("mamba_ssm OK: forward+backward on CUDA, out shape", tuple(y.shape))

q = torch.randn(1, 16, 64, device="cuda", dtype=torch.float16)  # (batch, dim, seqlen)
w = torch.randn(16, 3, device="cuda", dtype=torch.float16)  # (dim, width)
o = causal_conv1d_fn(q, w, bias=None, activation=None)
print("causal_conv1d OK:", tuple(o.shape))
print("cuda kernels: PASS")
EOF
