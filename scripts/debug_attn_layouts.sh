#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import sys, torch
sys.path.insert(0, "src")
from lyden.model.hybrid_attention import gqa_expand, HybridAttention

# gqa on bank tensors (B, G, KV, Dh):
bank = torch.randn(2, 16, 1, 32, device="cuda")
out = gqa_expand(bank, 4, 1)
print("bank gqa:", out.shape, "expected (2, 16, 4, 32)")

attn = HybridAttention(128, n_heads=4, n_kv_heads=1, window=16, n_summary=4, topk=8).cuda()
x = torch.randn(2, 32, 128, device="cuda")
k, v = attn._split_kv(attn.kv_proj(x))
print("k:", k.shape)

err = None
try:
    out = attn(x)
    print("forward ok:", out.shape)
except RuntimeError as e:
    err = e
    print("forward error:", str(e)[:200])
    # probe each branch with both layout variants
    for kk, vv in ((k, v),):
        for label, argq in (("q_headmajor", attn._proj_q("q_sliding", x)),):
            try:
                v = attn._sliding(argq, kk.transpose(1, 2), vv.transpose(1, 2))
                print(f"sliding headmajor keys ok {v.shape}")
            except RuntimeError as e2:
                print(f"sliding headmajor keys err: {str(e2)[:120]}")
            try:
                v = attn._compute_summary_bank(kk.transpose(1, 2), vv.transpose(1, 2))
                print(f"bank ok {v[0].shape}")
            except RuntimeError as e2:
                print(f"bank err: {str(e2)[:120]}")
EOF
