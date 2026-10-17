#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import sys, torch
sys.path.insert(0, "src")
import lyden.model.hybrid_attention as lib

attn = lib.HybridAttention(128, n_heads=4, n_kv_heads=1, window=16, n_summary=4, topk=8).cuda()
x = torch.randn(2, 32, 128, device="cuda")
k, v = attn._split_kv(attn.kv_proj(x))
k_kv = k.transpose(1, 2)
v_kv = v.transpose(1, 2)

print("A kv_proj:", attn.kv_proj.weight.shape)          # (64, 128)
print("B summary_queries:", attn.summary_queries.shape)  # (4, 1, 32)
kq = attn.summary_queries

b, kv_, l, dh = k_kv.shape
print("C shapes:", b, kv_, l, dh)
g = attn.n_summary
anchors = lib._anchors(l, g).cuda()
s_tok = kq[None].expand(b, g, kv_, dh)
print("D s_tok:", s_tok.shape, s_tok.stride())
s_tok2 = lib.rope_at_positions(s_tok, anchors, dh, attn.rope_base)
print("E roped:", s_tok2.shape, s_tok2.stride())
s = s_tok2.transpose(1, 2)
print("F s:", s.shape, s.stride())

kk = k_kv.transpose(-1, -2)
print("G k^T:", kk.shape)
scores = s @ kk
print("H scores:", scores.shape)
EOF
