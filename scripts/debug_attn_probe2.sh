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
k_kv = k.transpose(1, 2)  # (B, KV, L, Dh)
v_kv = v.transpose(1, 2)
q = attn._proj_q("q_compressed", x)  # (B, H, L, Dh)
print("k_kv", k_kv.shape, "q", q.shape)

# probe _compute_summary_bank shapes stepwise
b, kv_, l, dh = k_kv.shape
g = attn.n_summary
anchors = lib._anchors(l, g).cuda()
print("anchors", anchors.tolist())
s_tok = attn.summary_queries[None].expand(b, g, kv_, dh)
print("s_tok", s_tok.shape)
s_tok = lib.rope_at_positions(s_tok, anchors, dh, attn.rope_base)
s = s_tok.transpose(1, 2)
print("s", s.shape)
scores = s @ k_kv.transpose(-1, -2) / (dh**0.5)
print("scores", scores.shape)
allowed = torch.arange(l).cuda() <= anchors[:, None]
print("allowed", allowed.shape, bool(allowed.sum() > 0))
scores = scores.masked_fill(~allowed[None, None], float("-inf"))
rep = torch.softmax(scores, dim=-1) @ v_kv
print("rep", rep.shape)
raw = rep.permute(0, 2, 1, 3)
print("raw", raw.shape)
k_bank = lib.rope_at_positions(raw, anchors, dh, attn.rope_base)
print("k_bank", k_bank.shape)
bank_k = lib.gqa_expand(k_bank, 4, 1)
print("bank_k gqa", bank_k.shape)
bank_k = bank_k.transpose(1, 2)
print("bank_k transposed", bank_k.shape)
try:
    out = attn._compressed(q, k_kv, v_kv)
    print("compressed ok:", out.shape)
except RuntimeError as e:
    print("compressed ERROR:", str(e)[:160])
EOF
