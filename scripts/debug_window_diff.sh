#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, torch, math
sys.path.insert(0, "src")
import lyden.model.hybrid_attention as lib

torch.manual_seed(7)
attn = lib.HybridAttention(128, n_heads=4, n_kv_heads=1, window=8, n_summary=4, topk=2).cuda()
x = torch.randn(1, 16, 128, device="cuda")
k, v = attn._split_kv(attn.kv_proj(x))
cos, sin = lib.rope_cos_sin(16, 32, 10000.0, x.device)
k = lib.apply_rope(k, cos, sin)
k_h, v_h = attn._head_kv(k, v)
q = lib.apply_rope_pqmaj(attn._proj_q("q_sliding", x), cos, sin)
got = attn._sliding(q, k_h, v_h)
scores = q @ k_h.transpose(-1, -2) / math.sqrt(32)
t = torch.arange(16, device=x.device)
band = (t[None, :] <= t[:, None]) & (t[None, :] - t[:, None] >= 0)
band8 = (t[None, :] <= t[:, None]) & (t[None, :] - t[:, None] < 8)
ref8 = lib._join_heads(torch.softmax(scores.masked_fill(~band8[None, None], float("-inf")), -1) @ v_h)
ref = lib._join_heads(torch.softmax(scores.masked_fill(~band[None, None], float("-inf")), -1) @ v_h)
per_tok = (got - ref8).abs().amax(dim=(0, 2))
print("per-token maxdiff got_vs_band8:", per_tok.tolist())
per_tok2 = (got - ref).abs().amax(dim=(0, 2))
print("per-token maxdiff got_vs_fullcausal:", per_tok2.tolist())
PYEOF
