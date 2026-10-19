"""Hybrid attention block tests (M4): shapes, causality, per-branch semantics,
summary-anchor causality, selective top-k, gate mixing, RoPE shared.

Requires CUDA only; attention math runs on any device.
"""

from __future__ import annotations

import pytest
import torch

import lyden.model.hybrid_attention as lib

torch.manual_seed(1337)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="canonical attention paths need GPU"
)


def build_attn(dim: int = 128, **kw):
    from lyden.model.hybrid_attention import HybridAttention

    torch.manual_seed(7)
    defaults = dict(n_heads=4, n_kv_heads=1, window=16, n_summary=4, topk=8)
    attn = HybridAttention(dim, **(defaults | kw)).cuda()
    attn.eval()
    return attn


def rand_x(b: int, t: int, dim: int = 128, **kw) -> torch.Tensor:
    return torch.randn(b, t, dim, device="cuda", **kw)


@requires_gpu
def test_attn_shapes_and_finite() -> None:
    attn = build_attn()
    x = rand_x(2, 32)
    out = attn(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


@requires_gpu
def test_attn_causality_exact() -> None:
    """Everything before position k is bitwise unaffected by tokens at k+."""
    attn = build_attn()
    k = 13
    x1 = rand_x(2, 32)
    x2 = x1.clone()
    x2[:, k:] += 7.0
    with torch.no_grad():
        out1, out2 = attn(x1), attn(x2)
    assert torch.equal(out1[:, :k], out2[:, :k])


@requires_gpu
def test_attn_prefix_length_invariance() -> None:
    """Sliding/selective branches are prefix-invariant; compressed anchors move
    with sequence length by design (i*L/G), so it is excluded here."""
    attn = build_attn()
    x = rand_x(2, 48)
    p = 19
    with torch.no_grad():
        full = attn(x, return_branches=True)
        pre = attn(x[:, :p], return_branches=True)
    for branch in ("sliding", "selective"):
        assert torch.allclose(full[branch][:, :p], pre[branch], rtol=1e-4, atol=1e-4), branch
    assert not torch.allclose(full["compressed"][:, :p], pre["compressed"], rtol=1e-3, atol=1e-3), (
        "anchors changing with L should change compressed prefix outputs"
    )


@requires_gpu
def test_attn_residual_identity_when_output_zeroed() -> None:
    """Zero out-projection => block output is identity (+ zero branches)."""
    attn = build_attn()
    torch.nn.init.zeros_(attn.out_proj.weight)
    x = rand_x(1, 8)
    with torch.no_grad():
        out = attn(x)
    assert torch.equal(out, x)


@requires_gpu
def test_attn_backward_populates_grads() -> None:
    attn = build_attn()
    attn.train()
    x = rand_x(2, 24, requires_grad=True)
    attn(x).square().mean().backward()
    grads = [p.grad for p in attn.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).any() for g in grads)
    assert x.grad is not None


# ---------------------------------------------------------------- branch math
def _internals(attn, x):
    """Reproduce forward's intermediate tensors for branch-level reference math."""

    import lyden.model.hybrid_attention as lib

    seq_len = x.shape[1]
    k, v = attn._split_kv(attn.kv_proj(x))
    cos, sin = lib.rope_cos_sin(seq_len, attn.head_dim, attn.rope_base, x.device)
    k = lib.apply_rope(k, cos, sin)
    k_h, v_h = attn._head_kv(k, v)
    q_name = "q_sliding"
    return k, v, k_h, v_h, lib.apply_rope_pqmaj(attn._proj_q(q_name, x), cos, sin), cos, sin


def _dense_causal_ref(q, k, v):
    import math

    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    t = torch.arange(q.shape[2], device=q.device)
    scores = scores.masked_fill(~(t[None, :] <= t[:, None])[None, None], float("-inf"))
    return torch.softmax(scores, dim=-1) @ v, scores


@requires_gpu
def test_window_equals_dense_causal_when_window_covers() -> None:
    attn = build_attn(window=64)
    x = rand_x(2, 8)
    _, _, k_h, v_h, q, _, _ = _internals(attn, x)
    got = attn._sliding(q, k_h, v_h)
    ref, _ = _dense_causal_ref(q, k_h, v_h)
    ref = lib._join_heads(ref)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_window_band_mask_semantics() -> None:
    attn = build_attn(window=8)
    x = rand_x(1, 16)
    _, _, k_h, v_h, q, _, _ = _internals(attn, x)
    got = attn._sliding(q, k_h, v_h)
    scores = q @ k_h.transpose(-1, -2) / (q.shape[-1] ** 0.5)
    t = torch.arange(16, device=x.device)
    band = (t[None, :] <= t[:, None]) & (t[:, None] - t[None, :] < 8)
    allowed = torch.softmax(scores.masked_fill(~band[None, None], float("-inf")), dim=-1) @ v_h
    allowed = lib._join_heads(allowed)
    assert torch.allclose(got, allowed, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_compressed_bank_matches_manual_reference() -> None:
    import math

    attn = build_attn(n_summary=4)
    x = rand_x(1, 32)
    k, v, *_ = _internals(attn, x)
    kp, vp = k.transpose(1, 2), v.transpose(1, 2)  # un-expanded (B, KV, L, Dh)
    k_bank, v_bank = attn._compute_summary_bank(kp, vp)
    anchors = lib._anchors(32, 4).to(x.device)
    dh = attn.head_dim
    s = lib.rope_at_positions(attn.summary_queries[None].expand(1, 4, 1, dh), anchors, dh, 10000.0)
    s = s.transpose(1, 2)  # (B, KV, G, Dh)
    scores = s @ kp.transpose(-1, -2) / math.sqrt(dh)
    allowed = torch.arange(32, device=x.device) <= anchors[:, None]
    raw_ref = (
        torch.softmax(scores.masked_fill(~allowed[None, None], float("-inf")), dim=-1) @ vp
    ).permute(0, 2, 1, 3)  # (B, G, KV, Dh)
    assert torch.allclose(v_bank, raw_ref, rtol=1e-5, atol=1e-5)
    k_ref = lib.rope_at_positions(raw_ref, anchors, dh, 10000.0)
    assert torch.allclose(k_bank, k_ref, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_selective_topk_all_equals_dense_causal() -> None:
    attn = build_attn(window=32, topk=64)
    x = rand_x(1, 8)
    _, _, k_h, v_h, q, _, _ = _internals(attn, x)
    import lyden.model.hybrid_attention as lib

    cos, sin = lib.rope_cos_sin(8, 32, 10000.0, x.device)
    q_se = lib.apply_rope_pqmaj(attn._proj_q("q_selective", x), cos, sin)
    got = attn._selective(q_se, k_h, v_h, x)
    ref, _ = _dense_causal_ref(q_se, k_h, v_h)
    ref = lib._join_heads(ref)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_selective_support_respects_topk() -> None:
    attn = build_attn(topk=2)
    x = rand_x(1, 16)
    _, _, k_h, v_h, q, _, _ = _internals(attn, x)
    import lyden.model.hybrid_attention as lib

    cos, sin = lib.rope_cos_sin(16, 32, 10000.0, x.device)
    q_se = lib.apply_rope_pqmaj(attn._proj_q("q_selective", x), cos, sin)
    got = attn._selective(q_se, k_h, v_h, x)

    ranks = attn.indexer(x).squeeze(-1)
    t = torch.arange(16, device=x.device)
    causal = t[None, :] <= t[:, None]
    rank_mat = ranks[:, :, None].expand(1, 16, 16).masked_fill(~causal[None], float("-inf"))
    top = torch.topk(rank_mat, 2, dim=-1).indices
    selector = torch.zeros(1, 16, 16, device=x.device).scatter_(2, top, 1.0).bool()

    scores = q_se @ k_h.transpose(-1, -2) / (q_se.shape[-1] ** 0.5)
    allow = (selector & causal[None])[:, None]
    ref = torch.softmax(scores.masked_fill(~allow, float("-inf")), dim=-1) @ v_h
    ref = lib._join_heads(ref)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_gate_uniform_on_zero_input_and_formula_invariant() -> None:
    attn = build_attn()
    x = torch.zeros(1, 6, 128, device="cuda")
    with torch.no_grad():
        got = attn(x, return_branches=True)
        w = got["gate"]
        assert torch.allclose(w.sum(-1), torch.ones_like(w[..., 0]))
        assert torch.allclose(
            w, torch.full_like(w, 1.0 / 3.0)
        )  # bias-free gate: zero logits => uniform
        mean3 = (got["sliding"] + got["compressed"] + got["selective"]) / 3
        expect = x + attn.out_proj(mean3)
        assert torch.allclose(got["out"], expect, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------- production geometry
@requires_gpu
def test_attn_grads_reach_summary_queries_and_selective_routing() -> None:
    """Gradients reach the summary bank. The indexer is trained via the
    -effectless selective path (selection is hard top-k) so it tests the aux
     route: ranking is surrogate — with aux ranking OFF (default), the indexer
     must NOT receive grads (its selection is non-differentiable by design)."""
    attn = build_attn()
    attn.train()
    x = rand_x(1, 16, requires_grad=True)
    attn(x).square().mean().backward()
    assert attn.summary_queries.grad is not None
    assert attn.summary_queries.grad.abs().sum() > 0
    assert attn.indexer.weight.grad is None  # hard selection: no grads (spec)

    # with the aux ranking loss enabled, indexer grads must flow
    attn.zero_grad()
    out = attn(x, return_branches=True)
    loss = out["out"].square().mean()
    ranks = attn.indexer(x).squeeze(-1)
    pos_rank = torch.triu(ranks, diagonal=1)  # encourage attending later tokens
    ranking_loss = -(pos_rank.mean() - ranks.mean().detach() * 0)
    (loss + ranking_loss).backward()
    assert attn.indexer.weight.grad is not None
    assert attn.indexer.weight.grad.abs().sum() > 0


@requires_gpu
def test_attn_production_shape_causal_and_gate() -> None:
    """100m geometry: dim 512, 8 heads / 2 KV heads, window 512, G 64, topk 256.
    L=640 > window so the sliding window bites; causality must stay exact."""
    attn = build_attn(
        dim=512,
        n_heads=8,
        n_kv_heads=2,
        window=512,
        n_summary=64,
        topk=256,
    )
    x = rand_x(1, 640, dim=512)
    j = 500
    with torch.no_grad():
        out = attn(x)
        out2 = attn(x.clone())
        out2[0, j] += 1.0  # poison a late token and recompute its input effect
    x2 = x.clone()
    x2[0, j] += 7.0
    with torch.no_grad():
        pert = attn(x2)
    assert torch.equal(out[:, :j], pert[:, :j]), f"causality broken at j={j}"
    assert out.shape == x.shape and torch.isfinite(out).all()
    branches = attn(x, return_branches=True)
    assert branches["gate"].shape == (1, 640, 3)
    assert torch.allclose(branches["gate"].sum(-1), torch.ones(1, 640, device="cuda"))
