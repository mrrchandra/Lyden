"""Hybrid attention block (spec §3.2): three gated branch paths over shared GQA K/V.

Branches (each with its own Q projection; one shared K/V projection and one
RoPE table; internal tensor layout (B, heads, tokens, head_dim)):
  1. `_sliding`: causal windowed attention (band mask, manual ops).
  2. `_compressed`: learnable summary bank per KV head, summary i anchored at
     min((i+1)*L // G, L-1). Summary i attends tokens <= anchor_i; token t
     attends summaries with anchor_i <= t (causally consistent; the bank is
     rebuilt from cache/prefill states at decode time).
  3. `_selective`: learned indexer scores token relevance; hard top-k per
     query over causally-masked scores; gradients flow through the gathered
     values path (and softmax of gathered scores).

Gate: ``softmax(Linear(dim -> 3)(x_t))`` mixes branch outputs in the
concat-head space before one out-projection. RoPE (LLaMA rotate-half)
applies to Q and K in all branches at absolute positions. Output = residual.

M4 scope is training semantics (parallel full-sequence attention); recurrent
decode wiring arrives with model assembly and cache-prefill handling.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def rope_cos_sin(seqlen: int, head_dim: int, base: float, device: torch.device):
    """LLaMA-style reciprocal-freq table; (cos, sin) shaped (T, Dh//2)."""
    half = head_dim // 2
    positions = torch.arange(seqlen, device=device)
    inv_freq = base ** (-2.0 * torch.arange(half, device=device) / head_dim)
    angles = positions[:, None].float() * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate-half RoPE; x: (B, T, KV, Dh) token-major; cos/sin: (T, Dh//2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_ = cos[None, :, None]
    sin_ = sin[None, :, None]
    return torch.cat([x1 * cos_ - x2 * sin_, x2 * cos_ + x1 * sin_], dim=-1)


def apply_rope_pqmaj(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate-half RoPE; x: (B, H, L, Dh) head-major; cos/sin: (L, Dh//2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_ = cos[None, None, :, :]
    sin_ = sin[None, None, :, :]
    return torch.cat([x1 * cos_ - x2 * sin_, x2 * cos_ + x1 * sin_], dim=-1)


def rope_at_positions(x: Tensor, positions: Tensor, head_dim: int, base: float) -> Tensor:
    """RoPE at explicit positions; x: (B, G, KV, Dh), positions: (G,)."""
    half = head_dim // 2
    inv_freq = base ** (-2.0 * torch.arange(half, device=x.device) / head_dim)
    angles = positions[None, :, None].float() * inv_freq[None, None, :]  # (1, G, half)
    angles = angles[..., None, :]  # (1, G, 1, half) — keep KV axis from broadcasting
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def gqa_expand(k: Tensor, n_query_heads: int, n_kv_heads: int) -> Tensor:
    """(..., KV, Dh) -> (..., H, Dh) repeating each KV head over its group."""
    if n_query_heads == n_kv_heads:
        return k
    group = n_query_heads // n_kv_heads
    kv_axes = k.shape[:-1]
    out = k.unsqueeze(-2).expand(*kv_axes, group, k.shape[-1])
    return out.reshape(*kv_axes[:-1], kv_axes[-1] * group, k.shape[-1])


def _split_heads(x: Tensor, n_heads: int, head_dim: int) -> Tensor:
    """(B, L, D) -> (B, H, L, Dh)."""
    b, seq_len, _ = x.shape
    return x.view(b, seq_len, n_heads, head_dim).transpose(1, 2)


def _join_heads(x: Tensor) -> Tensor:
    """(B, H, L, Dh) -> (B, L, D)."""
    b, h, seq_len, dh = x.shape
    return x.transpose(1, 2).reshape(b, seq_len, h * dh)


def _anchors(seq_len: int, g: int) -> Tensor:
    """Anchor of summary i = i * L // G (spec formula i*(T/G)); anchor_0 = 0.

    anchor_0 = 0 guarantees every token has at least one readable summary
    (otherwise the branch softmax would be all -inf for early tokens).
    """
    return (torch.arange(g) * seq_len // g).clamp(max=seq_len - 1)


class HybridAttention(nn.Module):
    """Three-branch hybrid attention; causal consistency enforced in every branch."""

    def __init__(
        self,
        dim: int,
        *,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        window: int = 512,
        n_summary: int = 64,
        topk: int = 256,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.window = window
        self.n_summary = n_summary
        self.topk = topk
        self.rope_base = rope_base

        hd = self.head_dim
        self.kv_proj = nn.Linear(dim, 2 * n_kv_heads * hd, bias=False)
        self.q_sliding = nn.Linear(dim, n_heads * hd, bias=False)
        self.q_compressed = nn.Linear(dim, n_heads * hd, bias=False)
        self.q_selective = nn.Linear(dim, n_heads * hd, bias=False)

        self.summary_queries = nn.Parameter(torch.randn(n_summary, n_kv_heads, hd) * 0.02)

        self.indexer = nn.Linear(dim, 1, bias=False)
        self.gate = nn.Linear(dim, 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ pieces
    def _split_kv(self, kv: Tensor) -> tuple[Tensor, Tensor]:
        """(B, L, 2*KV*Dh) -> ((B, L, KV, Dh), (B, L, KV, Dh))."""
        k, v = kv.chunk(2, dim=-1)
        n_kv, hd = self.n_kv_heads, self.head_dim
        b, seq_len, _ = k.shape
        return k.view(b, seq_len, n_kv, hd), v.view(b, seq_len, n_kv, hd)

    def _proj_q(self, name: str, x: Tensor) -> Tensor:
        """(B, L, D) -> (B, H, L, Dh) rope-ready."""
        return _split_heads(getattr(self, name)(x), self.n_heads, self.head_dim)

    def _head_kv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """(B, L, KV, Dh) -> ((B, H, L, Dh), (B, H, L, Dh)) GQA-expanded."""

        def expand_and_layout(t: Tensor) -> Tensor:
            return gqa_expand(t, self.n_heads, self.n_kv_heads).transpose(1, 2)

        return expand_and_layout(k), expand_and_layout(v)

    # ------------------------------------------------------------------ branches
    def _sliding(self, q: Tensor, k: Tensor, v: Tensor, q_pos: Tensor | None = None) -> Tensor:
        """Causal window attention: j allowed iff t - window < j <= t.

        ``q_pos`` overrides query positions (absolute); keys always use
        0..Lk-1 (cached buffers are contiguous fresh re-encodings). Default
        None = 0..Lq-1, matching the training path exactly.
        """
        dh = self.head_dim
        lk = k.shape[2]
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)
        if q_pos is None:
            pos = torch.arange(lk, device=q.device)
            causal = pos[None, :] <= pos[:, None]
            in_win = pos[:, None] - pos[None, :] < self.window  # t - j < window
        else:
            pos_k = torch.arange(lk, device=q.device)
            causal = pos_k[None, :] <= q_pos[:, None]
            in_win = q_pos[:, None] - pos_k[None, :] < self.window
        scores = scores.masked_fill(~(causal & in_win)[None, None], float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        return _join_heads(out)

    def _compute_summary_bank(self, k_kv: Tensor, v_kv: Tensor) -> tuple[Tensor, Tensor]:
        """Anchored summary bank; k/v: (B, KV, L, Dh) un-expanded.

        Returns (K_bank, V_bank), each (B, G, KV, Dh); K rope'd at anchors.
        Summary i attends tokens <= anchor_i = i*L//G (anchor_0 = 0).
        """
        b, kv, seq_len, dh = k_kv.shape
        g = self.n_summary
        anchors = _anchors(seq_len, g).to(k_kv.device)
        s_tok = self.summary_queries[None].expand(b, g, kv, dh)
        s_tok = rope_at_positions(s_tok, anchors, dh, self.rope_base)  # (B, G, KV, Dh)
        s = s_tok.transpose(1, 2)  # (B, KV, G, Dh)
        scores = s @ k_kv.transpose(-1, -2) / math.sqrt(dh)  # (B, KV, G, L)
        allowed_g = torch.arange(seq_len, device=k_kv.device) <= anchors[:, None]  # (G, L)
        scores = scores.masked_fill(~allowed_g[None, None], float("-inf"))
        rep = torch.softmax(scores, dim=-1) @ v_kv  # (B, KV, G, Dh)
        raw = rep.permute(0, 2, 1, 3)  # (B, G, KV, Dh)
        k_bank = rope_at_positions(raw, anchors, dh, self.rope_base)
        return k_bank, raw

    def _compressed(
        self, q: Tensor, k_kv: Tensor, v_kv: Tensor, q_pos: Tensor | None = None
    ) -> Tensor:
        """Compressed global: token t reads summary anchors <= t.

        q: (B, H, L, Dh) head-major; k_kv/v_kv: (B, KV, L, Dh) un-expanded.
        ``q_pos`` overrides query positions; anchors derive from the key
        length (anchors_i = i * Lk // G), matching a fresh full re-encode.
        """
        seq_len = v_kv.shape[2]
        dh = self.head_dim
        k_bank, v_bank = self._compute_summary_bank(k_kv, v_kv)
        bank_k = gqa_expand(k_bank, self.n_heads, self.n_kv_heads)  # (B, G, H, Dh)
        bank_v = gqa_expand(v_bank, self.n_heads, self.n_kv_heads)
        bank_k, bank_v = bank_k.transpose(1, 2), bank_v.transpose(1, 2)  # (B, H, G, Dh)
        anchors = _anchors(seq_len, self.n_summary).to(q.device)
        scores = q @ bank_k.transpose(-1, -2) / math.sqrt(dh)  # (B, H, L, G)
        if q_pos is None:
            readable = anchors[None, :] <= torch.arange(seq_len, device=q.device)[:, None]
        else:
            readable = anchors[None, :] <= q_pos[:, None]
        scores = scores.masked_fill(~readable[None, None], float("-inf"))
        out = torch.softmax(scores, dim=-1) @ bank_v
        return _join_heads(out)

    def _selective(
        self, q: Tensor, k: Tensor, v: Tensor, x: Tensor, q_pos: Tensor | None = None
    ) -> Tensor:
        """Hard top-k attention where selection ranks come from the indexer.

        Selection (M4-frozen semantics): indexer token-importance scores of
        each QUERY token (constant per query row) causally masked then top-k
        per query -> boolean selector; ties resolve by index as in the full
        path, so decode rows reproduce recompute rows. Attention: standard
        rope'd q.k scores over the chosen support.
        ``q_pos`` overrides query positions.
        """
        b, h, lq, dh = q.shape
        lk = k.shape[2]
        ranks_q = self.indexer(x).squeeze(-1)  # (B, lq): x = pre-norm input for rows
        if q_pos is None:
            t_pos = torch.arange(lk, device=q.device)
            q_pos = t_pos
        k_pos = torch.arange(lk, device=q.device)
        causal = k_pos[None, :] <= q_pos[:, None]  # (q_t, cand j)
        rank_mat = ranks_q[:, :, None].expand(b, lq, lk)
        rank_mat = rank_mat.masked_fill(~causal[None], float("-inf"))  # (B, lq, lk)
        k_sel = min(self.topk, lk)
        top = torch.topk(rank_mat, k_sel, dim=-1).indices  # (B, lq, k)
        selector = torch.zeros(b, lq, lk, device=q.device).scatter_(2, top, 1.0).bool()

        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)  # (B, H, lq, lk)
        allow = (selector & causal)[:, None]  # (B, 1, lq, lk) broadcast over heads
        scores = scores.masked_fill(~allow, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v  # (B, H, lq, Dh)
        return _join_heads(out)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: Tensor,
        *,
        return_branches: bool = False,
        decode_caches: dict[str, Tensor] | None = None,
    ) -> dict | Tensor:
        """Full-sequence forward; `return_branches=True` exposes per-branch
        outputs and gate weights (dict) for tests and training-time logging.

        Note: the compressed branch's summary anchors depend on the total
        sequence length L (anchor_i = i * L / G), so its outputs for a
        prefix of one run are not the prefix outputs of another run — the
        sliding and selective branches are length-independent in prefix.

        ``decode_caches`` (prefill capture): given a dict, this call is a
        PREFILL — the training math runs unchanged and rope'd keys, values
        and indexer ranks are exported into it for the per-token step path.
        """
        b, seq_len, _ = x.shape
        k, v = self._split_kv(self.kv_proj(x))
        cos, sin = rope_cos_sin(seq_len, self.head_dim, self.rope_base, x.device)
        k = apply_rope(k, cos, sin)
        k_h, v_h = self._head_kv(k, v)

        q_sl = apply_rope_pqmaj(self._proj_q("q_sliding", x), cos, sin)
        q_co = apply_rope_pqmaj(self._proj_q("q_compressed", x), cos, sin)
        q_se = apply_rope_pqmaj(self._proj_q("q_selective", x), cos, sin)

        b_sl = self._sliding(q_sl, k_h, v_h)
        b_co = self._compressed(q_co, k.transpose(1, 2), v.transpose(1, 2))
        b_se = self._selective(q_se, k_h, v_h, x)

        weights = self.gate(x).softmax(dim=-1)  # (B, L, 3)
        mixed = (
            weights[..., 0:1] * b_sl
            + weights[..., 1:2] * b_co
            + weights[..., 2:3] * b_se
        )
        out = x + self.dropout(self.out_proj(mixed))
        if decode_caches is not None:
            decode_caches["k_c"] = k.detach()
            decode_caches["v_c"] = v.detach()
            decode_caches["ranks"] = self.indexer(x).squeeze(-1).detach()
        if return_branches:
            return {
                "out": out,
                "sliding": b_sl,
                "compressed": b_co,
                "selective": b_se,
                "gate": weights,
            }
        return out

    def step_branches(
        self, x_one: Tensor, caches: dict[str, Tensor], position: int
    ) -> Tensor:
        """One-token decode step; training-math equivalent.

        x_one: (B, 1, D) pre-norm input; ``caches`` holds rope'd keys
        (B, T, KV, Dh), values (B, T, KV, Dh), and indexer ranks (B, T)
        from prefill/steps; ``position`` is the query's absolute index.
        The compressed bank is rebuilt over the full cached context
        (anchor'ed at the key length, matching a fresh full re-encode).
        Mutates ``caches`` with the token's own k/v/rank.
        """
        dh = self.head_dim
        half = dh // 2
        k_new, v_new = self._split_kv(self.kv_proj(x_one))
        k_new = rope_at_positions(
            k_new, torch.tensor([position], device=x_one.device), dh, self.rope_base
        )
        k_c = torch.cat([caches["k_c"], k_new], dim=1)
        v_c = torch.cat([caches["v_c"], v_new], dim=1)
        ranks_c = torch.cat([caches["ranks"], self.indexer(x_one).squeeze(-1)], dim=1)

        # RoPE at the single absolute query position (matches rope_cos_sin
        # table entries exactly: same reciprocal-frequency definition).
        inv_freq = self.rope_base ** (-2.0 * torch.arange(half, device=x_one.device) / dh)
        pos_ang = float(position) * inv_freq
        ang, ang_sin = pos_ang.cos(), pos_ang.sin()

        def rope_q_at_pos(q: Tensor) -> Tensor:
            # q: (B, H, 1, Dh) -> rotate-half at one absolute position
            q1, q2 = q[..., :half], q[..., half:]
            return torch.cat([q1 * ang - q2 * ang_sin, q2 * ang + q1 * ang_sin], dim=-1)

        q_sl = rope_q_at_pos(self._proj_q("q_sliding", x_one))
        q_co = rope_q_at_pos(self._proj_q("q_compressed", x_one))
        q_se = rope_q_at_pos(self._proj_q("q_selective", x_one))

        k_h, v_h = self._head_kv(k_c, v_c)
        q_positions = torch.full((1,), float(position), device=x_one.device)
        b_sl = self._sliding(q_sl, k_h, v_h, q_pos=q_positions)
        b_co = self._compressed(q_co, k_c.transpose(1, 2), v_c.transpose(1, 2), q_pos=q_positions)
        b_se = self._selective(q_se, k_h, v_h, x_one, q_pos=q_positions)

        weights = self.gate(x_one).softmax(dim=-1)  # (B, 1, 3)
        mixed = (
            weights[..., 0:1] * b_sl
            + weights[..., 1:2] * b_co
            + weights[..., 2:3] * b_se
        )
        out = x_one + self.dropout(self.out_proj(mixed))
        caches["k_c"], caches["v_c"], caches["ranks"] = k_c, v_c, ranks_c
        return out
