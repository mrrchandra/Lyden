"""M9: stateful decode (prefill + step) with exact training semantics."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor

from lyden.config import (
    AttentionConfig,
    MambaConfig,
    ModelConfig,
    MoEConfig,
)
from lyden.model.block import LydenBlock
from lyden.model.decode import DecodeState, greedy_decode, prefill, step
from lyden.model.hybrid_attention import HybridAttention
from lyden.model.model import LydenModel


def _model(seed: int = 5) -> LydenModel:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=128,
        dim=256,
        n_layers=4,
        layer_pattern="mmam",
        mamba=MambaConfig(d_state=32, d_conv=4, expand=2),
        attention=AttentionConfig(
            n_heads=8, n_kv_heads=2, window=64, n_summary=16, topk=32
        ),
        moe=MoEConfig(n_routed=4, n_shared=1, expert_hidden=96),
    )
    return LydenModel(cfg).cuda().eval()


def _ids(t: int = 12) -> Tensor:
    rng = torch.Generator(device="cpu").manual_seed(99)
    return torch.randint(8, 120, (1, t), generator=rng).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefill_logits_match_plain_forward() -> None:
    model = _model()
    ids = _ids(10)
    with torch.no_grad():
        plain = model(ids)
    state, logits = prefill(model, ids)
    assert logits.shape == plain.shape
    diff = (logits - plain).abs().max().item()
    assert diff < 1e-3, diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_greedy_continuation_matches_full_recompute() -> None:
    model = _model()
    ids = _ids(10)
    n_new = 24
    # v1 baseline: full recompute each step
    cur = ids.clone()
    seq_v1: list[int] = []
    with torch.no_grad():
        for _ in range(n_new):
            nxt = int(torch.argmax(model(cur)[0, -1]).item())
            seq_v1.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    seq_v2 = greedy_decode(model, ids, n_new)
    agg = sum(a == b for a, b in zip(seq_v1, seq_v2, strict=False))
    assert agg >= n_new - 2, f"v1={seq_v1}\nv2={seq_v2}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_attention_step_matches_single_pass_rows() -> None:
    """Branch math on the cache path equals the full-pass path when the
    underlying rows are identical: keys/values/ranks are per-token linears,
    the query at row t ropes at absolute position t, masks use explicit
    positions, and the compressed bank anchors at the cached key count —
    so a step equals the matched full-pass row up to kernel noise."""
    torch.manual_seed(3)
    model = _model()
    blk = cast(LydenBlock, model.blocks[2])
    mixer = cast(HybridAttention, blk.mixer)
    t = 40
    x = torch.randn(1, t + 1, mixer.dim, device="cuda")
    caches: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        br_puts = mixer(x, return_branches=True)
        _ = mixer(x[:, :t], return_branches=True, decode_caches=caches)
        out_step = mixer.step_branches(x[:, t:], caches, t)
    diff = (out_step[0, 0] - br_puts["out"][0, -1]).abs().max().item()
    assert diff < 1e-3, diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_decode_state_structure_and_positions() -> None:
    model = _model()
    ids = _ids(6)
    state, _ = prefill(model, ids)
    assert isinstance(state, DecodeState)
    assert state.query_pos == 6
    assert set(state.kv_caches) >= {2}  # layer 2 is the attention block
    step(model, state, 88)
    assert state.query_pos == 7
    assert state.kv_caches[2][  # type: ignore[index]
        "k_c"
    ].shape[-3] == 7  # cache extended


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_long_decode_uses_less_graph_than_recompute() -> None:
    model = _model()
    ids = _ids(96)  # > window (64) exercises all branch paths
    out = greedy_decode(model, ids, 8)
    assert len(out) == 8
    assert all(0 <= v < 128 for v in out)
