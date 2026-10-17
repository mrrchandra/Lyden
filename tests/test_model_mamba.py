"""Mamba-2 block wrapper tests: shapes, causality, prefix invariance, cache
recurrence, var-length, backward, config passthrough.

Requires CUDA (mamba-ssm kernels); other devices skip.
"""

from __future__ import annotations

import pytest
import torch

torch.manual_seed(1337)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mamba-ssm Mamba2 kernels require CUDA"
)


def build_block(dim: int = 256, **kw):
    from lyden.model.mamba_block import MambaBlock

    torch.manual_seed(42)
    block = MambaBlock(dim, **kw).cuda()
    block.eval()
    return block


def rand_x(b: int, t: int, dim: int = 256, **kw) -> torch.Tensor:
    return torch.randn(b, t, dim, device="cuda", **kw)


@requires_cuda
def test_train_forward_shapes() -> None:
    block = build_block()
    x = rand_x(3, 37)
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


@requires_cuda
def test_causality() -> None:
    """Everything before a token is unaffected by that token (exact)."""
    block = build_block()
    t_k = 20
    x1 = rand_x(2, 32)
    x2 = x1.clone()
    x2[:, t_k:] += 10.0
    with torch.no_grad():
        out1, out2 = block(x1), block(x2)
    assert torch.equal(out1[:, :t_k], out2[:, :t_k])
    assert not torch.equal(out1[:, t_k:], out2[:, t_k:])


@requires_cuda
def test_prefix_length_invariance() -> None:
    """Truncating the sequence leaves all prefix outputs unchanged."""
    block = build_block()
    x = rand_x(2, 64)
    prefix = 21
    with torch.no_grad():
        out_full = block(x)
        out_pre = block(x[:, :prefix])
    assert torch.allclose(out_pre, out_full[:, :prefix], rtol=1e-4, atol=1e-4)


@requires_cuda
def test_config_passthrough() -> None:
    from lyden.config import MambaConfig

    cfg = MambaConfig(d_state=64, d_conv=3, expand=2)
    block = build_block(d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)
    assert block.mixer.d_state == cfg.d_state
    assert block.mixer.d_conv == cfg.d_conv
    assert block.mixer.expand == cfg.expand
    x = rand_x(1, 16)
    assert block(x).shape == x.shape


@requires_cuda
def test_dim_must_satisfy_headdim_constraint() -> None:
    """expand * dim must be a multiple of mamba-ssm headdim (64); error must be explicit."""
    with pytest.raises(ValueError, match="headdim"):
        build_block(dim=96, expand=3, d_state=8)


@requires_cuda
def test_dim_dinproj_alignment_validated() -> None:
    """d_in_proj = 4*dim + 2*d_state + 2*dim/64 must be 8-aligned (recurrent path)."""
    with pytest.raises(ValueError, match="d_in_proj"):
        build_block(dim=128)


@requires_cuda
def test_cache_prefill_step_equals_parallel_scan() -> None:
    """Prefill + recurrent steps must reproduce the parallel train path."""
    from mamba_ssm.utils.generation import InferenceParams

    block = build_block(dim=256, d_state=64)
    b, t, prefix = 2, 48, 16
    x = rand_x(b, t)

    with torch.no_grad():
        out_full = block(x)  # parallel scan, no cache

        cache = InferenceParams(max_seqlen=t, max_batch_size=b)
        out_pre = block(x[:, :prefix], cache)
        cache.seqlen_offset += prefix
        assert cache.key_value_memory_dict  # cache slots populated
        steps = [out_pre]
        for i in range(prefix, t):
            out_tok = block(x[:, i : i + 1], cache)
            cache.seqlen_offset += 1
            steps.append(out_tok)
        out_rec = torch.cat(steps, dim=1)

    assert torch.allclose(out_full[:, :prefix], out_pre, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out_full[:, prefix:], out_rec[:, prefix:], rtol=1e-2, atol=1e-2)


@requires_cuda
def test_cache_respects_layer_idx_slots() -> None:
    """Two blocks sharing one InferenceParams use distinct keyed slots."""
    from mamba_ssm.utils.generation import InferenceParams

    b0, b1 = build_block(), build_block()
    b1.layer_idx, b1.mixer.layer_idx = 1, 1  # second block owns slot 1
    x = rand_x(1, 12)
    cache = InferenceParams(max_seqlen=12, max_batch_size=1)
    with torch.no_grad():
        b0(x, cache)
        b1(x, cache)
    assert set(cache.key_value_memory_dict) == {0, 1}


@requires_cuda
def test_varlen_seqidx_isolates_documents() -> None:
    """Packed documents (seq_idx) must match per-document independent runs."""
    block = build_block()
    t_doc = 16
    d1 = torch.randn(1, t_doc, 256, device="cuda")
    d2 = torch.randn(1, t_doc, 256, device="cuda")
    packed = torch.cat([d1, d2], dim=1)
    seq_idx = torch.zeros(1, 2 * t_doc, dtype=torch.int32, device="cuda")
    seq_idx[:, t_doc:] = 1
    with torch.no_grad():
        out_packed = block(packed, seq_idx=seq_idx)
        out1, out2 = block(d1), block(d2)
    assert torch.allclose(out_packed[:, :t_doc], out1, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out_packed[:, t_doc:], out2, rtol=1e-3, atol=1e-3)


@requires_cuda
def test_backward_populates_grads() -> None:
    block = build_block(dropout=0.1)
    block.train()
    x = rand_x(2, 32, requires_grad=True)
    out = block(x)
    loss = out.square().mean()
    loss.backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).any() for g in grads)
    assert x.grad is not None
    assert not any(torch.isnan(g).all() for g in grads)
