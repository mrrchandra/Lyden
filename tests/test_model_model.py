"""Transformer block assembly + full hybrid model tests (M6).

Block contract (spec §3.1):
  m: x + Mamba(rms1(x)) -> out + DenseFFN(rms2(out))
  a: x + Attention(rms1(x)) -> out + MoE(rms2(out))
Model: emb -> blocks(layer_pattern) -> final rms -> tied-head logits.

Requires CUDA (mamba kernel + canonical paths).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from lyden.config import ModelConfig

torch.manual_seed(1337)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="canonical model tests need GPU"
)


def nano_cfg(**kw: Any) -> ModelConfig:
    utils: dict[str, Any] = dict(
        vocab_size=8192, dim=256, n_layers=4, layer_pattern="mmma"
    )
    utils.update(kw)
    return ModelConfig(**utils)


def ids(b: int, t: int, vocab: int = 8192) -> torch.Tensor:
    return torch.randint(0, vocab, (b, t), device="cuda")


# ---------------------------------------------------------------- DenseFFN
@requires_gpu
def test_dense_ffn_shapes_and_configs() -> None:
    from lyden.config import DenseFFNConfig
    from lyden.model.dense_ffn import DenseFFN

    for act in ("swiglu", "gelu"):
        ffn = DenseFFN(DenseFFNConfig(hidden=320, activation=act), dim=256).cuda()
        x = torch.randn(2, 5, 256, device="cuda")
        y = ffn(x)
        assert y.shape == x.shape and torch.isfinite(y).all()


@requires_gpu
def test_dense_ffn_swiglu_reference() -> None:
    import torch.nn.functional as F

    from lyden.model.dense_ffn import DenseFFN

    torch.manual_seed(5)
    ffn = DenseFFN(
        __import__("lyden.config", fromlist=["DenseFFNConfig"]).DenseFFNConfig(
            hidden=64, activation="swiglu"
        ),
        dim=32,
    ).cuda()
    x = torch.randn(1, 4, 32, device="cuda")
    ref = ffn.down(F.silu(ffn.up(x)) * ffn.gate(x))
    assert torch.equal(ffn(x), ref)


@requires_gpu
def test_dense_ffn_gelu_reference() -> None:
    import torch.nn.functional as F

    from lyden.model.dense_ffn import DenseFFN

    torch.manual_seed(5)
    ffn = DenseFFN(
        __import__("lyden.config", fromlist=["DenseFFNConfig"]).DenseFFNConfig(
            hidden=48, activation="gelu"
        ),
        dim=32,
    ).cuda()
    x = torch.randn(1, 4, 32, device="cuda")
    ref = ffn.fc2(F.gelu(ffn.fc1(x)))
    assert torch.equal(ffn(x), ref)


# ---------------------------------------------------------------- LydenBlock
@requires_gpu
def test_block_m_residual_plumbing_exact() -> None:
    from lyden.model.block import LydenBlock

    cfg = nano_cfg()
    torch.manual_seed(3)
    blk = LydenBlock("m", cfg, layer_idx=0).cuda().eval()
    x = torch.randn(1, 16, 256, device="cuda")
    with torch.no_grad():
        out = blk(x)
        ref = x + blk.mixer(blk.norm1(x))
        ref = ref + blk.ffn(blk.norm2(ref))
    assert torch.equal(out, ref)


@requires_gpu
def test_block_a_residual_plumbing_exact() -> None:
    from lyden.model.block import LydenBlock

    cfg = nano_cfg()
    torch.manual_seed(3)
    blk = LydenBlock("a", cfg, layer_idx=1).cuda().eval()  # unused idx, harmless
    x = torch.randn(1, 16, 256, device="cuda")
    with torch.no_grad():
        out = blk(x)
        ref = x + blk.mixer(blk.norm1(x))
        ref = ref + blk.ffn(blk.norm2(ref))
    assert torch.equal(out, ref)


@requires_gpu
def test_block_mixer_types_match_kind() -> None:
    from lyden.model.block import LydenBlock
    from lyden.model.hybrid_attention import HybridAttention
    from lyden.model.mamba_block import MambaBlock
    from lyden.model.moe import MoE

    cfg = nano_cfg()
    mb = LydenBlock("m", cfg, layer_idx=0)
    assert isinstance(mb.mixer, MambaBlock)
    from lyden.model.dense_ffn import DenseFFN

    assert isinstance(mb.ffn, DenseFFN)
    ab = LydenBlock("a", cfg, layer_idx=1)
    assert isinstance(ab.mixer, HybridAttention)
    assert isinstance(ab.ffn, MoE)
    with pytest.raises(ValueError, match="kind"):
        LydenBlock("q", cfg, layer_idx=0)


# ---------------------------------------------------------------- LydenModel
@requires_gpu
def test_model_logits_shapes_finite_deterministic() -> None:
    from lyden.model.model import LydenModel

    model = LydenModel(nano_cfg()).cuda().eval()
    tok = ids(2, 17)
    with torch.no_grad():
        logits = model(tok)
        logits2 = model(tok)
    assert logits.shape == (2, 17, 8192)
    assert torch.isfinite(logits).all()
    assert torch.equal(logits, logits2)


@requires_gpu
def test_model_end_to_end_causal_logits_exact() -> None:
    from lyden.model.model import LydenModel

    model = LydenModel(nano_cfg()).cuda().eval()
    tok = ids(1, 24)
    j = 13
    with torch.no_grad():
        base = model(tok)
        tok2 = tok.clone()
        tok2[0, j] = (tok2[0, j] + 7) % 8192
        pert = model(tok2)
    # kernel tolerance, not bitwise: MoE per-expert GEMM shapes shift when the
    # perturbed token reroutes (cuBLAS is shape-sensitive at ~1e-7)
    assert torch.allclose(base[:, :j], pert[:, :j], rtol=0.0, atol=1e-4)


@requires_gpu
def test_model_weight_tying() -> None:
    from lyden.model.model import LydenModel

    model_tie = LydenModel(nano_cfg(weight_tying=True)).cuda()
    assert model_tie.head.weight is model_tie.emb.weight
    model_free = LydenModel(nano_cfg(weight_tying=False)).cuda()
    assert model_free.head.weight is not model_free.emb.weight
    extra = sum(p.numel() for p in model_free.parameters()) - sum(
        p.numel() for p in model_tie.parameters()
    )
    assert extra == model_free.emb.weight.numel()


@requires_gpu
def test_model_backward_reaches_fast_and_slow_params() -> None:
    from lyden.model.model import LydenModel

    model = LydenModel(nano_cfg()).cuda().train()
    out = model(ids(1, 12)).square().mean()
    out.backward()
    fast = [p for n, p in model.named_parameters() if "router" in n]
    embg = [p.grad for p in model.emb.parameters()]
    assert fast and fast[0].grad is not None and fast[0].grad.abs().sum() > 0
    assert any(g is not None and g.abs().sum() > 0 for g in embg)


@requires_gpu
def test_model_unique_mamba_layer_idx() -> None:
    from lyden.model.model import LydenModel

    model = LydenModel(nano_cfg(n_layers=8, layer_pattern="mmma")).cuda()
    observed = [blk.mixer.layer_idx for blk in model.blocks if blk.kind == "m"]  # type: ignore[union-attr]
    expected = [i for i, kind in enumerate(model.cfg.layer_kinds()) if kind == "m"]
    assert observed == expected
    assert len(set(observed)) == len(observed)  # unique for mamba caches


@requires_gpu
def test_model_nano_yaml_builds_and_forward() -> None:
    from lyden.config import load_config
    from lyden.model.model import LydenModel

    cfg = load_config("configs/nano.yaml")
    model = LydenModel(cfg.model).cuda().eval()
    logits = model(ids(1, 33, vocab=cfg.model.vocab_size))
    assert logits.shape == (1, 33, cfg.model.vocab_size)
    assert torch.isfinite(logits).all()