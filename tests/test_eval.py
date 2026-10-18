"""M10 harness: val ppl, long-context probe, sample generation."""

from __future__ import annotations

import pytest
import torch

from lyden.config import (
    AttentionConfig,
    DenseFFNConfig,
    LydenConfig,
    MambaConfig,
    ModelConfig,
    MoEConfig,
    from_dict,
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="eval harness tests need GPU"
)


def tiny_lyden_cfg(seq_len: int = 64) -> LydenConfig:
    model = ModelConfig(
        vocab_size=512,
        dim=256,
        n_layers=4,
        layer_pattern="mmam",
        mamba=MambaConfig(d_state=32, d_conv=4, expand=2),
        attention=AttentionConfig(n_heads=8, n_kv_heads=2, window=64, n_summary=16, topk=16),
        dense_ffn=DenseFFNConfig(hidden=128, activation="swiglu"),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, expert_hidden=64),
    )
    payload = {
        "model": model,
        "data": {"dataset": "synthetic"},
        "training": {"seq_len": seq_len, "batch_size": 2, "total_steps": 10},
    }
    return from_dict(LydenConfig, payload)


@requires_gpu
def test_perplexity_untrained_matches_ln_vocab() -> None:
    from lyden.eval import perplexity
    from lyden.model.model import LydenModel

    cfg = tiny_lyden_cfg()
    torch.manual_seed(0)
    model = LydenModel(cfg.model).cuda().eval()
    res = perplexity(model, cfg, n_batches=2, device="cuda")
    expect = torch.log(torch.tensor(float(cfg.model.vocab_size))).item()
    assert abs(res["loss"] - expect) < 0.35, (res, expect)
    assert res["ppl"] > 0


@requires_gpu
def test_long_context_ppl_runs_all_lengths() -> None:
    from lyden.eval import long_context_ppl
    from lyden.model.model import LydenModel

    cfg = tiny_lyden_cfg(seq_len=256)
    torch.manual_seed(1)
    model = LydenModel(cfg.model).cuda().eval()
    out = long_context_ppl(model, cfg, lengths=(64, 128, 256), n_batches=1, device="cuda")
    assert set(out) == {"64", "128", "256"}
    for _sl, r in out.items():
        assert 0.0 < r["ppl"] < float("inf")


@requires_gpu
def test_sample_text_prefill_step_uses_state() -> None:
    from lyden.eval import sample_text
    from lyden.model.model import LydenModel

    cfg = tiny_lyden_cfg()
    torch.manual_seed(2)
    model = LydenModel(cfg.model).cuda().eval()

    class _FakeTok:
        bos_id = 1
        eos_id = -1  # sampling can never hit it; length is deterministic

        def encode(self, text: str, add_bos: bool = False) -> list[int]:
            return [17 + (i % 200) for i in range(len(text.split()))] or [3]

        def decode(self, ids: list[int], skip_special: bool = False) -> str:
            return " ".join(str(i) for i in ids)

    text = sample_text(model, _FakeTok(), "hello world", n_new=6, temperature=0.0)
    assert len(text.split()) == 6
