"""MoE feed-forward tests (M5): routing determinism, top-k semantics, shared
experts still active, backward reach, loss-free bias balancing, Switch aux
loss, expert math reference, config passthrough.

Requires CUDA only (same machine as all canonical tests).
"""

from __future__ import annotations

import pytest
import torch

from lyden.config import MoEConfig

torch.manual_seed(1337)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="canonical MoE tests need GPU"
)


def build_moe(n_routed: int = 4, **kw):
    from lyden.model.moe import MoE

    torch.manual_seed(7)
    cfg = MoEConfig(n_routed=n_routed, **kw)
    return MoE(cfg, dim=128, dropout=0.0).cuda()


def rand_x(b: int, t: int, dim: int = 128, **kw) -> torch.Tensor:
    return torch.randn(b, t, dim, device="cuda", **kw)


def routed_experts(moe):
    return [m for m in moe.routed.values()]


# ------------------------------------------------------------ wave 1: basics
@requires_gpu
def test_moe_shapes_and_finite() -> None:
    moe = build_moe()
    x = rand_x(2, 10)
    out = moe(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


@requires_gpu
def test_moe_deterministic() -> None:
    moe = build_moe()
    x = rand_x(1, 8)
    with torch.no_grad():
        a, b = moe(x), moe(x)
    assert torch.equal(a, b)


@requires_gpu
def test_moe_top_k_selection_counts() -> None:
    moe = build_moe(n_routed=4)
    x = rand_x(2, 5)
    with torch.no_grad():
        _, stats = moe(x, collect_stats=True)
    load = stats["load"]  # (n_routed,) fraction of token-expert assignments
    assert load.shape == (4,)
    assert torch.allclose(load.sum(), torch.tensor(2.0, device="cuda"))  # top_k=2


@requires_gpu
def test_moe_gate_weights_renormalized() -> None:
    moe = build_moe(n_routed=4)
    x = rand_x(3, 7)
    with torch.no_grad():
        gate_vals = moe(x, collect_stats=True)[1]["gate_values"]  # (B, L, k)
    assert gate_vals.shape == (3, 7, 2)
    sums = gate_vals.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@requires_gpu
def test_moe_shared_experts_active_with_routed_zeroed() -> None:
    """Zero all routed-expert weights: output must be exactly the shared path
    (non-zero), proving shared experts are always active."""
    moe = build_moe(n_routed=2)
    with torch.no_grad():
        for expert in moe.routed.values():
            for prm in expert.parameters():
                prm.zero_()
    x = rand_x(1, 6)
    with torch.no_grad():
        out = moe(x)
        ref = torch.zeros_like(x)
        for mod in moe.shared.values():
            ref = ref + moe._expert_forward(mod, x)
    assert out.abs().sum() > 0
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)


@requires_gpu
def test_moe_backward_reaches_router_and_experts() -> None:
    moe = build_moe()
    moe.train()
    x = rand_x(1, 12, requires_grad=True)
    moe(x).square().mean().backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    assert any(
        q.grad is not None and q.grad.abs().sum() > 0
        for expert in moe.routed.values()
        for q in expert.parameters()
    )
    assert any(
        q.grad is not None and q.grad.abs().sum() > 0
        for expert in moe.shared.values()
        for q in expert.parameters()
    )
    assert x.grad is not None


# ------------------------------------------------- wave 2: load balancing
@requires_gpu
def test_moe_loss_free_bias_not_trainable() -> None:
    moe = build_moe()
    assert moe.bias.shape == (4,)
    names = [n for n, _ in moe.named_parameters()]
    assert not any("bias" in n for n in names), "bias must be a buffer, not a parameter"


@requires_gpu
def test_moe_overloaded_expert_bias_decreases() -> None:
    """Crafted scores [0.9, 0.2, 0.1, 0.0] -> every token selects {0, 1};
    load (0.5, 0.5, 0, 0) vs target k/N=0.5: nothing overloaded strictly, so
    raise load to 1.0 by using top_k=1 -> experts 2,3 underloaded (bias up),
    expert 0 overloaded (bias down)."""
    moe = build_moe(n_routed=4, top_k=1)
    x = torch.zeros(1, 8, 128, device="cuda")
    x[..., 0] = 1.0
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 0.9
        moe.router.weight[1, 0] = 0.2
        moe.router.weight[2, 0] = 0.1
        s = moe.router(x)  # (1, 8, 4) = [0.9, 0.2, 0.1, 0]
        assert torch.allclose(s[0, 0], torch.tensor([0.9, 0.2, 0.1, 0.0], device="cuda"))
        _, stats = moe(x, collect_stats=True)
        assert torch.allclose(stats["load"], torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"))
        b0 = moe.bias.clone()
        moe.update_bias()
        assert moe.bias[0] < b0[0], "overloaded expert 0 gets decreased"
        for e in (1, 2, 3):
            assert moe.bias[e] > b0[e], f"underloaded expert {e} gets increased"
        # exact step size gamma
        assert torch.allclose(
            moe.bias - b0, torch.tensor([-0.001, 0.001, 0.001, 0.001], device="cuda")
        )


@requires_gpu
def test_moe_bias_shifts_selection_not_gate_values() -> None:
    moe = build_moe(n_routed=4)
    x = rand_x(1, 9)
    with torch.no_grad():
        out1, stats1 = moe(x, collect_stats=True)
        # if selection is unchanged for every token, bias may not alter gate values
        s = moe.router(x)
        top2 = torch.topk(s, 2, dim=-1).indices
        bias2 = torch.full_like(moe.bias, -10.0)
        bias2.scatter_(0, top2.reshape(-1).unique(), 10.0)
        moe.bias.copy_(bias2)
        out2, stats2 = moe(x, collect_stats=True)
    assert torch.allclose(out1, out2, atol=1e-6), "bias must not alter gate values"
    assert torch.allclose(stats1["gate_values"], stats2["gate_values"], atol=1e-6)
    assert torch.equal(stats1["selected"], stats2["selected"])


@requires_gpu
def test_moe_bias_shifts_selection_when_large() -> None:
    """Bias +10/+9/-9/-9 forces every token's selection to [0, 1], whatever
    the (seeded) random router gave before."""
    moe = build_moe(n_routed=4)
    x = rand_x(1, 9)
    with torch.no_grad():
        moe.router.weight.zero_()
        _, s0 = moe(x, collect_stats=True)
        ids0 = s0["selected"]
        moe.bias[:] = torch.tensor([10.0, 9.0, -9.0, -9.0], device="cuda")
        _, s1 = moe(x, collect_stats=True)
        ids1 = s1["selected"]
        assert not torch.equal(ids0, ids1)
        expected = torch.tensor([0, 1], device="cuda").expand(1, 9, 2)
        assert torch.equal(ids1, expected)
        # both biased experts are selected by every token (sums to k=2)
        assert torch.allclose(s1["load"], torch.tensor([1.0, 1.0, 0.0, 0.0], device="cuda"))


@requires_gpu
def test_moe_aux_mode_switch_loss_matches_reference() -> None:
    """Switch aux: L_aux = w * N * sum_e f_e * P_e; f_e = tokens routed to e
    divided by total tokens (sums to top_k)."""
    moe = build_moe(n_routed=4, balance="aux", aux_loss_weight=0.01)
    x = rand_x(2, 6)
    with torch.no_grad():
        out, stats = moe(x, collect_stats=True)
    assert torch.allclose(stats["load"].sum(), torch.tensor(2.0, device="cuda"))
    p = torch.softmax(moe.router(x).reshape(12, 4), dim=-1).mean(0)
    ref = 0.01 * 4.0 * (stats["load"] * p).sum()
    assert torch.allclose(stats["aux_loss"], ref, rtol=1e-4, atol=1e-6)
    assert stats["aux_loss"].item() > 0


@requires_gpu
def test_moe_loss_free_mode_reports_zero_aux() -> None:
    moe = build_moe()
    x = rand_x(1, 4)
    with torch.no_grad():
        _, stats = moe(x, collect_stats=True)
    assert stats["aux_loss"].item() == 0.0


# --------------------------------------------------- wave 3: expert math
@requires_gpu
def test_expert_swiglu_math_reference() -> None:
    from lyden.model.moe import Expert

    torch.manual_seed(3)
    ex = Expert(128, 64).cuda()
    x = rand_x(1, 5)
    ref = ex.down(torch.nn.functional.silu(ex.up(x)) * ex.gate(x))
    assert torch.allclose(ex(x), ref, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_moe_config_passthrough() -> None:
    moe = build_moe(
        n_routed=6, n_shared=1, top_k=3, expert_hidden=96, balance="aux", aux_loss_weight=0.02
    )
    assert len(moe.routed) == 6
    assert len(moe.shared) == 1
    assert moe.top_k == 3
    assert moe.routed["routed0"].hidden == 96
    x = rand_x(1, 5)
    with torch.no_grad():
        _, stats = moe(x, collect_stats=True)
    assert torch.allclose(stats["load"].sum(), torch.tensor(3.0, device="cuda"))
    kv = stats["gate_values"].sum(-1)
    assert torch.allclose(kv, torch.ones_like(kv))


@requires_gpu
def test_moe_routed_output_matches_manual_reference() -> None:
    """Full-path reference: renormalized gathered scores * expert outputs,
    over a loop over experts, plus shared experts additive."""
    moe = build_moe(n_routed=4)
    x = rand_x(2, 8)
    with torch.no_grad():
        out = moe(x)
        s = moe.router(x)
        top2 = torch.topk(s, 2, dim=-1).indices
        g = s.gather(-1, top2)
        g = g / g.sum(-1, keepdim=True)
        ref = torch.zeros_like(x)
        for i in range(len(moe.routed)):
            sel = top2 == i  # (B, L, k) bool
            if not sel.any():
                continue
            w = (g * sel.float()).sum(-1, keepdim=True)  # (B, L, 1): gate if selected
            ref = ref + moe._expert_forward(moe.routed[f"routed{i}"], x) * w
        for mod in moe.shared.values():
            ref = ref + moe._expert_forward(mod, x)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)