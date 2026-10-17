"""Trainer integration tests (M7): split-LR groups, warmup/cosine schedule,
learnable-stream loss decrease, checkpoint save/load/resume, keep-last
pruning, stability monitors, AMP variants, model stats plumbing.

Requires CUDA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lyden.config import LydenConfig, ModelConfig, MoEConfig
from lyden.model.model import LydenModel
from lyden.train import Trainer

torch.manual_seed(1337)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NANO_MOE = MoEConfig(
    n_routed=8, n_shared=2, top_k=2, expert_hidden=256, balance="loss_free"
)


class CycleBatcher:
    """Deterministic LEARNABLE stream: token i is followed by (i+1) mod 8.

    The next token depends only on the current one (static token-to-token
    map), so the task is learnable in tens of steps at 0.02-init scales; a
    plain arange-counting stream needs far more steps to converge.
    """

    def __init__(self, batch_size: int, seq_len: int) -> None:
        self.b = batch_size
        self.l = seq_len
        self.cursor = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        span = self.b * (self.l + 1)
        n = np.arange(self.cursor, self.cursor + span)
        chunk = ((n % 8) + 100).reshape(self.b, self.l + 1)
        self.cursor += span
        t = torch.from_numpy(chunk.astype(np.int64))
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.cursor = int(state["cursor"])


def trainer_for(**train_kw):
    torch.manual_seed(11)
    cfg = LydenConfig(
        model=ModelConfig(vocab_size=8192, dim=256, n_layers=4, moe=NANO_MOE)
    )
    for key, value in train_kw.items():
        assert hasattr(cfg.training, key), key
        setattr(cfg.training, key, value)
    model = LydenModel(cfg.model).cuda()
    trainer = Trainer(
        cfg, model, CycleBatcher(4, 64), val_batcher=CycleBatcher(2, 64)
    )
    return trainer


# ---------------------------------------------------------------- groups
@requires_gpu
def test_param_groups_split_lr_and_wd() -> None:
    tr = trainer_for(lr=1e-3, trunk_lr_mult=0.1)
    lrs = {i: (g["initial_lr"], g["weight_decay"]) for i, g in enumerate(tr.opt.param_groups)}
    # decode which names land in each group
    named = dict(tr.model.named_parameters())

    def find_group(param) -> int:
        for i, g in enumerate(tr.opt.param_groups):
            if any(p is param for p in g["params"]):
                return i
        raise AssertionError

    router = next(p for n, p in named.items() if "ffn.router" in n)
    gate = next(p for n, p in named.items() if "mixer.gate" in n)
    shared = next(p for n, p in named.items() if "ffn.shared" in n)
    emb = named["emb.weight"]
    norm1 = next(p for n, p in named.items() if "blocks.0.norm1" in n)
    router_idx, gate_idx, shared_idx = find_group(router), find_group(gate), find_group(shared)
    emb_idx, norm_idx = find_group(emb), find_group(norm1)
    assert lrs[gate_idx][0] == pytest.approx(1e-3) and lrs[shared_idx][0] == pytest.approx(1e-3)
    assert lrs[emb_idx][0] == pytest.approx(1e-4) and lrs[norm_idx][0] == pytest.approx(1e-4)
    assert lrs[router_idx][1] == pytest.approx(0.1)  # 2-D router weight -> decayed
    assert lrs[gate_idx][1] == pytest.approx(0.1)
    assert lrs[shared_idx][1] == pytest.approx(0.1)  # all fast params are 2-D linears
    assert lrs[emb_idx][1] == pytest.approx(0.1)
    assert lrs[norm_idx][1] == pytest.approx(0.0)


@requires_gpu
def test_schedule_warmup_cosine_floor() -> None:
    tr = trainer_for(warmup_steps=200, total_steps=1000)
    lam = tr._lr_lambda
    assert lam(0) == 0.0
    assert lam(100) == pytest.approx(0.5, abs=1e-6)
    assert lam(200) == pytest.approx(1.0, abs=1e-6)
    assert lam(1000) == pytest.approx(0.1, abs=1e-3)
    assert lam(2000) == pytest.approx(0.1, abs=1e-3)
    mid = lam(600)
    assert 0.2 < mid < 1.0


# ------------------------------------------------------------ step/loss
@requires_gpu
def test_step_math_and_fast_lr_runs() -> None:
    tr = trainer_for(total_steps=100, warmup_steps=10)
    info = tr.step()
    assert info["loss"] > 0 and bool(torch.isfinite(torch.tensor(info["loss"])))
    assert info["grad_norm"] >= 0.0
    assert info["tokens_per_s"] > 0
    assert "gate@3" in info["stats"] and "moe_load@3" in info["stats"]
    assert info["stats"]["moe_load@3"].sum().item() == pytest.approx(2.0, abs=1e-4)


@requires_gpu
def test_loss_decreases_on_learnable_stream() -> None:
    tr = trainer_for(amp="fp32", total_steps=80, warmup_steps=8, grad_accum=1)
    losses = [tr.step()["loss"] for _ in range(40)]
    assert losses[0] > losses[-1] + 0.5, f"no learning: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert all(v == v and np.isfinite(v) for v in losses)


@requires_gpu
def test_amp_bf16_step_running() -> None:
    tr = trainer_for(amp="bf16", total_steps=40, warmup_steps=5)
    info = tr.step()
    assert np.isfinite(info["loss"])


# --------------------------------------------------------- checkpoints
@requires_gpu
def test_checkpoint_resume_matches_full_run(tmp_path: Path) -> None:
    tr = trainer_for(amp="fp32", total_steps=100, warmup_steps=10, keep_last=5)
    for _ in range(3):
        tr.step()
    ckpt = tr.save_checkpoint()

    # Everything the checkpoint restores must be bitwise identical
    # (deterministic, from-file comparisons).
    snapshot_params = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
    snapshot_lrs = [g["lr"] for g in tr.opt.param_groups]
    snapshot_betas = [g["betas"] for g in tr.opt.param_groups]
    snapshot_opt = [
        {k: s[k].detach().clone() for k in ("exp_avg", "exp_avg_sq", "step")}
        for s in tr.opt.state.values()
    ]
    snapshot_batch = tr.train_batcher.next_batch()

    # Full-run step 4 (what the resumed trainer's first step must reproduce).
    full_four_loss = tr.step()["loss"]
    full_four = {n: p.detach().clone() for n, p in tr.model.named_parameters()}

    fresh = trainer_for(amp="fp32", total_steps=100, warmup_steps=10)
    fresh.load_checkpoint(ckpt)
    for n, p in fresh.model.named_parameters():
        assert torch.equal(p, snapshot_params[n]), n
    assert fresh.global_step == 3
    for g_new, lr, betas in zip(
        fresh.opt.param_groups, snapshot_lrs, snapshot_betas, strict=True
    ):
        assert g_new["lr"] == lr
        assert g_new["betas"] == betas
    for s_new, s_full in zip(fresh.opt.state.values(), snapshot_opt, strict=True):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(s_new[key].cpu(), s_full[key].cpu()), key
    # The batcher must resume mid-stream: the next batch equals the full
    # run's next batch (catches replay-from-zero bugs).
    xb_res, yb_res = fresh.train_batcher.next_batch()
    assert torch.equal(snapshot_batch[0], xb_res)
    assert torch.equal(snapshot_batch[1], yb_res)
    # One continuation step: losses match within GPU numeric noise (cuBLAS
    # algorithm choice can vary across processes, so no bitwise loss check).
    info = fresh.step()
    assert abs(info["loss"] - full_four_loss) < 1e-3, (info["loss"], full_four_loss)
    for n, p in fresh.model.named_parameters():
        assert torch.allclose(p, full_four[n], atol=1e-4), f"resume diverged at {n}"


@requires_gpu
def test_keep_last_prunes(tmp_path: Path) -> None:
    tr = trainer_for(keep_last=2)
    for _ in range(4):
        tr.step()
        tr.save_checkpoint()
    files = sorted(tr.ckpt_dir.glob("ckpt_*.pt"))
    assert len(files) == 2
    assert int(files[-1].stem.split("_")[1]) == 4


def test_packed_batcher_state_roundtrip() -> None:
    from lyden.data.dataloader import PackedBatcher, SyntheticDataset

    ds = SyntheticDataset(64, seed=3, tokens_total=100_000)
    b1 = PackedBatcher(ds, seq_len=16, batch_size=4, seed=7, shuffle=True)
    for _ in range(3):
        b1.next_batch()
    state = b1.state_dict()
    expect = [b1.next_batch() for _ in range(2)]

    b2 = PackedBatcher(ds, seq_len=16, batch_size=4, seed=7, shuffle=True)
    b2.load_state_dict(state)
    got = [b2.next_batch() for _ in range(2)]
    for (xf, yf), (xr, yr) in zip(expect, got, strict=True):
        assert torch.equal(xf, xr)
        assert torch.equal(yf, yr)


# ------------------------------------------------------------ monitors
@requires_gpu
def test_stability_monitors() -> None:
    tr = trainer_for()
    bad = {
        "moe_load@3": torch.tensor([0.6, 0.2, 0.1, 0.1], device="cuda"),
        "gate@3": torch.tensor([0.95, 0.03, 0.02], device="cuda"),
    }
    warnings = tr.stability_checks(bad, loss=1.0)
    assert any("expert_collapsed@moe_load@3" in w for w in warnings)
    assert any("gate_collapse@gate@3" in w for w in warnings)
    tr.recent_losses = [1.0] * 20
    warnings = tr.stability_checks({}, loss=5.0)
    assert any("loss_spike" in w for w in warnings)
    clean = {
        "moe_load@3": torch.tensor([0.25] * 4, device="cuda"),
        "gate@3": torch.tensor([0.34, 0.33, 0.33], device="cuda"),
    }
    assert tr.stability_checks(clean, loss=1.0) == []


@requires_gpu
def test_model_collect_stats_contract() -> None:
    torch.manual_seed(5)
    cfg = LydenConfig(
        model=ModelConfig(vocab_size=8192, dim=256, n_layers=4, moe=NANO_MOE)
    )
    model = LydenModel(cfg.model).cuda().eval()
    tok = torch.randint(0, 8192, (2, 10), device="cuda")
    with torch.no_grad():
        logits, stats = model(tok, collect_stats=True)
    assert logits.shape == (2, 10, 8192)
    assert stats["gate@3"].shape == (3,)
    assert stats["moe_load@3"].shape == (8,)
    assert torch.allclose(stats["gate@3"].sum(), torch.ones(1, device="cuda"))
    assert torch.allclose(stats["moe_load@3"].sum(), torch.tensor(2.0, device="cuda"))