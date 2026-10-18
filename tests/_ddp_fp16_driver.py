"""torchrun-launched driver: two ranks must make identical fp16 scale decisions.

Launched by test_ddp.py::test_two_rank_fp16_scale_lockstep (subprocess via
``python -m torch.distributed.run --standalone --nproc-per-node=2``). Checks:

A) one rank holds a non-finite gradient -> BOTH ranks skip the optimizer
   step, the shared scale backs off 65536 -> 32768, params are untouched and
   grads are dropped (GradScaler's per-rank found_inf would have let one
   rank step while the other skipped, silently diverging the weights);
B) clean gradients -> identical clip+step on every rank (weights stay
   bit-identical across ranks after the joint AdamW step);
C) a full Trainer.step() runs on the dist path: zero-filled missing grads +
   one flat all-reduce + world-invariant overflow decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from lyden.config import LydenConfig, ModelConfig, MoEConfig  # noqa: E402
from lyden.model.model import LydenModel  # noqa: E402
from lyden.train import Trainer  # noqa: E402
from test_train import CycleBatcher  # noqa: E402


def param_checksum(params: list[torch.nn.Parameter]) -> torch.Tensor:
    """FA bijective-enough vector of per-param stats; equal across ranks iff params equal."""
    sums = [(p.detach().double().sum() + (p.detach().double() * p.detach()).sum()) for p in params]
    return torch.stack(sums)


def cross_rank_equal(tensor: torch.Tensor) -> bool:
    lo = tensor.clone()
    torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN)
    hi = tensor.clone()
    torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX)
    return bool(torch.equal(lo, hi))


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(123)
    cfg = LydenConfig(
        model=ModelConfig(
            vocab_size=8192,
            dim=256,
            n_layers=4,
            moe=MoEConfig(n_routed=8, n_shared=2, top_k=2, expert_hidden=256, balance="loss_free"),
        )
    )
    cfg.training.amp = "fp16"
    cfg.training.grad_accum = 1
    cfg.training.total_steps = 40
    cfg.training.warmup_steps = 4
    model = LydenModel(cfg.model).cuda()
    trainer = Trainer(
        cfg,
        model,
        CycleBatcher(4, 64),
        val_batcher=CycleBatcher(2, 64),
        device="cuda",
        dist_rank=rank,
        dist_world=2,
        dist_backend="gloo",
    )
    assert trainer.use_dist_scale, "dist+fp16 must use the shared scale"
    assert not trainer.scaler.is_enabled(), "per-rank GradScaler must be off under DDP"
    assert trainer.dist_scale == 65536.0

    params = [p for p in model.parameters() if p.requires_grad]
    snapshot = [p.detach().clone() for p in params]

    # ---- A: rank-side overflow -> identical world skip + scale back-off
    for param in params:
        param.grad = torch.full_like(param, 0.25)
    if rank == 1:
        params[0].grad[0] = float("inf")
    grad_norm = trainer._dist_clip_and_step()
    assert np.isnan(grad_norm), f"rank{rank}: expected skipped-step NaN, got {grad_norm}"
    assert trainer.dist_scale == 32768.0, f"rank{rank}: scale {trainer.dist_scale}"
    assert trainer._scale_growth_tracker == 0
    for param, snap in zip(params, snapshot, strict=True):
        assert torch.equal(param, snap), f"rank{rank}: optimizer stepped during overflow skip"
        assert param.grad is None, f"rank{rank}: grads must be dropped on skip"

    # ---- B: clean grads -> identical joint step, weights bit-equal across ranks
    for param in params:
        param.grad = torch.full_like(param, 1.0 + 0.5 * rank)
    grad_norm = trainer._dist_clip_and_step()
    assert np.isfinite(grad_norm) and grad_norm >= 0.0
    assert trainer.dist_scale == 32768.0
    assert trainer._scale_growth_tracker == 1
    assert cross_rank_equal(
        param_checksum(params)
    ), f"rank{rank}: weights diverged from the other rank after the joint step"

    # ---- C: full step() on the dist path (MoE experts may leave grad=None;
    # the flat zero-filled sync must keep the collectives aligned)
    info = trainer.step()
    assert np.isfinite(info["loss"]), f"rank{rank}: loss {info['loss']}"
    info = trainer.step()
    assert np.isfinite(info["loss"])
    assert cross_rank_equal(
        param_checksum(params)
    ), f"rank{rank}: weights diverged after full steps"

    if rank == 0:
        print("[fp16-scale-lockstep] ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
