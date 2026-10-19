"""Trainer (spec §4): AdamW + split LR, warmup→cosine, AMP, grad accumulation,
checkpoint/resume, logging, stability monitors.

Split-LR grouping (spec §4): parameters whose name matches
``("routed", "shared", "router", "mixer.gate")`` are "fast" components
(experts, routers, attention branch gates) at the base LR; everything else
(embeddings, mamba, attention projections, norms, dense FFN, LM head) trains
at ``trunk_lr_mult`` × base. Weight decay applies to tensors with ndim > 1
only (norms/biases get 0.0).

Precision: bf16 autocast; fp16 via GradScaler on a single device and via a
world-shared Megatron-style loss scale under manual DDP (the synced-gradient
overflow check keeps every rank's skip/step decision identical); fp32 disables
AMP. SSM state stays fp32 inside the mamba kernel.

Stability monitors (train path): expert collapse (any routed-expert load >
50%), attention gate collapse (max gate weight > 0.9), and loss spikes (loss
> 3× the median of the last 20 logged losses). SSM-state-norm growth is an
inference-time monitor and is exercised by the generation loop, not here.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from lyden.config import LydenConfig, to_dict
from lyden.model.model import LydenModel

FAST_MARKERS = ("routed", "shared", "router", "mixer.gate")
_MEAN_KEYS = ("moe_load@", "gate@")


class Trainer:
    """Training loop over a packed batcher; run :meth:`train` for steps."""

    def __init__(
        self,
        cfg: LydenConfig,
        model: LydenModel,
        train_batcher: Any,
        val_batcher: Any | None = None,
        device: str = "cuda",
        dist_rank: int = 0,
        dist_world: int = 1,
        dist_backend: str = "nccl",
    ) -> None:
        if not 0 <= dist_rank < dist_world:
            raise ValueError(f"dist_rank {dist_rank} out of range for dist_world {dist_world}")
        self.cfg = cfg
        self.model = model
        self.train_batcher = train_batcher
        self.val_batcher = val_batcher
        self.device = device
        self.dist_rank = int(dist_rank)
        self.dist_world = int(dist_world)
        self.dist_backend = dist_backend
        self.amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[cfg.training.amp]
        self.use_amp = self.amp_dtype != torch.float32
        # Under manual DDP the loss scale must be a world-shared value: a
        # per-rank GradScaler records found_inf locally, so one rank could
        # step while the other skips (diverged weights) and scale updates
        # would diverge too. The decision is made on the SYNCED gradients
        # instead (see _dist_clip_and_step).
        self.use_dist_scale = self.dist_world > 1 and cfg.training.amp == "fp16"
        self.dist_scale = 65536.0
        self._scale_growth_tracker = 0
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=cfg.training.amp == "fp16" and not self.use_dist_scale
        )
        self.total_steps = cfg.training.total_steps
        self.global_step = 0
        self.recent_losses: list[float] = []
        self.out_dir = Path(cfg.training.out_dir) / cfg.run_name
        self.ckpt_dir = self.out_dir / "ckpts"
        self.opt = self._build_optimizer()
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_lambda)

    # ------------------------------------------------------------- schedule
    def _lr_lambda(self, step: int) -> float:
        warmup = self.cfg.training.warmup_steps
        floor = 0.1
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, self.total_steps - warmup)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    def _build_optimizer(self) -> torch.optim.AdamW:
        lr = self.cfg.training.lr
        trunk_lr = lr * self.cfg.training.trunk_lr_mult
        wd = self.cfg.training.weight_decay
        groups: dict[str, list[nn.Parameter]] = {
            "fast_decay": [],
            "fast_nowd": [],
            "trunk_decay": [],
            "trunk_nowd": [],
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_fast = any(marker in name for marker in FAST_MARKERS)
            no_decay = param.ndim <= 1
            key = ("fast" if is_fast else "trunk") + ("_nowd" if no_decay else "_decay")
            groups[key].append(param)
        if not groups["fast_decay"] and not groups["fast_nowd"]:
            raise RuntimeError("optimizer param groups: no fast (spec §4) params found")
        if not groups["trunk_decay"] and not groups["trunk_nowd"]:
            raise RuntimeError("optimizer param groups: no trunk params found")
        group_specs = [
            ("fast_decay", lr, wd),
            ("fast_nowd", lr, 0.0),
            ("trunk_decay", trunk_lr, wd),
            ("trunk_nowd", trunk_lr, 0.0),
        ]
        self.param_groups_meta = group_specs
        return torch.optim.AdamW(
            [
                {"params": groups[key], "lr": group_lr, "weight_decay": gwd}
                for key, group_lr, gwd in group_specs
            ],
            betas=self.cfg.training.betas,
        )

    # -------------------------------------------------------------- losses
    def _forward_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            logits, stats = self.model(
                x, collect_stats=True, grad_ckpt=self.cfg.training.activation_checkpoint
            )
            loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), y.view(-1)
            )
        return loss, stats

    @torch.no_grad()
    def micro_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Standalone eval loss for a batch (autocast applied, no stats)."""
        x, y = x.to(self.device), y.to(self.device)
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), y.view(-1)
            )
        return float(loss)

    def step(self) -> dict[str, Any]:
        """One optimizer step: grad_accum micro-batches → clip → AdamW step."""
        accum = self.cfg.training.grad_accum
        self.opt.zero_grad(set_to_none=True)
        agg: dict[str, Any] = {}
        total_loss = 0.0
        tokens = 0
        t0 = time.perf_counter()
        for _ in range(accum):
            x, y = self.train_batcher.next_batch()
            loss, stats = self._forward_loss(x, y)
            raw = loss / accum
            if self.scaler.is_enabled():
                raw = self.scaler.scale(raw)
            elif self.use_dist_scale:
                raw = raw * self.dist_scale
            raw.backward()
            total_loss += float(loss.detach())
            tokens += x.numel()
            merge_stats(agg, stats, scale=1.0 / accum)
        grad_norm = self._clip_and_step()
        self.sched.step()
        self.global_step += 1
        self._update_moe_bias(agg)
        elapsed = time.perf_counter() - t0
        mean_loss = total_loss / accum
        self.recent_losses.append(mean_loss)
        if len(self.recent_losses) > 40:
            self.recent_losses = self.recent_losses[-40:]
        warnings = self.stability_checks(agg, mean_loss)
        return {
            "step": self.global_step,
            "loss": mean_loss,
            "lr": next(iter(self.opt.param_groups))["lr"],
            "grad_norm": grad_norm,
            "tokens_per_s": tokens / max(elapsed, 1e-9),
            "vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "stats": agg,
            "warnings": warnings,
        }

    def _update_moe_bias(self, agg: dict[str, Any]) -> None:
        """M11: run the loss-free balancer once per optimizer step.

        M10 root cause of the load imbalance: the balancer's call site simply
        did not exist — stats were collected but ``update_bias`` never ran, so
        the bias stayed zero for the whole run. Loads are averaged across
        micro-batches (merge_stats) and, when distributed, across ranks
        (world-mean) so every rank applies the identical delta.
        """
        if self.cfg.model.moe.balance != "loss_free":
            return
        from lyden.model.moe import MoE

        for i, blk in enumerate(self.model.blocks):
            if blk.kind != "a":
                continue
            load = agg.get(f"moe_load@{i}")
            if load is None:
                continue
            if self.dist_world > 1 and torch.distributed.is_initialized():
                load = load.clone()
                torch.distributed.all_reduce(load, op=torch.distributed.ReduceOp.SUM)
                load = load / self.dist_world
            ffn = blk.ffn
            if not isinstance(ffn, MoE):
                continue
            ffn.update_bias(load=load, step=self.global_step)

    def _sync_grads(self) -> torch.Tensor:
        """Zero-fill missing grads, flatten, one all-reduce (SUM) — returns the flat buffer.

        Zero-filling before flattening makes the collective sequence
        unreachable-by-desync across ranks: MoE routing selects expert subsets
        *per rank*, so a rank that never used an expert has ``grad is None``
        there; with the previous per-param all-reduce the ranks skipped
        different params and their NCCL queues misaligned (the M10 Kaggle
        failure: a 512- and a 262144-element collective at one sequence id).
        """
        if self.dist_world <= 1 or not torch.distributed.is_initialized():
            raise RuntimeError("_sync_grads requires an initialized process group")
        params = [p for p in self.model.parameters() if p.requires_grad]
        for param in params:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
        flat = torch._utils._flatten_dense_tensors([p.grad for p in params])  # type: ignore[attr-defined]
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        return flat

    def _dist_clip_and_step(self) -> float:
        """World-consistent clip+step: average, shared fp16 scale, clip, step.

        The fp16 overflow decision is taken on the SYNCED buffer (a rank-
        invariant result), so every rank reaches the identical skip/step
        decision and keeps the same scale value — GradScaler's per-rank
        found_inf would let one rank step while the other skips.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        flat = self._sync_grads()
        flat.div_(self.dist_world)
        if self.use_dist_scale:
            healthy = torch.isfinite(flat).all().to(torch.float32)
            torch.distributed.all_reduce(healthy, op=torch.distributed.ReduceOp.MIN)
            if float(healthy) < 1.0:
                # Overflow: identical world decision — drop this step's grads
                # and back the shared scale off. The next step zero-fills.
                self.dist_scale = max(self.dist_scale * 0.5, 1.0)
                self._scale_growth_tracker = 0
                for param in params:
                    param.grad = None
                return float("nan")
            flat.div_(self.dist_scale)
            self._scale_growth_tracker += 1
            if self._scale_growth_tracker >= 2000 and self.dist_scale < 2.0**24:
                self.dist_scale *= 2.0
                self._scale_growth_tracker = 0
        grads = torch._utils._unflatten_dense_tensors(flat, [p.grad for p in params])  # type: ignore[attr-defined]
        for param, grad in zip(params, grads, strict=True):
            assert param.grad is not None  # zero-filled in _sync_grads
            param.grad.copy_(grad)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.cfg.training.grad_clip))
        self.opt.step()
        for param in params:
            param.grad = None
        return grad_norm

    def _clip_and_step(self) -> float:
        if self.dist_world > 1 and torch.distributed.is_initialized():
            return self._dist_clip_and_step()
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.opt)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.training.grad_clip,
            )
        )
        self.scaler.step(self.opt)
        self.scaler.update()
        return grad_norm

    # ------------------------------------------------------------ monitors
    def stability_checks(self, stats: dict[str, Any], loss: float | None = None) -> list[str]:
        """Pure monitor: warning strings for anomalies in the step's stats."""
        warnings: list[str] = []
        for key, value in stats.items():
            if key.startswith("moe_load@") and value.numel() and float(value.max()) > 0.5:
                warnings.append(f"expert_collapsed@{key}: max load {float(value.max()):.2f} > 0.5")
            if key.startswith("gate@") and value.numel() and float(value.max()) > 0.9:
                warnings.append(f"gate_collapse@{key}: gate {float(value.max()):.2f} > 0.9")
        if loss is not None and len(self.recent_losses) >= 20:
            med = float(np.median(self.recent_losses[:-1]))
            if med > 0 and loss > 3.0 * med:
                warnings.append(f"loss_spike {loss:.3f} > 3x median {med:.3f}")
        return warnings

    # ---------------------------------------------------------------- val
    @torch.no_grad()
    def validate(self, n_batches: int = 50) -> float:
        assert self.val_batcher is not None, "validate() needs a val_batcher"
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        for _ in range(n_batches):
            x, y = self.val_batcher.next_batch()
            losses.append(self.micro_loss(x, y))
        if was_training:
            self.model.train()
        return float(sum(losses) / len(losses))

    # -------------------------------------------------------- checkpoints
    def save_checkpoint(self) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"ckpt_{self.global_step:07d}.pt"
        payload = {
            "step": self.global_step,
            "total_steps": self.total_steps,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        if self.use_dist_scale:
            payload["loss_scale"] = self.dist_scale
            payload["scale_growth"] = self._scale_growth_tracker
        config_yaml = yaml.safe_dump(to_dict(self.cfg), default_flow_style=False, sort_keys=False)
        payload["config_yaml"] = config_yaml
        if hasattr(self.train_batcher, "state_dict"):
            if self.dist_world <= 1:
                payload["batcher"] = self.train_batcher.state_dict()
            else:
                payload[f"batcher_rank{self.dist_rank}"] = self.train_batcher.state_dict()
        torch.save(payload, path)
        self._prune_keep_last()
        return path

    def _prune_keep_last(self) -> None:
        keep = self.cfg.training.keep_last
        ckpts = sorted(self.ckpt_dir.glob("ckpt_*.pt"))
        for old in ckpts[:-keep]:
            old.unlink()

    def load_checkpoint(self, path: Path, *, reset_flags: bool | None = None) -> None:
        """Resume from ``path``. ``reset_flags`` overrides the config's
        reset_scheduler/reset_moe_bias: None = config (explicit cross-phase
        resume), False = auto-resume of this trainer's own phase (a mid-run
        crash must NOT restart the schedule it already started)."""
        reset_sched = self.cfg.training.reset_scheduler if reset_flags is None else reset_flags
        reset_bias = self.cfg.training.reset_moe_bias if reset_flags is None else reset_flags
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if reset_bias:
            # M11: start balancing from a clean slate instead of continuing
            # from biases locked into a collapsed solution.
            from lyden.model.moe import MoE

            for module in self.model.modules():
                if isinstance(module, MoE):
                    module.reset_bias()
        self.opt.load_state_dict(payload["optimizer"])
        if reset_sched:
            # M11: continue-train phase — start the schedule at step 0 so the
            # new warmup/cosine applies over the new total_steps. The lambda
            # reads the scheduler's own epoch, leaving ``global_step`` free
            # to keep its checkpoint value (for naming/logging continuity).
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_lambda)
        else:
            self.sched.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        if self.use_dist_scale:
            if "loss_scale" not in payload:
                # Pre-dist-scale checkpoint: GradScaler used the same default.
                print("[resume] no loss_scale in checkpoint; starting at 65536.0")
            else:
                self.dist_scale = float(payload["loss_scale"])
                self._scale_growth_tracker = int(payload.get("scale_growth", 0))
        self.global_step = int(payload["step"])
        if hasattr(self.train_batcher, "load_state_dict"):
            key = "batcher" if self.dist_world <= 1 else f"batcher_rank{self.dist_rank}"
            if key not in payload:
                if self.dist_world <= 1 and "batcher" in payload:
                    self.train_batcher.load_state_dict(payload["batcher"])
                else:
                    # Resuming a DDP run from a single-GPU checkpoint: loading
                    # that shared sampler state would make ranks sample
                    # identical windows, so each rank keeps its own fresh
                    # (rank-shifted) stream position instead.
                    print(
                        f"[resume] no batcher_rank{self.dist_rank} in checkpoint; "
                        "each rank keeps its own sampler state"
                    )
                return
            self.train_batcher.load_state_dict(payload[key])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in payload["rng"]["cuda"]])
        np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])


def merge_stats(agg: dict[str, Any], new: dict[str, Any], scale: float = 1.0) -> None:
    """Running merge across micro-batches; mean-keys are scaled to means."""
    for key, value in new.items():
        if isinstance(value, torch.Tensor):
            scaled = value.detach().float() * scale
            if key not in agg:
                agg[key] = scaled
            else:
                try:
                    agg[key] = agg[key] + scaled
                except RuntimeError:
                    agg[key] = scaled
        else:
            agg[key] = value
