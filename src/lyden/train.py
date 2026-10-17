"""Trainer (spec §4): AdamW + split LR, warmup→cosine, AMP, grad accumulation,
checkpoint/resume, logging, stability monitors.

Split-LR grouping (spec §4): parameters whose name matches
``("routed", "shared", "router", "mixer.gate")`` are "fast" components
(experts, routers, attention branch gates) at the base LR; everything else
(embeddings, mamba, attention projections, norms, dense FFN, LM head) trains
at ``trunk_lr_mult`` × base. Weight decay applies to tensors with ndim > 1
only (norms/biases get 0.0).

Precision: bf16 autocast (fp16 + GradScaler fallback; fp32 disables AMP).
SSM state stays fp32 inside the mamba kernel.

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
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_batcher = train_batcher
        self.val_batcher = val_batcher
        self.device = device
        self.amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[cfg.training.amp]
        self.use_amp = self.amp_dtype != torch.float32
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.amp == "fp16")
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
            logits, stats = self.model(x, collect_stats=True)
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
            (loss / accum).backward()
            total_loss += float(loss)
            tokens += x.numel()
            merge_stats(agg, stats, scale=1.0 / accum)
        grad_norm = self._clip_and_step()
        self.sched.step()
        self.global_step += 1
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

    def _clip_and_step(self) -> float:
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
        config_yaml = yaml.safe_dump(to_dict(self.cfg), default_flow_style=False, sort_keys=False)
        payload["config_yaml"] = config_yaml
        if hasattr(self.train_batcher, "state_dict"):
            payload["batcher"] = self.train_batcher.state_dict()
        torch.save(payload, path)
        self._prune_keep_last()
        return path

    def _prune_keep_last(self) -> None:
        keep = self.cfg.training.keep_last
        ckpts = sorted(self.ckpt_dir.glob("ckpt_*.pt"))
        for old in ckpts[:-keep]:
            old.unlink()

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.opt.load_state_dict(payload["optimizer"])
        self.sched.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.global_step = int(payload["step"])
        if "batcher" in payload and hasattr(self.train_batcher, "load_state_dict"):
            self.train_batcher.load_state_dict(payload["batcher"])
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
