"""Logging facade: console + optional TensorBoard backend.

Records loss, learning rate, gradient norm, throughput, memory usage, MoE
routing statistics, and attention branch gate values, plus the run config and
environment for provenance.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

from lyden.config import LydenConfig, save_config


class Logger:
    """Unified logger writing to console and (optionally) TensorBoard."""

    def __init__(self, cfg: LydenConfig, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir).expanduser()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.backend: Any = None
        if cfg.training.log_backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.backend = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except ImportError:
                print("[lyden] TensorBoard unavailable; falling back to console", file=sys.stderr)
        save_config(cfg, self.run_dir / "config.yaml")
        self._log_env()

    def log_config(self) -> None:
        """Print the full run config as JSON."""
        print(json.dumps({"config": json.loads(json.dumps(self.cfg, default=str))}, indent=2))

    def _log_env(self) -> None:
        """Record Python/platform and torch/CUDA versions into the run dir."""
        try:
            import torch

            env: dict[str, Any] = {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
            if torch.cuda.is_available():
                env["gpu"] = torch.cuda.get_device_name(0)
                env["vram_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 2**30, 2
                )
        except Exception as exc:  # pragma: no cover - logging must never crash training
            env = {"error": str(exc)}
        with open(self.run_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log a dict of scalar metrics at a given optimizer/global step."""
        parts = " ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items()))
        print(f"[step {step}] {parts}", flush=True)
        if self.backend is not None:
            for key, value in metrics.items():
                self.backend.add_scalar(key, float(value), step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a tensor/list of values as a TensorBoard histogram (console no-op)."""
        if self.backend is not None:
            self.backend.add_histogram(tag, values, step)

    def close(self) -> None:
        """Flush and release the backend."""
        if self.backend is not None:
            self.backend.close()
