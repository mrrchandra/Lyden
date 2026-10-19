"""Configuration system for Lyden.

All hyperparameters live here as nested dataclasses, loaded from YAML files.
Nothing in the codebase hardcodes hyperparameters; everything is config-driven.
Unknown keys in a YAML file are an error (catches typos early).
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MambaConfig:
    """Selective SSM (Mamba-2) block hyperparameters."""

    d_state: int = 128
    d_conv: int = 4
    expand: int = 2


@dataclass
class AttentionConfig:
    """Hybrid attention block hyperparameters.

    The block runs three parallel branches (sliding window, compressed global,
    selective top-k) over a shared GQA K/V projection, combined by a learned gate.
    """

    n_heads: int = 8
    n_kv_heads: int = 2
    window: int = 512
    n_summary: int = 64
    topk: int = 256
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )


@dataclass
class DenseFFNConfig:
    """Dense feed-forward network used after Mamba blocks."""

    hidden: int = 2048
    activation: str = "swiglu"  # "swiglu" or "gelu"

    def __post_init__(self) -> None:
        if self.activation not in ("swiglu", "gelu"):
            raise ValueError(f"activation must be 'swiglu' or 'gelu', got '{self.activation}'")


@dataclass
class MoEConfig:
    """Mixture-of-experts feed-forward used after attention blocks."""

    n_routed: int = 32
    n_shared: int = 2
    top_k: int = 2
    expert_hidden: int = 1024
    balance: str = "loss_free"  # "loss_free" (DeepSeek-style bias) or "aux" (Switch-style)
    aux_loss_weight: float = 0.01
    bias_gamma: float = 0.001  # loss-free bias adjustment step size
    bias_gamma_ramp: tuple[float, float, int] | None = None
    # (gamma_start, gamma_end, ramp_steps): linear gamma decay over the ramp;
    # None = constant ``bias_gamma`` (M5..M10 semantics).
    router_znorm: bool = False  # per-token standardize router scores (M11 fix)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.top_k > self.n_routed:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed n_routed ({self.n_routed})")
        if self.n_routed > 64:
            raise ValueError(
                f"n_routed ({self.n_routed}) exceeds the 64-expert scale ceiling from the spec"
            )
        if self.balance not in ("loss_free", "aux"):
            raise ValueError(f"balance must be 'loss_free' or 'aux', got '{self.balance}'")
        if self.aux_loss_weight < 0 or self.bias_gamma <= 0:
            raise ValueError(
                f"aux_loss_weight ({self.aux_loss_weight}) must be >= 0 and "
                f"bias_gamma ({self.bias_gamma}) must be > 0"
            )
        if self.bias_gamma_ramp is not None:
            if len(self.bias_gamma_ramp) != 3:
                raise ValueError("bias_gamma_ramp must be (gamma_start, gamma_end, ramp_steps)")
            gamma_start, gamma_end, ramp_steps = self.bias_gamma_ramp
            if gamma_start <= 0 or gamma_end <= 0 or ramp_steps < 1:
                raise ValueError(
                    f"bias_gamma_ramp values invalid: {self.bias_gamma_ramp} "
                    "(gammas must be > 0 and ramp_steps >= 1)"
                )


@dataclass
class ModelConfig:
    """Model architecture configuration.

    ``layer_pattern`` is a string over {'m', 'a'} repeated to build the stack:
    'm' = Mamba block + dense FFN, 'a' = attention block + MoE FFN.
    """

    vocab_size: int = 32768
    dim: int = 768
    n_layers: int = 24
    layer_pattern: str = "mmma"
    mamba: MambaConfig = field(default_factory=MambaConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    dense_ffn: DenseFFNConfig = field(default_factory=DenseFFNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    dropout: float = 0.0
    weight_tying: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.vocab_size <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size, dim and n_layers must be positive")
        if not self.layer_pattern or set(self.layer_pattern) - {"m", "a"}:
            raise ValueError(
                f"layer_pattern must be a non-empty string over {{'m', 'a'}}, "
                f"got '{self.layer_pattern}'"
            )
        if self.n_layers % len(self.layer_pattern) != 0:
            raise ValueError(
                f"n_layers ({self.n_layers}) must be a multiple of "
                f"len(layer_pattern) ({len(self.layer_pattern)})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def layer_kinds(self) -> list[str]:
        """Expand the repeating pattern into per-layer kinds ('m' or 'a')."""
        repeats = self.n_layers // len(self.layer_pattern)
        return list(self.layer_pattern) * repeats


@dataclass
class TrainingConfig:
    """Training loop hyperparameters.

    ``lr`` is the learning rate for fast-learning components (experts, routers,
    branch gates). The trunk (embeddings, Mamba, attention projections, norms,
    LM head) is trained at ``lr * trunk_lr_mult``.
    """

    lr: float = 3.0e-4
    trunk_lr_mult: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 100_000
    grad_accum: int = 4
    amp: str = "bf16"  # "bf16", "fp16", or "fp32"
    seq_len: int = 2048
    batch_size: int = 8
    activation_checkpoint: bool = False
    checkpoint_every: int = 5000
    keep_last: int = 3
    log_every: int = 10
    log_backend: str = "tensorboard"  # "tensorboard" or "console"
    out_dir: str = "runs"
    reset_scheduler: bool = False  # on resume: start a fresh lambda phase (M11)
    reset_moe_bias: bool = False  # on resume: zero the loss-free bias buffers (M11)

    def __post_init__(self) -> None:
        if self.trunk_lr_mult <= 0 or self.trunk_lr_mult > 1:
            raise ValueError(f"trunk_lr_mult must be in (0, 1], got {self.trunk_lr_mult}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if len(self.betas) != 2:
            raise ValueError(f"betas must be a pair, got {self.betas}")
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1, got {self.grad_accum}")
        if self.amp not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"amp must be 'bf16', 'fp16' or 'fp32', got '{self.amp}'")
        if self.log_backend not in ("tensorboard", "console"):
            raise ValueError(
                f"log_backend must be 'tensorboard' or 'console', got '{self.log_backend}'"
            )
        if self.keep_last < 1:
            raise ValueError(f"keep_last must be >= 1, got {self.keep_last}")


@dataclass
class DataConfig:
    """Dataset and tokenization paths."""

    dataset: str = "tinystories"  # "tinystories" or "fineweb_mix"
    data_dir: str = "~/lyden-data"
    tokenizer_dir: str = "~/lyden-data/tokenizer"
    val_fraction: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")


@dataclass
class ClmConfig:
    """Editable context memory (CLM) configuration.

    The buffer lives in RAM. The model emits textual edit commands between
    ``edit_start`` and ``edit_end`` markers, which the runtime applies to the
    buffer between ``context_start`` and ``context_end`` markers in the prompt.
    """

    buffer_max_chars: int = 20_000
    edit_start: str = "<|edit|>"
    edit_end: str = "<|/edit|>"
    context_start: str = "<|context|>"
    context_end: str = "<|/context|>"

    def __post_init__(self) -> None:
        markers = (self.edit_start, self.edit_end, self.context_start, self.context_end)
        if len(set(markers)) != 4:
            raise ValueError("edit/context marker strings must be distinct")


@dataclass
class LydenConfig:
    """Root configuration for a Lyden training/inference run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    clm: ClmConfig = field(default_factory=ClmConfig)
    seed: int = 1337
    run_name: str = "lyden-run"

    def layer_kinds(self) -> list[str]:
        """Convenience passthrough to the model config."""
        return self.model.layer_kinds()


def _tuple_hint(hint: Any) -> Any:
    """Return the tuple subclass of a (possibly Optional) type hint, else None.

    YAML flows are lists, so fields like ``tuple[float, float, int] | None``
    (M11 ``bias_gamma_ramp``) need the list→tuple conversion too.
    """
    if typing.get_origin(hint) is tuple:
        return hint
    return next(
        (
            arg
            for arg in typing.get_args(hint)
            if arg is not type(None) and typing.get_origin(arg) is tuple
        ),
        None,
    )


def _build(cls: type, payload: Mapping[str, Any]) -> Any:
    """Build a dataclass instance from a plain mapping, recursing into nested dataclasses.

    Unknown keys raise ``ValueError`` so typos in YAML configs fail loudly.
    Stringified type annotations are resolved via ``typing.get_type_hints``.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"Cannot construct non-dataclass {cls!r}")
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in field_names:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        resolved = hints.get(key, Any)
        is_cfg_cls = isinstance(resolved, type) and dataclasses.is_dataclass(resolved)
        if isinstance(value, Mapping) and is_cfg_cls:
            value = _build(resolved, value)
        elif isinstance(value, list) and _tuple_hint(resolved) is not None:
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def from_dict(cls: type, payload: Mapping[str, Any]) -> Any:
    """Build a config dataclass (recursively) from a plain mapping."""
    return _build(cls, payload)


def load_config(path: str | Path) -> LydenConfig:
    """Load a :class:`LydenConfig` from a YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On unknown keys or invalid values.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return from_dict(LydenConfig, payload)


def to_dict(cfg: LydenConfig) -> dict[str, Any]:
    """Serialize a :class:`LydenConfig` to a plain dict (YAML-compatible)."""
    return dataclasses.asdict(cfg)


def save_config(cfg: LydenConfig, path: str | Path) -> None:
    """Write a config to YAML (used for checkpoint provenance)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
