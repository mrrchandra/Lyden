"""Tests for the config system: defaults, YAML round-trip, validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyden.config import (
    AttentionConfig,
    LydenConfig,
    ModelConfig,
    from_dict,
    load_config,
    save_config,
    to_dict,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_construct() -> None:
    cfg = LydenConfig()
    assert cfg.model.layer_kinds() == ["m", "m", "m", "a"] * 6
    assert cfg.model.attention.n_heads == 8
    assert cfg.training.trunk_lr_mult == 0.1


def test_layer_kinds_expansion() -> None:
    cfg = ModelConfig(n_layers=9, layer_pattern="mma")
    assert cfg.layer_kinds() == ["m", "m", "a"] * 3


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        from_dict(LydenConfig, {"model": {"nonexistent": 1}})


def test_nested_dict_construction() -> None:
    cfg = from_dict(
        LydenConfig,
        {
            "model": {"dim": 256, "n_layers": 4, "attention": {"n_summary": 8}},
            "seed": 7,
        },
    )
    assert cfg.model.dim == 256
    assert cfg.model.attention.n_summary == 8
    assert cfg.model.moe.n_routed == 32  # untouched fields keep defaults
    assert cfg.seed == 7


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = from_dict(
        LydenConfig,
        {
            "model": {"dim": 256, "n_layers": 4, "moe": {"n_routed": 8}},
            "training": {"betas": [0.9, 0.999]},
            "seed": 42,
        },
    )
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert to_dict(loaded) == to_dict(cfg)
    assert loaded.training.betas == (0.9, 0.999)


def test_load_shipped_configs() -> None:
    for name in ("nano", "100m", "250m", "500m"):
        cfg = load_config(CONFIG_DIR / f"{name}.yaml")
        kinds = cfg.layer_kinds()
        assert kinds.count("m") == 3 * kinds.count("a")
        assert cfg.model.n_layers == len(kinds)


def test_betas_list_to_tuple() -> None:
    cfg = from_dict(LydenConfig, {"training": {"betas": [0.8, 0.99]}})
    assert cfg.training.betas == (0.8, 0.99)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_layers", 7),  # not a multiple of pattern length
        ("layer_pattern", "mx"),
        ("dropout", 1.0),
        ("dim", 0),
    ],
)
def test_model_validation_errors(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**{field: value})


def test_attention_gqa_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        AttentionConfig(n_heads=7, n_kv_heads=2)
