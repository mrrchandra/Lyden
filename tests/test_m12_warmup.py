"""M12 warmup config generator: 1024 -> 32K context warming phases."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "scripts")

import m12_warmup_configs as m12  # noqa: E402
from lyden.config import load_config  # noqa: E402

build_phase_configs = m12.build_phase_configs
phase_edges = m12.phase_edges
WINDOWS_AND_STEPS = m12.WINDOWS_AND_STEPS

ENTRY_STEP = 9600


def test_five_phases_with_planned_windows_and_steps() -> None:
    assert WINDOWS_AND_STEPS == [
        (2048, 750),
        (4096, 450),
        (8192, 220),
        (16384, 70),
        (32768, 130),
    ]
    configs = build_phase_configs()
    assert [c["training"]["seq_len"] for c in configs] == [w for w, _ in WINDOWS_AND_STEPS]
    assert [c["training"]["total_steps"] for c in configs] == [s for _, s in WINDOWS_AND_STEPS]


def test_absolute_bound_steps_survive_any_resume() -> None:
    """Crash-resume regression: own-phase resume can't recompute the relative
    entry, so each phase config pins its ABSOLUTE edge (training.bound_step)."""
    edges = phase_edges(entry_step=ENTRY_STEP)
    for cfg, edge in zip(build_phase_configs(ENTRY_STEP), edges, strict=False):
        assert cfg["training"]["bound_step"] == edge


def test_effective_batch_tokens_within_budget() -> None:
    """Per-rank optimizer tokens (batch * seq * accum) must stay in 16K..524K."""
    for cfg in build_phase_configs():
        t = cfg["training"]
        per_rank = t["batch_size"] * t["seq_len"] * t["grad_accum"]
        assert 16_384 <= per_rank <= 524_288, per_rank


def test_phase_edges_chain_from_entry() -> None:
    assert phase_edges(entry_step=ENTRY_STEP) == [10350, 10800, 11020, 11090, 11220]


def test_reset_scheduler_and_moe_fixes_kept() -> None:
    for cfg in build_phase_configs():
        assert cfg["training"]["reset_scheduler"] is True
        assert cfg["training"]["reset_moe_bias"] is True
        assert cfg["model"]["moe"]["router_znorm"] is True
        assert list(cfg["model"]["moe"]["bias_gamma_ramp"]) == [0.004, 0.001, 400]


def test_per_phase_run_names_and_data_dirs() -> None:
    configs = build_phase_configs()
    assert [c["run_name"] for c in configs] == [
        "100m-m12-a",
        "100m-m12-b",
        "100m-m12-c",
        "100m-m12-d",
        "100m-m12-e",
    ]
    for cfg in configs:
        assert "m12" in cfg["data"]["data_dir"]


def test_written_yaml_round_trips_through_load_config(tmp_path: Path) -> None:
    m12.write_phase_configs(out_dir=tmp_path)
    path = tmp_path / "m12-phase-a.yaml"
    cfg = load_config(path)
    assert cfg.training.seq_len == 2048
    assert cfg.training.total_steps == 750
    assert cfg.training.bound_step == 10350
    assert cfg.run_name == "100m-m12-a"


def _unused(_x: object) -> None:  # pragma: no cover
    pass
