"""Distributed training tests (M10): per-rank batcher independence, rank-keyed
batcher checkpoint state, and a 2-rank end-to-end torchrun smoke.

The smoke launches scripts/train.py via ``torch.distributed.run`` with two
ranks on ONE cuda device (the *b848b7* model is tiny) and the gloo backend —
validates env detection, per-rank batchers, gradient averaging, rank-0-only
checkpointing with rank-keyed batcher state, and clean teardown.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from lyden.data.dataloader import PackedBatcher, SyntheticDataset
from test_train import trainer_for

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------- batcher over ranks
def test_packed_batcher_train_ranks_are_independent() -> None:
    ds = SyntheticDataset(64, seed=3, tokens_total=200_000)
    b0 = PackedBatcher(ds, seq_len=16, batch_size=4, seed=7, shuffle=True, rank=0, world_size=2)
    b1 = PackedBatcher(ds, seq_len=16, batch_size=4, seed=7, shuffle=True, rank=1, world_size=2)
    x0, _ = b0.next_batch()
    x1, _ = b1.next_batch()
    assert not torch.equal(x0, x1), "ranks must not sample the same window"
    assert not torch.equal(x0[0][:8], x1[0][:8])


def test_packed_batcher_val_ranks_cover_disjoint_spans() -> None:
    ds = SyntheticDataset(64, seed=3, tokens_total=200_000)
    span = 4 * 17
    b0 = PackedBatcher(ds, seq_len=16, batch_size=4, shuffle=False, rank=0, world_size=2)
    b1 = PackedBatcher(ds, seq_len=16, batch_size=4, shuffle=False, rank=1, world_size=2)
    assert b0.state_dict()["sequential_offset"] == 0
    assert b1.state_dict()["sequential_offset"] == span
    xa, _ = b0.next_batch()
    xb, _ = b1.next_batch()
    # rank strides by world_size: after one call rank0 sits at 2*span, rank1 at 3*span
    assert b0.state_dict()["sequential_offset"] == 2 * span
    assert b1.state_dict()["sequential_offset"] == 3 * span
    assert not torch.equal(xa, xb)
    # Footprint check: rank0's second window starts exactly at 2*span of the stream
    assert b0.next_batch()[0][0, 0].item() == int(ds.read(2 * span, 1)[0])
    assert b1.next_batch()[0][0, 0].item() == int(ds.read(3 * span, 1)[0])


def test_packed_batcher_rank_world_validation() -> None:
    ds = SyntheticDataset(64, seed=3, tokens_total=200_000)
    try:
        PackedBatcher(ds, seq_len=16, batch_size=4, rank=2, world_size=2)
    except ValueError as exc:
        assert "rank 2 out of range" in str(exc)
    else:
        raise AssertionError("rank >= world_size must fail loudly")


# ------------------------------------------- rank-keyed batcher checkpoint state
def test_trainer_writes_and_resumes_rank_keyed_batcher_state() -> None:
    tr = trainer_for(total_steps=100, warmup_steps=10, keep_last=3)
    tr.dist_rank, tr.dist_world = 1, 2
    info = tr.step()
    assert np.isfinite(info["loss"])
    ckpt = tr.save_checkpoint()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "batcher_rank1" in payload and "batcher" not in payload
    cursor_after_save = tr.train_batcher.cursor

    fresh = trainer_for(total_steps=100, warmup_steps=10, keep_last=3)
    fresh.dist_rank, fresh.dist_world = 1, 2
    fresh.load_checkpoint(ckpt)
    assert fresh.global_step == 1
    assert fresh.train_batcher.cursor == cursor_after_save


# -------------------------------------------------------- 2-rank train.py smoke
_DDP_CFG = """
model:
  vocab_size: 512
  dim: 256
  n_layers: 2
  layer_pattern: "ma"
  mamba: {{d_state: 16, d_conv: 4, expand: 2}}
  attention: {{n_heads: 2, n_kv_heads: 1, window: 32, n_summary: 8, topk: 8, rope_base: 10000.0}}
  dense_ffn: {{hidden: 128, activation: "swiglu"}}
  moe: {{n_routed: 8, n_shared: 1, top_k: 2, expert_hidden: 64, balance: "loss_free",
        aux_loss_weight: 0.01}}
  dropout: 0.0
  weight_tying: true
training:
  lr: 1.0e-3
  trunk_lr_mult: 0.1
  betas: [0.9, 0.95]
  weight_decay: 0.1
  grad_clip: 1.0
  warmup_steps: 2
  total_steps: 6
  grad_accum: 1
  amp: "fp32"
  seq_len: 32
  batch_size: 2
  activation_checkpoint: false
  checkpoint_every: 2
  keep_last: 2
  log_every: 1
  log_backend: "console"
  out_dir: "{outdir}"
data:
  dataset: "synthetic"
seed: 1337
run_name: "ddp_smoke"
"""


def test_two_rank_fp16_scale_lockstep() -> None:
    """fp16 + DDP: one rank's overflow must not diverge the world (M10 fix).

    Regression for the Kaggle casualty chain: per-param all-reduce skipped
    grads per rank (queues desynced) and a per-rank GradScaler would step on
    one rank while the other skipped. The driver asserts both ranks take the
    identical skip/step decisions under the flat-sync + shared-scale path.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests" / "_ddp_fp16_driver.py"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        cwd=str(ROOT),
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert out.count("[fp16-scale-lockstep] ok") == 1, out
    assert out.count("assert") == 0, out


def test_train_py_two_rank_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "ddp_smoke.yaml"
        cfg_path.write_text(_DDP_CFG.format(outdir=tmp.replace("\\", "/")), encoding="utf-8")
        env = dict(os.environ)
        env["LYDEN_DIST_BACKEND"] = "gloo"
        env.setdefault("OMP_NUM_THREADS", "1")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(cfg_path),
                "--max-steps",
                "2",
                "--val-batches",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(ROOT),
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        assert "[dist] rank 0/2 on cuda" in out, out
        assert "[dist] rank 1/2 on cuda" in out, out
        assert "[train] config=" in out and "world=2" in out
        assert out.count("[done]") == 1, out
        ckpts = sorted((Path(tmp) / "ddp_smoke" / "ckpts").glob("ckpt_*.pt"))
        assert ckpts, out
        payload = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        # Rank 0 saves; its key names its rank. Rank 1's batcher state is
        # intentionally not persisted (stateless random-window sampling).
        assert "batcher_rank0" in payload and "batcher" not in payload
        assert "batcher_rank1" not in payload
        assert payload["config_yaml"].startswith("model:")


def test_train_py_cross_phase_continue_runs_new_steps() -> None:
    """M11 regression: config total_steps counts NEW steps on cross-phase resume.

    scripts/train.py must bound the loop at entry_step + total_steps when
    continuing from an explicit --resume checkpoint (entry=2, cfg total=6 ->
    run to step 8), not at the absolute cfg number (which would exit instantly:
    '7600 < 2000' was the Kaggle casualty of 2026-10-19).
    """
    with tempfile.TemporaryDirectory() as tmp:

        def make_cfg(outdir: str) -> Path:
            cfg_path = Path(outdir) / "ddp_smoke.yaml"
            cfg_path.write_text(_DDP_CFG.format(outdir=outdir.replace("\\", "/")), encoding="utf-8")
            return cfg_path

        phase1_dir = Path(tmp) / "phase1"
        phase1_dir.mkdir()
        cfg_path = make_cfg(str(phase1_dir))
        env = dict(os.environ)
        env["LYDEN_DIST_BACKEND"] = "gloo"
        env.setdefault("OMP_NUM_THREADS", "1")

        def run(cfg_file: Path):
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    str(ROOT / "scripts" / "train.py"),
                    "--config",
                    str(cfg_file),
                    *extra,
                ],
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                cwd=str(ROOT),
            )

        # Phase 1: 2 fresh steps (--max-steps = new steps in this process)
        extra = ["--max-steps", "2"]
        p1 = run(cfg_path)
        out1 = p1.stdout + p1.stderr
        assert p1.returncode == 0, out1
        ckpt_dir = phase1_dir / "ddp_smoke" / "ckpts"
        entry_ckpt = ckpt_dir / "ckpt_0000002.pt"
        assert entry_ckpt.exists(), out1

        # Phase 2: DIFFERENT out_dir (fresh phase) — explicit --resume must be
        # honored; bound = entry + 6, not 6
        phase2_dir = Path(tmp) / "phase2"
        phase2_dir.mkdir()
        cfg2_path = make_cfg(str(phase2_dir))
        extra = ["--resume", str(entry_ckpt)]
        p2 = run(cfg2_path)
        out2 = p2.stdout + p2.stderr
        assert p2.returncode == 0, out2
        assert "cross-phase" in out2, out2
        assert "[done] reached step 8" in out2, out2
        assert (phase2_dir / "ddp_smoke" / "ckpts" / "ckpt_0000008.pt").exists(), out2
