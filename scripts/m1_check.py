#!/usr/bin/env python3
"""M1 verification battery: exercises every built component with pass/fail evidence.

Run inside WSL:  uv run --project /mnt/c/Users/Acer/Desktop/Projs/Lyden python scripts/m1_check.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        PASS.append(f"{name:>28}  {detail}")
        print(f"[PASS] {name:>28}  {detail}")
    except Exception as exc:  # noqa: BLE001 - report every failure with context
        FAIL.append(f"{name:>28}  {type(exc).__name__}: {exc}")
        print(f"[FAIL] {name:>28}  {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- cuda / torch
def _torch_import():
    import torch

    return f"torch {torch.__version__}"


def _cuda_available():
    import torch

    assert torch.cuda.is_available(), "cuda not available"
    return torch.cuda.get_device_name(0)


def _bf16_support():
    import torch

    assert torch.cuda.is_bf16_supported(), "bf16 unsupported"
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    torch.cuda.synchronize()
    return "bf16 matmul ok"


def _fp32_ssm_dtype():
    """Sanity: fp32 matmul & sqrt norms (used for SSM state later)."""
    import torch

    x = torch.randn(128, 128, device="cuda")
    y = x @ x
    assert torch.isfinite(y).all()
    return "fp32 matmul finite"


# ---------------------------------------------------------------- mamba / conv
def _mamba2_fwd_bwd():
    import torch
    from mamba_ssm import Mamba2

    m = Mamba2(d_model=128, d_state=32, d_conv=4, expand=2).cuda()
    x = torch.randn(2, 128, 128, device="cuda", requires_grad=True)
    y = m(x)
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "non-finite grad"
    grads = sum(1 for p in m.parameters() if p.requires_grad)
    assert grads > 0, "no params"
    return f"out {tuple(y.shape)}, ∇finite, {grads} param tensors"


def _mamba2_determinism():
    import torch
    from mamba_ssm import Mamba2

    m = Mamba2(d_model=64, d_state=16, d_conv=4, expand=2).cuda().eval()
    x = torch.randn(1, 32, 64, device="cuda")
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b), "nondeterministic forward"
    return "repeated fwd identical"


def _causal_conv1d_vs_reference():
    """Numerical correctness: causal_conv1d vs torch conv1d with left padding."""
    import torch
    from causal_conv1d import causal_conv1d_fn

    torch.manual_seed(0)
    b, d, t, w = 2, 16, 256, 4
    x = torch.randn(b, d, t, device="cuda", dtype=torch.float32)
    weight = torch.randn(d, w, device="cuda")
    bias = torch.randn(d, device="cuda")

    out = causal_conv1d_fn(x, weight, bias, activation=None)

    ref = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x, (w - 1, 0)), weight[:, None, :], bias, groups=d
    )
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"max abs err {err}"
    return f"max|Δ|={err:.2e} vs conv1d reference"


# ---------------------------------------------------------------- config
def _configs_load():
    from lyden.config import load_config

    names = ["nano", "100m", "250m", "500m"]
    out = []
    for n in names:
        cfg = load_config(f"configs/{n}.yaml")
        kinds = cfg.layer_kinds()
        assert kinds.count("m") == 3 * kinds.count("a"), f"{n}: 3:1 violated"
        assert cfg.model.n_layers == len(kinds)
        out.append(n)
    return ", ".join(out)


def _config_roundtrip():
    from lyden.config import load_config, save_config, to_dict

    cfg = load_config("configs/250m.yaml")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.yaml"
        save_config(cfg, p)
        loaded = load_config(p)
        assert to_dict(loaded) == to_dict(cfg)
    return "250m save/load identical"


def _config_unknown_key():
    from lyden.config import LydenConfig, from_dict

    try:
        from_dict(LydenConfig, {"model": {"dims": 999}})
        raise AssertionError("unknown key accepted")
    except ValueError as e:
        assert "Unknown config key" in str(e)
    return "typo keys rejected"


def _config_error_flags():
    from lyden.config import ModelConfig

    for kwargs in ({"n_layers": 7}, {"layer_pattern": "mx"}, {"dropout": 1.5}):
        try:
            ModelConfig(**kwargs)
            raise AssertionError(f"accepted invalid {kwargs}")
        except ValueError:
            pass
    return "validation errors fire"


# ---------------------------------------------------------------- seed
def _seed_reproducibility():
    import random

    import numpy as np
    import torch

    from lyden.utils.seed import set_seed

    set_seed(1337)
    a = [random.random(), float(np.random.rand()), float(torch.rand(1).item())]
    set_seed(1337)
    b = [random.random(), float(np.random.rand()), float(torch.rand(1).item())]
    assert a == b, f"seed mismatch {a} vs {b}"
    return "random/np/torch identical after re-seed"


# ---------------------------------------------------------------- logging
def _logger_artifacts():
    import os

    from lyden.config import load_config
    from lyden.utils.logging import Logger

    cfg = load_config("configs/100m.yaml")  # log_backend: tensorboard
    assert cfg.training.log_backend == "tensorboard"
    with tempfile.TemporaryDirectory() as td:
        log = Logger(cfg, td)
        log.log_metrics({"loss": 5.4321, "lr": 0.001}, step=17)
        log.log_metrics({"loss": 5.1234}, step=18)
        log.close()
        files = sorted(os.listdir(td))
        assert "config.yaml" in files and "environment.json" in files, files
        tb_dir = Path(td) / "tb"
        events = list(tb_dir.rglob("events*"))
        assert events, "no tensorboard event file"
    return "config.yaml + environment.json + tb events"


def _env_record_content():
    import json

    from lyden.config import load_config
    from lyden.utils.logging import Logger

    cfg = load_config("configs/100m.yaml")
    with tempfile.TemporaryDirectory() as td:
        log = Logger(cfg, td)
        log.close()
        env = json.loads((Path(td) / "environment.json").read_text())
        assert env["cuda_available"] is True
        assert "gpu" in env and "vram_total_gb" in env
    return f"gpu={env['gpu'].split('NVIDIA')[-1].strip()[:20]}"


# ---------------------------------------------------------------- CLI deps
def _tokenizer_lib():
    import tokenizers

    assert tokenizers.Tokenizer is not None
    return "tokenizers importable"


def _datasets_lib():
    import datasets

    assert datasets
    return f"datasets {datasets.__version__}"


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}")
    print("-" * 70)
    checks = [
        ("torch import", _torch_import),
        ("cuda device", _cuda_available),
        ("bf16 (training amp)", _bf16_support),
        ("fp32 (ssm dtype)", _fp32_ssm_dtype),
        ("mamba2 fwd+bwd", _mamba2_fwd_bwd),
        ("mamba2 determinism", _mamba2_determinism),
        ("causal_conv1d math", _causal_conv1d_vs_reference),
        ("configs 3:1 pattern", _configs_load),
        ("config yaml roundtrip", _config_roundtrip),
        ("config typo rejection", _config_unknown_key),
        ("config validation errs", _config_error_flags),
        ("seed reproducibility", _seed_reproducibility),
        ("logger artifacts", _logger_artifacts),
        ("env record has gpu", _env_record_content),
        ("tokenizers lib", _tokenizer_lib),
        ("datasets lib", _datasets_lib),
    ]
    for name, fn in checks:
        check(name, fn)
    print("-" * 70)
    print(f"TOTAL: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print(" ", f)
    shutil_ruff = shutil.which("ruff")
    print(f"(ruff binary seen: {shutil_ruff})")
    sys.exit(1 if FAIL else 0)
