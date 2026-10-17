#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
timeout 90 /home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, time
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from train import build_batchers, set_all_seeds
from lyden.config import load_config
from lyden.model.model import LydenModel
from lyden.train import Trainer

cfg = load_config("configs/nano.yaml")
cfg.training.amp = "fp32"
print("building...", flush=True)
tr_b, va_b = build_batchers(cfg)
model = LydenModel(cfg.model).cuda()
trainer = Trainer(cfg, model, tr_b, val_batcher=va_b)

print("3 steps...", flush=True)
for _ in range(3):
    info = trainer.step()
    print("step", info["step"], info["loss"], flush=True)

import time
t0 = time.time()
v = trainer.validate(2)
print("validate(2):", v, f"{time.time()-t0:.1f}s", flush=True)

t0 = time.time()
v = trainer.validate(20)
print("validate(20):", v, f"{time.time()-t0:.1f}s", flush=True)
PYEOF
