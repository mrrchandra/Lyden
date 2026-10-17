#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, torch
sys.path.insert(0, "src")
sys.path.insert(0, "tests")
from test_train import trainer_for, CountingBatcher

tr = trainer_for(amp="fp32", total_steps=80, warmup_steps=8, grad_accum=1)
losses = []
for i in range(12):
    losses.append(tr.step()["loss"])
print("losses:", ["%.3f" % l for l in losses])
PYEOF
