#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, torch
sys.path.insert(0, "src")
sys.path.insert(0, "tests")
from test_train import trainer_for

tr = trainer_for(amp="fp32", total_steps=100, warmup_steps=10, keep_last=5)
for _ in range(3):
    tr.step()
ckpt = tr.save_checkpoint()
fresh = trainer_for(amp="fp32", total_steps=100, warmup_steps=10)
fresh.load_checkpoint(ckpt)

# check post-load optimizer state keys per group type
for name, grp in (("tr", tr.opt), ("fr", fresh.opt)):
    kinds = {}
    for i, g in enumerate(grp.param_groups):
        for p in g["params"]:
            st = grp.state.get(p, {})
            kinds.setdefault(i, []).append(tuple(sorted(st.keys()))[-1:] if st else "<empty>")
    flat = {i: set(kinds[i]) for i in kinds}
    print(name, "groups state key types:", flat)
print("sched lr TR", [g["lr"] for g in tr.opt.param_groups])
print("sched lr FR", [g["lr"] for g in fresh.opt.param_groups])
PYEOF
