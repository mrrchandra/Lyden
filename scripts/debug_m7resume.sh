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
snap = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
p4_full = {}
fresh = trainer_for(amp="fp32", total_steps=100, warmup_steps=10)
fresh.load_checkpoint(ckpt)
for i in range(3):
    fresh.step()
    if i == 0:
        snap4 = {n: p.detach().clone() for n, p in fresh.model.named_parameters()}
for _ in range(1):
    tr.step()
full4 = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
for _ in range(2):
    tr.step()
full6 = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
print("step4 vs full-run step4:")
mx = max((snap4[n] - full4[n]).abs().max().item() for n in snap4)
print("  max diff:", mx)
mx6 = max((fresh.model.state_dict()[n] - full6[n]).abs().max().item() for n in snap4)
print("step6 vs full-run step6 max diff:", mx6)
PYEOF
