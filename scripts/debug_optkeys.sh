#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import torch
p = torch.nn.Parameter(torch.zeros(4))
opt = torch.optim.AdamW([{"params": [p]}])
loss = (p ** 2).sum()
loss.backward()
opt.step()
print(sorted(opt.state[list(opt.state)[0]].keys()))
PYEOF
