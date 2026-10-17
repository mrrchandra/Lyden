#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, math, torch
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from train import build_batchers
from lyden.config import load_config
from lyden.model.model import LydenModel

cfg = load_config("configs/nano.yaml")
tr_b, va_b = build_batchers(cfg)
x, y = tr_b.next_batch()
print("batch shapes:", x.shape, y.shape, x.dtype)
print("token id range:", int(x.min()), int(x.max()))
print("expected init loss:", math.log(cfg.model.vocab_size))
model = LydenModel(cfg.model).cuda().eval()
with torch.no_grad():
    logits = model(x[:2].cuda())
    print("logit stats: mean %.3f std %.3f" % (logits.mean(), logits.std()))
    loss = torch.nn.functional.cross_entropy(
        logits.float().view(-1, logits.shape[-1]), y[:2].cuda().view(-1)
    )
    print("sub-batch loss:", float(loss))
PYEOF
