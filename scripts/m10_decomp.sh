#!/usr/bin/env bash
# Decompose the trainer step: micro fwd+bwd, opt step, batcher fetch.
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF' 2>&1 | tail -6
import sys, time, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from lyden.config import load_config
from lyden.model.model import LydenModel
from lyden.train import Trainer
from train import build_batchers

cfg = load_config('configs/100m.yaml')
torch.manual_seed(0)
model = LydenModel(cfg.model).cuda()
train_b, val_b = build_batchers(cfg)
tr = Trainer(cfg, model, train_b, val_batcher=val_b, device='cuda')

# warm 2 steps
for _ in range(2):
    tr.step()
torch.cuda.synchronize()

def avg(label, fn, n=5):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    print(f'{label}: {(time.perf_counter()-t0)/n*1000:.0f} ms')

x, y = train_b.next_batch()
def micro():
    tr._forward_loss(x, y)[0].backward()
    tr.opt.zero_grad(set_to_none=True)
avg('4x micro fwd+bwd (accum unit)', lambda: micro(), n=2)   # single micro cost
def one_full_backward_accum():
    tr.opt.zero_grad(set_to_none=True)
    for _ in range(tr.cfg.training.grad_accum):
        micro()
    tr.scaler.unscale_(tr.opt) if tr.scaler.is_enabled() else None
avg('full accum (4 micro)', one_full_backward_accum, n=3)
avg('opt.step', lambda: tr.opt.step(), n=5)
avg('batcher.next_batch', lambda: train_b.next_batch(), n=8)
PYEOF
