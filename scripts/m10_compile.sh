#!/usr/bin/env bash
# torch.compile A/B on the 100m shape (b4 seq 2048 ckpt d_state 128).
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF' 2>&1 | tail -4
import sys, time, torch, warnings, copy
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from lyden.config import load_config, MambaConfig
from lyden.model.model import LydenModel
import torch.nn.functional as F

cfg = load_config('configs/100m.yaml')
cfg.model = copy.copy(cfg.model)
cfg.model.mamba = MambaConfig(d_state=128, d_conv=4, expand=2)
torch.manual_seed(0)
x = torch.randint(0, cfg.model.vocab_size, (4, 2048), device='cuda')
y = torch.randint(0, cfg.model.vocab_size, x.shape, device='cuda')

model = LydenModel(cfg.model).cuda()
cm = torch.compile(model)
def bench(m, tag, steps=4):
    for _ in range(3):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = m(x, grad_ckpt=True)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
        loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(steps):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = m(x, grad_ckpt=True)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
        loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    print(f'{tag}: {dt*1000:.0f} ms/step | {x.numel()/dt/1000:.1f}k tok/s | mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
    torch.cuda.reset_peak_memory_stats()

bench(model, 'eager   ')
bench(cm,   'compiled')
PYEOF
