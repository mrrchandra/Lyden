#!/usr/bin/env bash
# Profile one micro-batch (b4, seq 2048, ckpt) at d_state 128.
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF' 2>&1 | tail -30
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
for _ in range(2):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x, grad_ckpt=True)
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
    loss.backward(); model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, grad_ckpt=True)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
        loss.backward(); model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=14, max_name_column_width=52))
PYEOF
