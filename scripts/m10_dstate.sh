#!/usr/bin/env bash
# d_state A/B on the 100m shape (cfg dataset val-split only for dims).
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, time, torch, warnings, copy
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from lyden.config import load_config, MambaConfig
from lyden.model.model import LydenModel
import torch.nn.functional as F

cfg = load_config("configs/100m.yaml")
torch.manual_seed(0)
BS = 2
x = torch.randint(0, cfg.model.vocab_size, (BS, cfg.training.seq_len), device="cuda")
y = torch.randint(0, cfg.model.vocab_size, x.shape, device="cuda")

for dstate in (int(a) for a in sys.argv[1:] if a.isdigit()) or (128, 64):
    c2 = copy.copy(cfg)
    c2.model = copy.copy(cfg.model)
    c2.model.mamba = MambaConfig(d_state=dstate, d_conv=4, expand=2)
    torch.manual_seed(0)
    model = LydenModel(c2.model).cuda()
    torch.cuda.reset_peak_memory_stats()
    # warmup once (kernels, cuBLAS)
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, grad_ckpt=True)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 5
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, grad_ckpt=True)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    print(f"d_state={dstate}: {dt*1000:.0f} ms/step | {x.numel()/dt/1000:.1f}k tok/s | mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    del model, logits, loss
    torch.cuda.empty_cache()
PYEOF
