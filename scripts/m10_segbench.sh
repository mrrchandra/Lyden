#!/usr/bin/env bash
# M10 seg-bench: locate the 100m slow path (fwd / fwd+bwd, with/without ckpt).
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import os
import sys, time, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from lyden.config import load_config
from lyden.model.model import LydenModel
import torch.nn.functional as F

cfg = load_config("configs/100m.yaml")
model = LydenModel(cfg.model).cuda()
torch.manual_seed(0)
BS = int(os.environ.get("PROBE_BS", str(cfg.training.batch_size)))
x = torch.randint(0, cfg.model.vocab_size, (BS, cfg.training.seq_len), device="cuda")
y = torch.randint(0, cfg.model.vocab_size, x.shape, device="cuda")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

def run(grad_ckpt: bool, label: str, steps: int = 3) -> None:
    model.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(x, grad_ckpt=grad_ckpt)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    toks = x.numel() / dt
    print(f"{label}: {dt*1000:.0f} ms/step | {toks/1000:.1f}k tok/s | mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    torch.cuda.reset_peak_memory_stats()

run(False, "grad   ")
run(True,  "gradck ")
model.eval()
with torch.no_grad():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    print(f"fwd-only no_grad: {(time.perf_counter()-t0)/5*1000:.0f} ms")

with torch.no_grad():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    print(f"fwd-only grad: {(time.perf_counter()-t0)/5*1000:.0f} ms")
PYEOF
