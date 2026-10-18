#!/usr/bin/env bash
# Isolate trainer micro cost: ckpt x collect_stats at the 100m train shape.
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, time, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from lyden.config import load_config
from lyden.model.model import LydenModel
import torch.nn.functional as F

cfg = load_config("configs/100m.yaml")
torch.manual_seed(0)
BS = cfg.training.batch_size
x = torch.randint(0, cfg.model.vocab_size, (BS, cfg.training.seq_len), device="cuda")
y = torch.randint(0, cfg.model.vocab_size, x.shape, device="cuda")
model = LydenModel(cfg.model).cuda()
model.train()

def micro(stats: bool, ckpt: bool) -> None:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if stats:
            logits, _ = model(x, collect_stats=True, grad_ckpt=ckpt)
        else:
            logits = model(x, grad_ckpt=ckpt)
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)

def bench(label: str, stats: bool, ckpt: bool, steps: int = 3) -> None:
    for _ in range(2):
        micro(stats, ckpt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        micro(stats, ckpt)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    print(f"{label}: {dt*1000:.0f} ms/micro | {x.numel()/dt/1000:.1f}k tok/s | mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    torch.cuda.reset_peak_memory_stats()

bench("no-ckpt no-stats", False, False)
bench("ckpt     no-stats", False, True)
bench("ckpt     stats    ", True, True)
PYEOF
