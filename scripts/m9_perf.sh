#!/usr/bin/env bash
# M9 gates: stateful decode tok/s, time/step, VRAM, and sample coherence.
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, time, torch
from pathlib import Path
sys.path.insert(0, "src")

from lyden.config import load_config
from lyden.data.tokenizer import LydenTokenizer
from lyden.model.model import LydenModel
from lyden.model.decode import prefill, step, greedy_decode
from lyden.memory import generate_with_clm

cfg = load_config("configs/nano.yaml")
ckpt = Path("runs/nano/ckpts/ckpt_0005000.pt")
model = LydenModel(cfg.model).cuda().eval()
payload = torch.load(ckpt, map_location="cuda", weights_only=False)
model.load_state_dict(payload["model"])
tok = LydenTokenizer.load(cfg.data.tokenizer_dir)

prompt = "Once upon a time, there was a little dragon named Sparky."
ids = torch.tensor([tok.encode(prompt, add_bos=True)], device="cuda")
n = 200

# ---- stateful decode throughput ----
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
state, logits = prefill(model, ids)
for _ in range(n):
    nxt = int(torch.argmax(logits[0, -1]).item())
    if nxt == tok.eos_id:
        break
    logits = step(model, state, nxt)
dt = time.perf_counter() - t0
peak_alloc = torch.cuda.max_memory_allocated() / 2**30
peak_reserved = torch.cuda.max_memory_reserved() / 2**30
print(f"[m9] n_steps={n} wall={dt:.2f}s tok/s={n/dt:.1f} ms/step={dt/n*1000:.1f}")
print(f"[m9] peak alloc {peak_alloc:.2f} GiB | reserved {peak_reserved:.2f} GiB")

# ---- greedy sanity text ----
out = greedy_decode(model, ids, 120, stop_on_specials={tok.eos_id})
print("[m9] greedy sample:")
print(tok.decode(out, skip_special=True))

# ---- v1 full-recompute baseline timing ----
t0 = time.perf_counter()
cur = ids
for _ in range(60):
    lg = model(cur)
    nxt = int(torch.argmax(lg[0, -1]).item())
    cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
dt1 = time.perf_counter() - t0
print(f"[m9] v1: 60 tok in {dt1:.2f}s -> {60/dt1:.1f} tok/s")

# ---- CLM stateful path ----
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
text, buffer = generate_with_clm(
    model, tok, prompt, max_new_tokens=160, temperature=0.8, top_k=50, max_edits=2)
dt2 = time.perf_counter() - t0
print(f"[m9] clm gen {dt2:.2f}s; peak {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
print("[m9] clm tail:")
print("\n".join(text.splitlines()[-12:]))
print("[m9] clm buffer state lines:", len(buffer.lines))
PYEOF
