#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, torch
sys.path.insert(0, "src")
from lyden.config import ModelConfig
from lyden.model.model import LydenModel

torch.manual_seed(1337)
torch.manual_seed(1337)
ModelConfig()
cfg = ModelConfig(vocab_size=8192, dim=256, n_layers=4, layer_pattern="mmma")
model = LydenModel(cfg).cuda().eval()
tok = torch.randint(0, 8192, (1, 24), device="cuda")
j = 13

def hook_blocks(inp):
    acts = {}
    x = model.emb(inp)
    acts["emb"] = x.clone()
    for i, blk in enumerate(model.blocks):
        n = blk.norm1(x)
        mixed_out = blk.mixer(n) if blk.kind == "a" else blk.mixer(n, cache=None)
        r1 = x + mixed_out
        acts[f"blk{i}_mix"] = r1.clone()
        r2 = r1 + blk.ffn(blk.norm2(r1))
        acts[f"blk{i}_ffn"] = r2.clone()
        x = r2
    acts["logits"] = model.head(model.final_norm(x))
    return acts

base = hook_blocks(tok)
tok2 = tok.clone(); tok2[0, j] = (tok2[0, j] + 7) % 8192
pert = hook_blocks(tok2)

for name in base:
    d = (base[name][:, :j] - pert[name][:, :j]).abs().max().item()
    nnz = (base[name][:, :j] != pert[name][:, :j]).any().item()
    print(f"{name:12s} maxdiff={d:.3e} differs={nnz}")
PYEOF
