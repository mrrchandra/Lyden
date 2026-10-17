#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys, numpy as np, torch
sys.path.insert(0, "src")
sys.path.insert(0, "tests")
from test_train import trainer_for

class CycleBatcher:
    def __init__(self, batch_size, seq_len):
        self.b, self.l = batch_size, seq_len
        self.cursor = 0
    def next_batch(self):
        span = self.b * (self.l + 1)
        n = np.arange(self.cursor, self.cursor + span)
        tok = (n % 8) + 100
        chunk = tok.reshape(self.b, self.l + 1)
        self.cursor += span
        t = torch.from_numpy(chunk.astype(np.int64))
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()
    def state_dict(self):
        return {"cursor": self.cursor}
    def load_state_dict(self, st):
        self.cursor = int(st["cursor"])

from lyden.config import LydenConfig, ModelConfig, MoEConfig
from lyden.model.model import LydenModel
from lyden.train import Trainer

torch.manual_seed(11)
cfg = LydenConfig(model=ModelConfig(vocab_size=8192, dim=256, n_layers=4,
                                    moe=MoEConfig(n_routed=8, n_shared=2, top_k=2, expert_hidden=256)))
cfg.training.lr = 1e-3
cfg.training.warmup_steps = 5
cfg.training.total_steps = 200
cfg.training.grad_accum = 1
tr = Trainer(cfg, LydenModel(cfg.model).cuda(), CycleBatcher(4, 64), val_batcher=CycleBatcher(2, 64))
losses = [tr.step()["loss"] for _ in range(50)]
print("losses:", " ".join("%.2f" % l for l in losses))
PYEOF
