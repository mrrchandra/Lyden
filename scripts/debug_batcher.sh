#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import sys
sys.path.insert(0, "src")
import numpy as np, torch
from tests.test_dataset import TEXT, toy_encode, BOS, EOS
from lyden.data.dataset import pretokenize
from lyden.data.dataloader import PackedBatcher, ShardDataset
import tempfile, pathlib

td = pathlib.Path(tempfile.mkdtemp())
pretokenize({"train": lambda: iter(TEXT for _ in range(2000))}, toy_encode, out_dir=td, tokens_per_shard=100_000)
ds = ShardDataset(td, "train")
print("stream len:", len(ds))

a = PackedBatcher(ds, seq_len=8, batch_size=4, seed=7, shuffle=True)
rng = np.random.default_rng(7)
s1 = int(rng.integers(0, len(ds) - 36)); s2 = int(rng.integers(0, len(ds) - 36))
print("sampled starts:", s1, s2, "| mod 19:", s1 % 19, s2 % 19)
x1, _ = a.next_batch(); x2, _ = a.next_batch()
print("equal:", torch.equal(x1, x2), "| starts mod19 equal:", s1 % 19 == s2 % 19)
EOF
