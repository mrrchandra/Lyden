#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python scripts/generate.py \
  --config configs/nano.yaml \
  --ckpt runs/nano/ckpts/ckpt_0005000.pt \
  --prompt 'Things to remember
My favorite color is teal.
<|edit|>WRITE with "The user likes stories about cats."<|/edit|>' \
  --max-new-tokens 40 --temperature 0.7 --top-k 40 --seed 1 --clm 2>/dev/null
