#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
timeout 240 /home/madrin/venvs/lyden/bin/python scripts/train.py --config configs/nano.yaml --max-steps 50 2>/dev/null
