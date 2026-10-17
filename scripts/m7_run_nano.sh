#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
nohup /home/madrin/venvs/lyden/bin/python scripts/train.py --config configs/nano.yaml --max-steps 5000 \
  > /tmp/nano_run.log 2>&1 &
echo "PID $!"
