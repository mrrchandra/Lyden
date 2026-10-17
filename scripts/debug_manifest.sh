#!/usr/bin/env bash
set -uo pipefail
head -c 400 /home/madrin/lyden-data-nano/shards/train/manifest.json
echo
head -c 400 /home/madrin/lyden-data-nano/shards/val/manifest.json
