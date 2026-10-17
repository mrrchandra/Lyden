#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import sys, tempfile, os
sys.path.insert(0, "src")
from lyden.config import load_config
from lyden.utils.logging import Logger
cfg = load_config("configs/nano.yaml")
td = tempfile.mkdtemp()
log = Logger(cfg, td)
print("backend:", type(log.backend))
log.log_metrics({"loss": 1.0}, step=1)
log.close()
for root, dirs, files in os.walk(td):
    for f in files:
        print(os.path.join(root, f))
EOF
