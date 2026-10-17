#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
try:
    from torch.utils.tensorboard import SummaryWriter
    print("summarywriter imports OK")
except Exception as e:
    import traceback
    traceback.print_exc()
EOF
