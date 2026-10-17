#!/usr/bin/env bash
set -euo pipefail
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import torch
print("torch:", torch.__version__, torch.version.cuda)
print("cxx11abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
EOF
