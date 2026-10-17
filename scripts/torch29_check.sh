#!/usr/bin/env bash
set -euo pipefail
python3 - <<'EOF'
import json, urllib.request
meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/torch/2.9.0/json"))
deps = meta["info"]["requires_dist"] or []
for d in deps:
    if "nvidia" in d or "triton" in d:
        print(d)
EOF
