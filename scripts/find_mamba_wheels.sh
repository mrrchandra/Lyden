#!/usr/bin/env bash
# List wheel asset URLs for given release tags (all ABI variants).
set -euo pipefail
curl -s "https://api.github.com/repos/state-spaces/mamba/releases/tags/v2.3.2.post1" |
  python3 -c '
import json, sys
for a in json.load(sys.stdin).get("assets", []):
    n = a["name"]
    if "cp312" in n and ("torch2.8" in n or "torch2.9" in n):
        print(a["browser_download_url"])
'
