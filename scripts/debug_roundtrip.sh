#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import traceback

import sys
sys.path.insert(0, "src")
from lyden.data import LydenTokenizer

tok = LydenTokenizer.load("/home/madrin/lyden-data/tokenizer")
text = "Once upon a time there was a little fish named Fin. <|edit|>WRITE new fact<|/edit|>"
ids = tok.encode(text, add_bos=True, add_eos=True)
print("ids:", ids[:12], "...", ids[-8:])
print("first/last:", ids[0], ids[-1])
specials = {3, 4}
print("specials present:", specials.issubset(set(ids)))
out = tok.decode(ids)
print("EQUAL?", out == text)
if out != text:
    a, b = out, text
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            print(f"first diff at {i}: decoded {a[max(0,i-30):i+30]!r}")
            print(f"                 source  {b[max(0,i-30):i+30]!r}")
            break
print("decoded:", repr(out))
EOF
