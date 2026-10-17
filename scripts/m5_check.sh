#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python -c 'from lyden.model import MoE, Expert; print("import ok")'
/home/madrin/venvs/lyden/bin/python -m pytest -q | tail -1
/home/madrin/venvs/lyden/bin/ruff check . | tail -1
/home/madrin/venvs/lyden/bin/mypy src tests scripts | tail -1
