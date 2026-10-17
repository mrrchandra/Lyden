#!/usr/bin/env bash
set -uo pipefail
for dim in 128 256 384 512 768 1024 1280; do
  for ds in 64 128; do
    p=$(( 4*dim + 2*ds + 2*dim/64 ))
    if (( p % 8 == 0 )); then state=OK; else state=FAIL; fi
    echo "dim=$dim d_state=$ds d_in_proj=$p $state"
  done
done
