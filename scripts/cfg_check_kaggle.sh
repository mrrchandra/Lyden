#!/usr/bin/env bash
set -e
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'PYEOF'
import sys
sys.path.insert(0, 'src')
from lyden.config import load_config
c = load_config('configs/100m-kaggle.yaml')
t = c.training
tok_per_step = t.seq_len * t.batch_size * t.grad_accum
print('ok:', t.seq_len, t.batch_size, t.grad_accum, '->', tok_per_step, 'tok/step;',
      t.total_steps, 'steps;', t.total_steps * tok_per_step, 'tokens;', t.amp,
      '| d_state', c.model.mamba.d_state, '| data', c.data.data_dir)
PYEOF
