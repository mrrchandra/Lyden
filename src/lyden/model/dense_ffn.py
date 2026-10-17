"""Dense feed-forward used after Mamba blocks (spec §3.1).

``swiglu``: down(silu(up(x)) * gate(x)) — always three linears.
``gelu``: fc2(gelu(fc1(x))).

Mamba blocks have no MoE (explicit user directive); this is their FFN. It is
trunk-speed (0.1× LR in the split-LR grouping), so submodule names avoid the
fast prefixes ("routed"/"shared"/"router").
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from lyden.config import DenseFFNConfig


class DenseFFN(nn.Module):
    def __init__(self, cfg: DenseFFNConfig, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = cfg.hidden
        self.activation = cfg.activation
        dropout_mod = nn.Dropout(dropout)
        if cfg.activation == "swiglu":
            self.up = nn.Linear(dim, cfg.hidden, bias=False)
            self.gate = nn.Linear(dim, cfg.hidden, bias=False)
            self.down = nn.Linear(cfg.hidden, dim, bias=False)
            self.dropout = dropout_mod
        else:
            self.fc1 = nn.Linear(dim, cfg.hidden, bias=False)
            self.fc2 = nn.Linear(cfg.hidden, dim, bias=False)
            self.dropout = dropout_mod

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            return self.down(self.dropout(F.silu(self.up(x)) * self.gate(x)))
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))
