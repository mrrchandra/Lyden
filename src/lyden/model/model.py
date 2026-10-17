"""Full Lyden model (spec §3): embedding → mmma stack → final norm → head.

Weight tying shares the embedding and LM-head parameter.

Decode/inference plumbing (caches, KV reuse for the attention branch, CLM
loop) lands with the generation work; today the model is the training graph.
"""

from __future__ import annotations

from torch import Tensor, nn

from lyden.config import ModelConfig
from lyden.model.block import build_blocks


class LydenModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = build_blocks(cfg)
        self.final_norm = nn.RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.weight_tying:
            self.head.weight = self.emb.weight

    def forward(self, token_ids: Tensor, *, seq_shift: int = 0) -> Tensor:
        """Compute next-token logits (B, L, vocab). ``seq_shift`` is reserved
        for decode-time rope/cache continuity (v1 training: 0)."""
        x = self.emb(token_ids)
        for blk in self.blocks:
            x = blk(x, mixer_cache=None)
        return self.head(self.final_norm(x))
