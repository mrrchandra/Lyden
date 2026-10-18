"""Full Lyden model (spec §3): embedding → mmma stack → final norm → head.

Weight tying shares the embedding and LM-head parameter.

Decode/inference plumbing (caches, KV reuse for the attention branch, CLM
loop) lands with the generation work; today the model is the training graph.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

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
        # Init policy: embeddings (and the tied head) N(0, 0.02) — GPT-style.
        # PyTorch defaults are left for Linear layers; mamba-ssm initializes
        # its own projections/dt parameters internally.
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        if not cfg.weight_tying:
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: Tensor,
        *,
        seq_shift: int = 0,
        collect_stats: bool = False,
        grad_ckpt: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute next-token logits (B, L, vocab). ``seq_shift`` is reserved
        for decode-time rope/cache continuity (v1 training: 0).

        With ``collect_stats=True`` returns ``(logits, stats)`` where stats
        aggregates, per attention layer index: attention gate weight means and
        MoE routing stats (``gate@<i>``, ``moe_load@<i>``, ``moe_entropy@<i>``,
        ``moe_aux@<i>``).
        """
        x = self.emb(token_ids)
        stats: dict[str, Tensor] = {}
        do_ckpt = grad_ckpt and self.training
        for i, blk in enumerate(self.blocks):
            if collect_stats and blk.kind == "a":
                if do_ckpt:
                    out, block_stats = checkpoint(  # type: ignore[assignment]
                        lambda xi, bk=blk: bk(xi, mixer_cache=None, collect_moe_stats=True),
                        x,
                        use_reentrant=False,
                    )
                else:
                    out, block_stats = blk(x, mixer_cache=None, collect_moe_stats=True)
                gate = block_stats["gate"].mean(dim=(0, 1)).detach()
                stats[f"gate@{i}"] = gate
                stats[f"moe_load@{i}"] = block_stats["load"].detach()
                stats[f"moe_entropy@{i}"] = block_stats["router_entropy"].detach()
                stats[f"moe_aux@{i}"] = block_stats["aux_loss"].detach()
                x = out
            elif do_ckpt:
                # Activation checkpointing: trade ~30% step time for linear
                # activation memory. Blocks are checkpointed individually;
                # use_reentrant=False composes with AMP and mamba-ssm's
                # custom autograd.
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x, mixer_cache=None)
        logits = self.head(self.final_norm(x))
        if collect_stats:
            return logits, stats
        return logits
