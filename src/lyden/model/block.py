"""Lyden transformer block (spec §3.1): pre-norm mixer + pre-norm FFN.

  m: x + Mamba(rms1(x))       →  + DenseFFN(rms2(·))
  a: x + HybridAttention(rms1(x)) → + MoE(rms2(·))

Build from :class:`lyden.config.ModelConfig` via ``LydenBlock(kind, cfg, i)``;
``i`` is the global layer index (mamba's ``layer_idx`` must be unique across
the stack for decode-time caches).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from torch import Tensor, nn

from lyden.config import ModelConfig
from lyden.model.dense_ffn import DenseFFN
from lyden.model.hybrid_attention import HybridAttention
from lyden.model.mamba_block import MambaBlock
from lyden.model.moe import MoE

BlockCacheKind = MutableMapping | None


class LydenBlock(nn.Module):
    """One pre-norm residual block; ``kind`` is 'm' or 'a'."""

    def __init__(self, kind: str, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.kind = kind
        dim = cfg.dim
        self.mixer: nn.Module  # MambaBlock | HybridAttention
        self.ffn: nn.Module  # DenseFFN | MoE
        if kind == "m":
            m = cfg.mamba
            self.mixer = MambaBlock(
                dim,
                d_state=m.d_state,
                d_conv=m.d_conv,
                expand=m.expand,
                layer_idx=layer_idx,
                dropout=cfg.dropout,
            )
            self.ffn = DenseFFN(cfg.dense_ffn, dim, dropout=cfg.dropout)
        elif kind == "a":
            att = cfg.attention
            self.mixer = HybridAttention(
                dim,
                n_heads=att.n_heads,
                n_kv_heads=att.n_kv_heads,
                window=att.window,
                n_summary=att.n_summary,
                topk=att.topk,
                rope_base=att.rope_base,
                dropout=cfg.dropout,
            )
            self.ffn = MoE(cfg.moe, dim, dropout=cfg.dropout)
        else:
            raise ValueError(f"kind must be 'm' or 'a', got '{kind}'")
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def forward(
        self,
        x: Tensor,
        *,
        mixer_cache: BlockCacheKind = None,
        seq_idx: Tensor | None = None,
        collect_moe_stats: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Training-semantics forward.

        ``mixer_cache``/``seq_idx`` are honored for mamba blocks (recurrent
        decode at inference); attention blocks accept no cache yet (decode
        wiring arrives with the generation loop).
        """
        normed = self.norm1(x)
        if self.kind == "m":
            mixed = self.mixer(normed, cache=mixer_cache, seq_idx=seq_idx)
        else:
            if mixer_cache is not None:
                raise ValueError("attention blocks accept no mixer_cache yet")
            if collect_moe_stats:
                branches = self.mixer(normed, return_branches=True)
                stats: dict[str, Any] = {"gate": branches["gate"]}
                mixed = branches["out"]
            else:
                mixed = self.mixer(normed)
        out = x + mixed
        normed2 = self.norm2(out)
        if self.kind == "a" and collect_moe_stats:
            ffn_out, moe_stats = self.ffn(normed2, collect_stats=True)
            stats.update(moe_stats)
            return out + ffn_out, stats
        return out + self.ffn(normed2)


def build_blocks(cfg: ModelConfig) -> nn.ModuleList:
    """Expand ``layer_pattern`` with rotation into the full stack."""
    blocks = nn.ModuleList()
    for i, kind in enumerate(cfg.layer_kinds()):
        blocks.append(LydenBlock(kind, cfg, layer_idx=i))
    return blocks
