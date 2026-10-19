"""Mamba-2 block: RMSNorm → official mamba-ssm Mamba2 → residual (+dropout).

The selective SSM core is the official ``mamba_ssm`` Mamba-2 module (per
spec §3.1: no custom SSM). ``MambaBlock`` is the thin in-project wrapper:

- ``forward(x, cache=None)`` trains via the parallel chunk scan (no cache)
  or runs recurrently when an ``InferenceParams`` shared-cache object is
  passed (prefill at ``seqlen_offset == 0``, single-token ``step`` after).
- The caller owns ``cache.seqlen_offset`` (advance once per decode step
  after passing the cache to every block, as in ``mamba_ssm.generation``).

x: (B, L, dim) float; out: (B, L, dim).
"""

from __future__ import annotations

from mamba_ssm.modules.mamba2 import Mamba2
from torch import Tensor, nn


class MambaBlock(nn.Module):
    """Pre-norm Mamba-2 residual unit wrapping the official mamba-ssm mixer.

    Constraint (from the causal-conv1d kernel used on the recurrent path):
    ``d_in_proj = 4*dim + 2*d_state + 2*dim/64`` must be a multiple of 8.
    All canonical sizes (dim >= 256) satisfy this; the explicit ValueError
    replaces an opaque CUDA assert that would otherwise surface at cache
    prefill time.
    """

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        head_dim = 64  # mamba-ssm default; kernel asserts (expand * d_model) % headdim == 0
        if (expand * dim) % head_dim:
            raise ValueError(
                f"expand * dim ({expand} * {dim} = {expand * dim}) must be divisible by the "
                f"mamba-ssm headdim of {head_dim}; adjust dim or expand"
            )
        d_in_proj = 4 * dim + 2 * d_state + 2 * dim // 64
        if d_in_proj % 8:
            raise ValueError(
                f"d_in_proj ({d_in_proj}, computed as 4*dim + 2*d_state + 2*dim/64 = "
                f"4*{dim} + 2*{d_state} + 2*{dim}/64) must be a multiple of 8 for the "
                f"causal-conv1d recurrent path; adjust dim or d_state"
            )
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.mixer = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
            # Same algebra for training, cache prefill, and single-token steps:
            # the fused mem-eff path and the manual decomposition differ numerically
            # (~1e-2 in the residual chain), which would break exact decode-vs-recompute
            # equivalences; keep both on the manual causal-conv1d + chunk-scan path.
            use_mem_eff_path=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: Tensor,
        cache: object | None = None,
        seq_idx: Tensor | None = None,
    ) -> Tensor:
        """x: (B, L, dim) -> (B, L, dim).

        cache: shared ``InferenceParams`` for recurrent decode (caller
        advances ``seqlen_offset`` once per step across all blocks), None for
        training's parallel scan. seq_idx: optional (B, L) int segment ids
        that isolate packed documents (var-length training).
        """
        return x + self.dropout(self.mixer(self.norm(x), inference_params=cache, seq_idx=seq_idx))
