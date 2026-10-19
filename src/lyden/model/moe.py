"""Mixture-of-experts feed-forward (spec §3.3).

Top-k of ``n_routed`` routed experts + ``n_shared`` always-active shared
experts. Router is a plain ``Linear(dim, n_routed)``.

Load balancing:
- default ``loss_free``: DeepSeek-V3-style bias adjustment. Bias is added to
  the router scores for *selection only* (gate values come from the raw
  scores, so the bias never affects the magnitude of a selected expert's
  contribution). The bias is a non-trainable buffer, updated outside the
  optimizer: ``b_e -= gamma * sign(load_e - k/N)`` via :meth:`MoE.update_bias`.
- ``aux``: Switch-style aux loss ``w * N * sum_e f_e * P_e`` reported in the
  stats dict (``f_e`` = fraction of tokens routed to expert e, sums to k).

Experts are SwiGLU MLPs; the routed path loops over experts (v1, documented
optimization path: grouped GEMM). No capacity limits by default.

Submodule names are stable for the training loop's split-LR grouping:
``routed{i}``, ``shared{i}`` are the "fast" components along with ``router``.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lyden.config import MoEConfig


class Expert(nn.Module):
    """SwiGLU MLP: down(silu(up(x)) * gate(x))."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.dropout(F.silu(self.up(x)) * self.gate(x)))


class MoE(nn.Module):
    """Top-k routed experts + always-active shared experts."""

    def __init__(self, cfg: MoEConfig, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.n_routed = cfg.n_routed
        self.n_shared = cfg.n_shared
        self.top_k = min(cfg.top_k, cfg.n_routed)
        self.balance = cfg.balance
        self.aux_weight = cfg.aux_loss_weight
        self.bias_gamma = cfg.bias_gamma
        self._moe_cfg = cfg
        self.router_znorm = cfg.router_znorm

        # Router + loss-free bias: FAST module named "router" (split LR gr 0).
        self.router = nn.Linear(dim, cfg.n_routed, bias=False)
        bias: Tensor = torch.zeros(cfg.n_routed)
        self.register_buffer("bias", bias)

        self.routed = nn.ModuleDict(
            {f"routed{i}": Expert(dim, cfg.expert_hidden, dropout) for i in range(cfg.n_routed)}
        )
        self.shared = nn.ModuleDict(
            {f"shared{i}": Expert(dim, cfg.expert_hidden, dropout) for i in range(cfg.n_shared)}
        )
        self.expert_hidden = cfg.expert_hidden

    def _expert_forward(self, expert: nn.Module, x: Tensor) -> Tensor:
        """Single call site for the expert evaluation (grouped-GEMM swap point)."""
        return expert(x)

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (scores (B, L, N), selected (B, L, k), gate_values (B, L, k))."""
        scores = self.router(x)
        if self.router_znorm:
            # M11: standardize per token so bias deltas act on a bounded
            # scale regardless of router weight drift (affine ⇒ top-k
            # selection is unchanged).
            scores = (scores - scores.mean(-1, keepdim=True)) / (
                scores.std(-1, unbiased=False, keepdim=True) + 1e-5
            )
        selection = scores + self.bias  # bias only perturbs selection
        k = self.top_k
        top = torch.topk(selection, k, dim=-1).indices
        gate_values = scores.gather(-1, top)
        gate_values = gate_values / (gate_values.sum(-1, keepdim=True) + 1e-9)
        return scores, top, gate_values

    def forward(
        self, x: Tensor, *, collect_stats: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        b, seq_len, _ = x.shape
        device = x.device
        scores, top, gate_values = self._route(x)
        k = top.shape[-1]
        n = scores.shape[-1]

        # Routed path: loop over experts (v1), accumulate per selected token.
        flat = x.reshape(b * seq_len, -1)
        out_flat = torch.zeros_like(flat)
        top_flat = top.reshape(b * seq_len, k)
        gate_flat = gate_values.reshape(b * seq_len, k)
        for i, (_name, expert) in enumerate(self.routed.items()):
            sel = top_flat == i  # (B*L, k) bool
            if not sel.any():
                continue
            token_rows = sel.any(-1).nonzero(as_tuple=True)[0]
            if token_rows.numel() == 0:
                continue
            y = self._expert_forward(expert, flat[token_rows])  # (rows, dim)
            w = (gate_flat * sel.float()).sum(-1)[token_rows]  # (rows,)
            out_flat.index_add_(0, token_rows, y * w.unsqueeze(-1))
        for expert in self.shared.values():
            out_flat = out_flat + self._expert_forward(expert, flat)
        out = out_flat.view(b, seq_len, -1)

        if not collect_stats:
            return out

        counts = torch.zeros(n, device=device)
        counts.scatter_add_(0, top_flat.reshape(-1), torch.ones(b * seq_len * k, device=device))
        load = counts / (b * seq_len)  # tokens routed to e / total tokens; sums to k
        probs = F.softmax(scores.detach(), dim=-1).mean(dim=(0, 1))
        entropy = float(-(probs * probs.clamp_min(1e-9).log()).sum().item())
        aux = torch.zeros((), device=device)
        if self.balance == "aux":
            coefficients = n * load * probs
            aux = self.aux_weight * coefficients.sum()
        stats = {
            "load": load,
            "counts": counts,
            "router_entropy": torch.tensor(entropy, device=device),
            "aux_loss": aux,
            "gate_values": gate_values,
            "selected": top,
            "scores": scores.detach(),
        }
        self._last_load = load.detach()
        return out, stats

    def reset_bias(self) -> None:
        """Zero the loss-free bias (continue-train restart of balancing)."""
        cast("Tensor", self.bias).zero_()

    def update_bias(
        self,
        load: Tensor | None = None,
        *,
        gamma: float | None = None,
        step: int = 0,
    ) -> None:
        """DeepSeek-V3 loss-free balancing: called once per optimizer step.

        ``load`` defaults to the last ``collect_stats=True`` forward. Overloaded
        experts (load > k/N) get their bias decreased; underloaded get it
        increased. ``gamma`` defaults to the config's ramp at ``step``
        (plain ``bias_gamma`` without a ramp). The bias is a non-trainable
        buffer, updated outside the optimizer.
        """
        if load is None:
            load = self._last_load
            if load is None:
                raise RuntimeError("update_bias() needs load: forward first or pass load")
        assert load is not None  # narrowed by the branch above
        if gamma is None:
            gamma = effective_bias_gamma(self._moe_cfg, step)
        target = self.top_k / self.n_routed
        bias = cast("Tensor", self.bias)
        bias -= gamma * (load - target).sign()


def effective_bias_gamma(cfg: MoEConfig, step: int) -> float:
    """Per-step loss-free gamma (M11): linear ramp from ``gamma_start`` down
    to ``bias_gamma`` over ``ramp_steps``; constant without a ramp."""
    ramp = cfg.bias_gamma_ramp
    if ramp is None:
        return cfg.bias_gamma
    gamma_start, gamma_end, ramp_steps = ramp
    if step >= ramp_steps:
        return gamma_end
    return gamma_start + (gamma_end - gamma_start) * (step / ramp_steps)
