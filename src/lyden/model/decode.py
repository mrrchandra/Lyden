"""Stateful decode: prefill + per-token step with exact training semantics.

v1 decode is prefill once, then single-token steps:

- mamba blocks: official mamba-ssm recurrent path through a shared
  ``InferenceParams`` (conv/ssm states updated in-place by the module).
- attention blocks: rope'd keys/values + indexer ranks cached at prefill
  (positions 0..T-1); each step appends the new token's rope'd k/v/rank and
  rebuilds only the compressed summary bank over the cached context —
  sliding/selective/top-k math is reproduced with explicit positions, so a
  step row equals the corresponding row of a full re-encode (up to kernel
  numeric noise ~1e-5; tests assert greedy-continuation equivalence).

Batch 1 only (generation). The caller owns the cache: ``prefill`` builds
a ``DecodeState`` from ids, ``step`` extends it by one token, and a mid-
stream edit simply prefills again from an updated buffer (v1 CLM policy).
"""

from __future__ import annotations

from typing import cast

import torch
from mamba_ssm.utils.generation import InferenceParams
from torch import Tensor

from lyden.model.block import LydenBlock
from lyden.model.model import LydenModel


class DecodeState:
    """Per-generation cache bundle owned by the caller."""

    def __init__(self, inference_params: InferenceParams) -> None:
        self.inference_params = inference_params
        self.kv_caches: dict[int, dict[str, Tensor]] = {}
        self.query_pos = 0  # absolute position the NEXT token occupies


def _device_of(model: LydenModel) -> torch.device:
    return next(model.parameters()).device


def _attention_indices(model: LydenModel) -> list[int]:
    return [i for i, blk in enumerate(model.blocks) if blk.kind == "a"]


@torch.no_grad()
def prefill(model: LydenModel, token_ids: Tensor) -> tuple[DecodeState, Tensor]:
    """Full forward over (1, T) ids building the caches; returns (state, logits).

    Positions are 0..T-1 (a fresh re-encode of the live context), matching
    the v1 generation loop's per-edit rebuild semantics.
    """
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError(f"prefill expects (1, T) ids, got {tuple(token_ids.shape)}")
    t_total = token_ids.shape[1]
    device = _device_of(model)
    state = DecodeState(
        InferenceParams(
            max_seqlen=t_total,
            max_batch_size=int(token_ids.shape[0]),
        )
    )
    state.kv_caches = {i: {} for i in _attention_indices(model)}
    x = model.emb(token_ids.to(device))
    for i, blk_u in enumerate(model.blocks):
        blk = cast(LydenBlock, blk_u)
        if blk.kind == "a":
            normed = blk.norm1(x)
            mixed = blk.mixer(normed, decode_caches=state.kv_caches[i])
            x_res = x + mixed
            normed2 = blk.norm2(x_res)
            x = x_res + blk.ffn(normed2)
        else:
            x = blk(x, mixer_cache=state.inference_params)
    state.query_pos = t_total
    state.inference_params.seqlen_offset = t_total
    logits = model.head(model.final_norm(x))
    return state, logits


@torch.no_grad()
def step(model: LydenModel, state: DecodeState, token_id: int) -> Tensor:
    """Extend the state by one token; returns its logits (1, vocab)."""
    if state.inference_params.seqlen_offset != state.query_pos:
        state.inference_params.seqlen_offset = state.query_pos
    device = _device_of(model)
    ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    x = model.emb(ids)
    for i, blk_u in enumerate(model.blocks):
        blk = cast(LydenBlock, blk_u)
        if blk.kind == "a":
            normed = blk.norm1(x)
            mixed = blk.mixer.step_branches(  # type: ignore[union-attr, operator]
                normed, state.kv_caches[i], state.query_pos
            )
            x_out = x + mixed
            normed2 = blk.norm2(x_out)
            x = x_out + blk.ffn(normed2)
        else:
            x = blk(x, mixer_cache=state.inference_params)
    state.query_pos += 1
    state.inference_params.seqlen_offset = state.query_pos
    return model.head(model.final_norm(x))


@torch.no_grad()
def greedy_decode(
    model: LydenModel,
    token_ids: Tensor,
    n_new: int,
    eos_id: int | None = None,
    stop_on_specials: set[int] | None = None,
) -> list[int]:
    """Greedy continuation from ids; returns generated ids (excl. prompt)."""
    state, logits = prefill(model, token_ids)
    specials = stop_on_specials or set()
    out: list[int] = []
    for _ in range(n_new):
        nxt = int(torch.argmax(logits[0, -1]).item())
        if nxt in specials:
            break
        out.append(nxt)
        logits = step(model, state, nxt)
    return out


@torch.no_grad()
def sampled_step(
    model: LydenModel,
    state: DecodeState,
    logits: Tensor,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> int:
    """Sample the next id from last-position logits."""
    last = logits[0, -1].float()
    if temperature <= 0.0:
        return int(torch.argmax(last).item())
    last = last / temperature
    if top_k is not None:
        k = min(top_k, last.shape[-1])
        thr = torch.topk(last, k).values[-1]
        last = torch.where(last < thr, float("-inf"), last)
    probs = torch.softmax(last, dim=-1)
    return int(torch.multinomial(probs, 1).item())
