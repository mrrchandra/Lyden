"""Evaluation harness: validation perplexity, long-context extrapolation,
and sample generation (M13 scope "standard benchmarks"; M10 ships ppl +
samples; HellaSwag-style MCQ scoring lands later with data provisioning).

Val perplexity reuses the trainer's val batcher construction (identical
packing/batching), so numbers are directly comparable to training val loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from lyden.config import LydenConfig
from lyden.data.dataloader import PackedBatcher, ShardDataset, SyntheticDataset
from lyden.model.model import LydenModel


@dataclass
class _ValSetups:
    def __init__(self, cfg: LydenConfig) -> None:
        self.cfg = cfg


def _val_batcher(cfg: LydenConfig, seq_len: int, batch_size: int, seed: int) -> PackedBatcher:
    if cfg.data.dataset == "synthetic":
        val_ds: SyntheticDataset | ShardDataset = SyntheticDataset(
            cfg.model.vocab_size, seed=seed + 1, tokens_total=1_000_000
        )
    else:
        shards = Path(cfg.data.data_dir).expanduser() / "shards"
        val_ds = ShardDataset(shards, "val")
    return PackedBatcher(val_ds, seq_len=seq_len, batch_size=batch_size, seed=seed, shuffle=False)


@torch.no_grad()
def perplexity(
    model: LydenModel,
    cfg: LydenConfig,
    n_batches: int = 40,
    seq_len: int | None = None,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Held-out ppl over packed val batches at the training geometry."""
    sl = seq_len or cfg.training.seq_len
    bs = batch_size or min(4, cfg.training.batch_size)
    batcher = _val_batcher(cfg, sl, bs, cfg.seed)
    model.eval()
    total = 0.0
    tokens = 0
    for _ in range(n_batches):
        x, y = batcher.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(
            device_type="cuda" if x.is_cuda else "cpu",
            dtype=torch.bfloat16,
            enabled=x.is_cuda,
        ):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum"
            )
        total += float(loss)
        tokens += y.numel()
    nll = total / tokens
    return {
        "loss": nll,
        "ppl": math.exp(nll),
        "loss_tokens": float(tokens),
    }


@torch.no_grad()
def long_context_ppl(
    model: LydenModel,
    cfg: LydenConfig,
    lengths: tuple[int, ...] = (512, 1024, 2048),
    n_batches: int = 4,
    device: str | torch.device = "cuda",
) -> dict[str, dict[str, float]]:
    """PPL as a function of packed context length (extrapolation probe).

    seq_len always divides the val stream position so different lengths read
    the same token stream start (shuffle off); batches are not comparable
    across lengths in tokens, but ppl is length-invariant per token.
    """
    out: dict[str, dict[str, float]] = {}
    for sl in lengths:
        if sl > cfg.training.seq_len and (sl % cfg.training.seq_len) != 0:
            raise ValueError(f"unsupported probe length {sl}")
        res = perplexity(model, cfg, n_batches=n_batches, seq_len=sl, batch_size=1, device=device)
        out[str(sl)] = {"loss": res["loss"], "ppl": res["ppl"]}
    return out


@torch.no_grad()
def sample_text(
    model: LydenModel,
    tokenizer: object,
    prompt: str,
    n_new: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 50,
    seed: int = 0,
) -> str:
    """Stateful greedy/sampled continuation; returns plain text."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    from lyden.model.decode import prefill, step

    ids = [tokenizer.bos_id] + tokenizer.encode(prompt)  # type: ignore[attr-defined]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    state, logits = prefill(model, x)
    gen: list[int] = []
    eos = tokenizer.eos_id  # type: ignore[attr-defined]
    for _ in range(n_new):
        last = logits[0, -1].float()
        if temperature <= 0.0:
            nxt = int(torch.argmax(last).item())
        else:
            last = last / temperature
            if top_k is not None:
                k = min(top_k, last.shape[-1])
                thr = torch.topk(last, k).values[-1]
                last = torch.where(last < thr, float("-inf"), last)
            nxt = int(torch.multinomial(torch.softmax(last, -1), 1).item())
        if nxt == eos:
            break
        gen.append(nxt)
        logits = step(model, state, nxt)
    return tokenizer.decode(gen, skip_special=True)  # type: ignore[attr-defined]


# ----------------------------------------------------------- M11: fact recall
@torch.no_grad()
def fact_fix_recall(
    model: LydenModel,
    tokenizer: object,
    sessions: list[tuple[str, str]],
    device: str | torch.device = "cuda",
    n_new: int = 24,
) -> dict[str, float | int]:
    """Deterministic (T=0) fact-fix recall over held-out CLM sessions.

    For each (prompt, target): generate ``n_new`` tokens from the prompt and
    score the expected corrected-fact answer as present in the decoded text.
    Returns {"recall": fraction correct, "n": sessions, "matches": hits}.
    """
    hits = 0
    for prompt, target in sessions:
        text = sample_text(model, tokenizer, prompt, n_new=n_new, temperature=0.0, seed=0)
        if target in text:
            hits += 1
    return {
        "recall": hits / len(sessions) if sessions else 0.0,
        "n": len(sessions),
        "matches": hits,
    }
