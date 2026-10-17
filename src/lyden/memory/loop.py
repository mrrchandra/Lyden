"""CLM generation loop: full-recompute decode with edit-command handling.

Flow (spec 4.6): after a generation step, check for edit commands; apply them
to the RAM buffer; re-tokenize (v1: full recompute per step/edit); continue
generation with the updated buffer. Editing mid-context never invalidates a
KV cache in v1 because we recompute the whole buffer after each edit.

Design decision (documented): the model's generated PROSE is part of the
live context, so it is folded into the buffer when an edit interrupts
generation (and at the end of decoding). The buffer render is therefore the
final visible text.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from lyden.memory.commands import CLOSE, OPEN, apply_edit, consume_edit_commands
from lyden.memory.context_buffer import ContextBuffer

EDIT_COMPLETION_BUDGET = 96


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
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


def _append(cur_ids: torch.Tensor, nxt: int) -> torch.Tensor:
    return torch.cat([cur_ids, torch.tensor([[nxt]], device=cur_ids.device)], dim=1)


def _ids(buffer: ContextBuffer, tokenizer: object, device: torch.device) -> torch.Tensor:
    ids = [tokenizer.bos_id] + tokenizer.encode(buffer.render())  # type: ignore[attr-defined]
    return torch.tensor([ids], dtype=torch.long, device=device)


def generate_with_clm(
    model: torch.nn.Module,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 50,
    max_edits: int = 4,
    summarizer: Callable[[str], str] | None = None,
) -> tuple[str, ContextBuffer]:
    """Generate while honoring edit commands; returns (final_text, buffer).

    User-written commands in the prompt are consumed first (the buffer holds
    content only). Model-emitted edits are executed against the buffer; their
    result echoes are appended as result lines the model can see.
    """
    device = next(model.parameters()).device
    specials: set[int] = set(tokenizer.special_token_ids().values())  # type: ignore[attr-defined]
    tag_ids: set[int] = {
        tokenizer.special_token_ids()[t]  # type: ignore[attr-defined]
        for t in ("<|edit|>", "<|/edit|>")
    }
    break_on = specials - tag_ids

    rest, cmds = consume_edit_commands(prompt)
    buffer = ContextBuffer(rest.split("\n"))
    for cmd in cmds:
        apply_edit(buffer, cmd, summarizer=summarizer)

    cur_ids = _ids(buffer, tokenizer, device)
    gen_ids: list[int] = []
    ids_left = max_new_tokens
    edits_done = 0

    while ids_left > 0:
        logits = model(cur_ids)
        nxt = _sample_next(logits, temperature, top_k)
        cur_ids = _append(cur_ids, nxt)
        gen_ids.append(nxt)
        ids_left -= 1
        if nxt in break_on:
            break
        seg = tokenizer.decode(gen_ids)  # type: ignore[attr-defined]
        if OPEN not in seg:
            continue

        # An edit command started: force-complete it, then execute.
        if edits_done >= max_edits:
            break
        close_ids: list[int] = tokenizer.encode(CLOSE)  # type: ignore[attr-defined]
        for _ in range(EDIT_COMPLETION_BUDGET):
            logits = model(cur_ids)
            nxt = _sample_next(logits, temperature, top_k)
            cur_ids = _append(cur_ids, nxt)
            gen_ids.append(nxt)
            ids_left -= 1
            if nxt in break_on:
                break
            if gen_ids[-len(close_ids) :] == close_ids:
                break
        seg = tokenizer.decode(gen_ids)  # type: ignore[attr-defined]
        start = seg.find(OPEN)
        close = seg.find(CLOSE, start)
        if start == -1 or close == -1:
            break  # failed to complete a command; stop generation
        cmd_text = seg[start : close + len(CLOSE)]
        _, cmds2 = consume_edit_commands(cmd_text)
        # Visible prose generated before the command folds into the buffer.
        pre_text = seg[:start].strip("\n")
        if pre_text and OPEN not in pre_text:
            buffer.write(pre_text)
        edits_done += 1
        results = []
        for cmd in cmds2:
            results.append(apply_edit(buffer, cmd, summarizer=summarizer))
        for r in results:
            buffer.write(f"[edit-{edits_done}] {r}")
        # Rebuild context from the updated buffer (v1: full recompute).
        cur_ids = _ids(buffer, tokenizer, device)
        gen_ids = []

    # Fold any trailing prose into the buffer and return the full context.
    if gen_ids:
        emitted_text = tokenizer.decode(gen_ids, skip_special=True)  # type: ignore[attr-defined]
        if emitted_text:
            buffer.write(emitted_text)
    return buffer.render(), buffer
