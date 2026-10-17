#!/usr/bin/env python
"""Greedy / sampled generation from a trained checkpoint (v1: full recompute).

Usage:
  /home/madrin/venvs/lyden/bin/python scripts/generate.py \
      --config configs/nano.yaml --ckpt runs/nano/ckpt_5000.pt \
      --prompt "Once upon a time" --max-new-tokens 120 [--temperature 0.8] [--top-k 50] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from lyden.config import LydenConfig, load_config  # noqa: E402
from lyden.data.tokenizer import LydenTokenizer  # noqa: E402
from lyden.memory import generate_with_clm  # noqa: E402
from lyden.model.model import LydenModel  # noqa: E402


def load_model(config_path: str, ckpt_path: str, device: str) -> tuple[LydenModel, LydenConfig]:
    cfg = load_config(config_path)
    model = LydenModel(cfg.model).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    step = payload.get("step", "?")
    print(f"[load] {ckpt_path} (step {step})")
    return model, cfg


@torch.no_grad()
def generate(
    model: LydenModel,
    ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    device = next(model.parameters()).device
    out = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(out)  # (1, T, V): full recompute every step (v1)
        logits = logits[0, -1].float()
        if temperature <= 0.0:
            nxt = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                thr = torch.topk(logits, k).values[-1]
                logits = torch.where(logits < thr, float("-inf"), logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        out = torch.cat([out, torch.tensor([[nxt]], device=device)], dim=1)
    return out[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clm", action="store_true", help="enable CLM editable memory")
    parser.add_argument("--max-edits", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    model, cfg = load_model(args.config, args.ckpt, device)
    tok = LydenTokenizer.load(cfg.data.tokenizer_dir)

    prompt = args.prompt
    if prompt:
        ids = tok.encode(prompt, add_bos=True)
    else:
        ids = [tok.bos_id]
    print(f"[prompt] {prompt!r}")
    if args.clm:
        text, buffer = generate_with_clm(
            model,
            tok,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_edits=args.max_edits,
        )
        print("[clm buffer]")
        print(text)
        (Path(cfg.training.out_dir) / "clm_buffer.json").write_text(
            str(buffer.state_dict()), encoding="utf-8"
        )
        return
    out_ids = generate(
        model,
        ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tok.decode(out_ids, skip_special=True)
    print("[output]")
    print(text)


if __name__ == "__main__":
    main()
