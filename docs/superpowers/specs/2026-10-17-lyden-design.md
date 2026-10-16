# Lyden — Hybrid SSM–Attention–MoE Language Model with CLM Memory

**Status**: Approved design (2026-10-17). Source spec: `Prompt.txt`.

## 1. Mission

A from-scratch, trainable 100M–500M parameter language model in PyTorch combining:

1. **SSM backbone** (Mamba-2) — linear-time, constant-memory sequence mixing.
2. **Hybrid attention islands** — sliding-window + compressed-global + selective top-K, gated.
3. **MoE feed-forward** — 32 routed + 2 shared experts, top-2, on attention blocks only.
4. **CLM editable context memory** — RAM-only buffer the model edits via textual commands.

Runs on a single consumer GPU (validated on 6GB RTX 4050 via a nano config; full configs target 8–24GB GPUs). Correctness, clarity, reproducibility > benchmark numbers.

## 2. Hard constraints (from source spec)

- PyTorch only. Trainable from random init, no pretrained checkpoints.
- CLM memory is RAM-only, no vector DB, no external retrieval, no separate service.
- Every run seedable + reproducible; config, seeds, env logged.
- Interleaving 1 attention : 3 mamba (Jamba-style). Mamba blocks get dense FFN; attention blocks get MoE FFN. No MoE on Mamba blocks. No MLA. No learned absolute positional embeddings. RoPE on attention branches only.

## 3. Architecture

```
tokens → embedding (32K × 768, tied head) → [layer_pattern "mmma" × repeat]
   m: RMSNorm → Mamba-2 → residual → RMSNorm → dense SwiGLU FFN → residual
   a: RMSNorm → HybridAttention → residual → RMSNorm → MoE FFN → residual
→ final RMSNorm → LM head (tied, no bias) → next token (+ optional CLM edit command)
```

### 3.1 Mamba block (`src/lyden/model/mamba_block.py`)
- Official **`mamba-ssm`** package (Mamba-2, selective scan CUDA kernel) behind a thin `MambaBlock` API exposing train (parallel scan) / infer (recurrent) modes, var-length support, config: `d_state=128, d_conv=4, expand=2`.
- No custom SSM implementation (explicit user directive). No MoE, no attention in Mamba blocks.

### 3.2 Hybrid attention block (`src/lyden/model/hybrid_attention.py`)
Three parallel branches over a **shared GQA K/V projection** (8 query heads, 2 KV heads), each branch has its own Q projection and attention logic; a learned gate (query-conditioned softmax over 3 logits) combines branch outputs:

1. **Sliding window** (W=512): causal windowed attention via SDPA + band mask.
2. **Compressed global**: G=64 learnable summary queries, anchor i at position `i·(T/G)`; summary i attends to tokens ≤ anchor; token t attends to summaries with anchor ≤ t (causally consistent in both train and cached decode; summaries computed at prefill, refreshed on cache rebuild).
3. **Selective fine-grained**: learned indexer (small separate projection) scores token importance, hard top-256 per query (causal-masked before top-k); gradients flow via gathered values; optional ranking aux loss (off by default).

RoPE (base 10K) applied to queries and keys in all branches, shared. No dense attention default, no MLA.

### 3.3 MoE (`src/lyden/model/moe.py`)
Top-2 of 32 routed experts + 2 always-active shared experts. Router: `Linear(dim, n_routed)`. Load balancing: DeepSeek-V3-style **loss-free bias adjustment** (default) and Switch-style aux loss (config-selectable, weight 0.01). Experts: SwiGLU MLP, hidden 1024, loop-over-experts v1. No capacity limits by default.

### 3.4 CLM memory (`src/lyden/memory/`)
- `ContextBuffer`: in-RAM list of text lines. Ops: READ / WRITE / REPLACE / DELETE / SUMMARIZE.
- `EditEngine`: parses `<|edit|>OP args<|/edit|>` from generated text, applies to buffer.
- `clm.py`: generation-loop integration — after each step, scan for edit commands, apply, recompute KV (v1: full recompute; documented optimization path).
- SUMMARIZE = the model summarizes the span via its own `generate()` with a summarization prompt (zero-shot, no extra model).
- Buffer injected into the prompt as `<|context|>…lines…<|/context|>`. Zero-shot only in v1; RL policy training is a stretch goal (out of scope).

## 4. Training recipe

- **Data**: TinyStories (~2GB) for local validation; `prepare_data.py` implements the real recipe (99% FineWeb-Edu + 1% Cosmopedia-v2, streamed, pre-tokenized shards). HF `tokenizers` BPE, vocab 32K, packed var-length sequences, shard-level shuffle. Seq len 2048 (up to 8192).
- **Optimizer**: AdamW, betas (0.9, 0.95), weight decay 0.1 (0.0 on norms/biases), grad clip 1.0, **split LR**: experts + routers + branch gates at base LR, everything else (trunk) at 0.1×.
- **Schedule**: linear warmup 2000 steps → cosine to 10% of max. Grad accum 4. 100K steps default for 250M.
- **Precision**: torch.amp bfloat16 (fp16 + loss-scale fallback); optimizer state fp32; SSM state fp32.
- **Checkpointing**: every 5000 steps, keep last 3; saves optimizer, scheduler, RNG, config, tokenizer. Resume supported.
- **Logging**: loss, LR, grad norm, tokens/sec, VRAM; MoE routing stats (expert load, entropy, active experts); attention branch gate values; stability warnings (expert collapse >50%, SSM state norm growth, gate collapse, loss spikes). TensorBoard default, backend configurable.

## 5. Configs

Nested dataclasses loaded from YAML. One schema, four sizes:

| Config | dim | layers | pattern | d_state | experts | ~Params | Purpose |
|--------|-----|--------|---------|---------|---------|---------|---------|
| nano   | 256 | 8      | mmma    | 64      | 8+2     | ~5M     | CI/smoke on any machine |
| 100m   | 640 | 16     | mmma    | 128     | 16+2    | ~100M   | local (6GB) w/ small batch + accum |
| 250m   | 768 | 24     | mmma    | 128     | 32+2    | ~275M   | default, single consumer GPU 8GB+ |
| 500m   | 1024| 32     | mmma    | 128     | 32+2    | ~500M   | larger GPU |

`scripts/param_count.py` prints exact per-component params from any config (the source spec's table is approximate; we target ~275M total for the 250m config).

## 6. Environment

- **WSL2 Ubuntu 26.04** (explicit user directive; virtualization feature enabled 2026-10-17, reboot applied). GPU passthrough verified: RTX 4050 Laptop 6GB.
- **uv** 0.13.1 (WSL) manages Python 3.12 + venv at `~/venvs/lyden` (WSL fs for I/O speed; repo stays on `/mnt/c` for Windows visibility; UV_PROJECT_ENVIRONMENT set accordingly).
- torch CUDA wheels; **mamba-ssm + causal-conv1d** installed via matching prebuilt wheels, fallback = source build with CUDA toolkit (reported if it happens).
- pytest, ruff, mypy; TensorBoard for logging.

## 7. Testing strategy

- Unit: every component (shape, grads finite, var-length, batch>1).
- Integration: full fwd/bwd, one training step on synthetic data, one cached generation step, one CLM edit.
- Smoke: nano model 100 steps — loss decreases; checkpoint save/load/resume.
- Stability: extreme inputs → no NaN/Inf; SSM state norms bounded; gate values in [0,1].
- Eval: held-out perplexity, routing + gate stats, optional lm-eval subset.

## 8. Milestones (spec §6.3 order, stop-and-report at each)

1. Scaffold + config system + logging + env ← current
2. Tokenizer + data pipeline
3. Mamba block (mamba-ssm wrapper)
4. Dense FFN
5. Mamba-only nano model trains
6. Hybrid attention block (3 branches, individually tested)
7. Attention-only nano model trains
8. Hybrid interleaved model trains
9. MoE (routing, load balancing, expert stats)
10. Full hybrid + MoE
11. CLM memory layer
12. Full system on real data (TinyStories 100M locally)
13. Eval harness
14. Documentation pass (README/ARCHITECTURE/TRAINING/CLM)
15. Optimization pass (profile, chunked SSD if beneficial, cleanup)

## 9. Known limitations & risks

- 6GB VRAM: local ceiling is the 100m config with small batch + grad accum; 250m/500m validated by instantiation + param counts, trained on larger GPUs later.
- Pure-PyTorch-free Mamba via mamba-ssm ties us to CUDA (fine under WSL2; documented portability note).
- Performance targets (>10K tok/s on 3090, <50ms/token, <100ms/edit) are stated for larger GPUs; we report actual local numbers with explanation.
- Hard top-K selection is non-differentiable through indices (values carry gradients); aux ranking loss available if recall quality demands it.
- Compressed branch summary bank is anchored to prefill length (v1); NSA-style block compression is the upgrade path.

## 10. Explicitly out of scope (v1)

Diffusion decoder head, RL memory policy training, multi-GPU training, distributed inference, web UI/API, multimodal. Each has an extension path but must not block the core.
