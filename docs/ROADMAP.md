# Lyden Long-Term Roadmap

Recorded 2026-10-19 after M10 (the 100M prototype trained end-to-end on
Kaggle 2×T4). This captures the *scale vision* the user set, on top of the
milestone order fixed in `Prompt.txt` §6.3 — milestones stay the process;
this file is where the "where we are going" ladder lives.

## Confirmed targets (user, 2026-10-19)

1. **Long context:** push the effective context window up through
   2K → 8K → 32K → 128K, final target **256K tokens**.
2. **Scale:** grow the model after the architecture is proven —
   next rungs 250M → 650M–1B → **3B**, stretch goal **7B** final.

## Why the current architecture should carry both

- **Mamba-2 layers** are linear-time / constant-memory and are the global
  carriers; long windows cost almost nothing extra (Mamba lineage has been
  demonstrated at ~1M contexts).
- **Attention islands** stay bounded: sliding window (512 at 100M) +
  summary tokens never blow up with sequence length; only counts and RoPE
  base need tuning per rung.
- **MoE** is the FLOP-saving vehicle for 3B–7B (a 7B-MoE trains like a
  ~2B dense). The M10 lesson: router balance must be fixed/re-validated at
  every scale-up, or experts collapse (M10 saw 2/8 effective experts).
- **CLM memory** stays valuable even at 256K: it handles *unbounded*
  sessions via editable summary RAM, complementing the fixed window.

## Constraints to respect at each scale rung

| Rung | T2×T4 (this tier) | Needs |
|---|---|---|
| ≤1B | OK (weeks of quota) | nothing new |
| 3B | painful | rented A100/H100 (vast.ai-class, days) |
| 7B | no | rented multi-GPU + grouped-GEMM MoE + real corpus |

Also required before 3B+: a **real tokenizer** (~50–64K vocab; the 22,781
TinyStories one is a prototype) and a **real corpus mix** (not only
TinyStories; must include romanized-Hinglish data for delivery targets).

## Milestone ladder (process unchanged)

| M | Piece | Status |
|---|---|---|
| 1–8 | scaffold → tokens/Mamba/attention/hybrid/memory pieces | done |
| 9 | MoE piece + hybrid smoke | done |
| 10 | 100M full-stack on real data (this milestone completed the prototype) | **done** |
| 11 | CLM memory continue-train (1024) + MoE balance fix | next |
| 12 | full-system integration on real data with memory | planned |
| 13 | eval harness / standard benchmarks | planned |
| 14 | docs pass | planned |
| 15 | optimization pass | planned |

### Long-context ladder (its own experiments, one run per hop)

Hop order: **2K → 8K → 32K → 128K → 256K**. Each hop: continue-train from
the previous ckpt, long-document mixture (concatenated/pack corpus),
RoPE base bump, verify with long-range needle/recall probes. Mamba layers
absorb the global load; attention window stays local.

### Scale ladder

Rung order: **250M → 650M–1B → 3B → 7B** (MoE everywhere ≥650M).
Data/tokenizer redo lands before the 650M rung.
