# Medusa Compute

<img width="1600" height="563" alt="Medusa Compute" src="https://github.com/user-attachments/assets/c00340dc-35f1-4fc7-9dee-d4671a39f663" />

**LLM-driven automatic TPU kernel optimization, built on JAX and Pallas.**

Medusa Compute closes the gap between raw TPU silicon and the performance it actually delivers. Today that gap is paid in hand-written kernels — slow to author, hard to port, always behind the latest model architectures. Our bet: **kernel writing is the right level for LLMs to operate at**, and Pallas + JAX give us the surface to make that loop work.

## What we work on

- **TPU kernels in Pallas / JAX** — fused attention, linear-attention recurrences (GDN / Mamba-style), embedding gathers on SparseCore, jagged-tensor pipelines for recommendation workloads.
- **Automatic kernel optimization with LLMs** — agent loops that author, mutate, profile, and rank Pallas kernels against measured TPU performance, not just static heuristics.
- **Hardware-honest benchmarking** — apples-to-apples H100 vs TPU v6e measurements (matched serving flags, prefix-caching disabled, per-cell medians) so that kernel wins are not confounded by serving-stack differences.
- **SparseCore for recommendation systems** — establishing SC as the right engine for embedding lookup on TPU, then scaling that result to multi-chip and to next-gen ICI-direct DMA paths.

## Highlighted results

*All measurements on a single Google TPU v6e (918 TFLOPS BF16, 32 GB HBM).*

- **HSTU Attention — 1.16–1.69× over the XLA auto-generated kernel.** Our optimized kernel also handles large-N configs (≥64 GB) that cause the XLA baseline to OOM, lifting the practical scale ceiling for HSTU on a single chip.

- **Splash Attention — 1.15–1.38× over JAX Splash Attention (MaxText).** Improved tiling and on-chip data flow squeeze additional headroom out of an already heavily tuned reference kernel.

- **Llama-3.1-8B prefill (end-to-end) — 1.01–1.10× over the JAX/MaxText baseline.** The speedup grows with context length, because attention's share of prefill compute rises from 2% at N=512 to 21% at N=8192 — exactly where our attention kernel pays back.

- **Linear Attention — 1.27–1.67× over the XLA auto-generated kernel.** Linear-attention recurrences are the canonical "non-standard op" that XLA cannot lower well; a hand-built Pallas kernel reclaims most of the gap in one shot.

## Roadmap

1. **Scale from single-chip to multi-chip.** Carry our kernel wins onto TPU pods with sharding-aware kernels and collective-friendly data flow.
2. **Close the loop on automatic kernel optimization.** Make the Medusa LLM-driven pipeline produce kernels that match or beat hand-tuned baselines, end-to-end.
3. **Bring it to the next generation of TPU hardware.** Stay on the frontier as new accelerators ship.

## Stack

JAX · Pallas · TPU v6e (Trillium) today, next-generation TPU tomorrow. Cross-hardware comparison runs on vLLM with matched flags on both sides.
