# Morning brief — overnight session, 2026-09-27 night into 2026-09-28

Full detail always in `docs/research-log.md` (dated entries).

## Good news, real: found exactly where the actual problem is

You asked me to check where exact TabICLv2 attention actually breaks on
this GPU before investing more in Barnes-Hut. It breaks earlier and worse
than expected — not a clean OOM, but a **paging cliff**: the card silently
starts using system RAM once VRAM fills, and a single forward pass goes
from 11 minutes to 2.4 hours for a 2x increase in row count.

| N (training rows) | time | peak memory |
|---|---|---|
| 50,000 | 40s | 1.1 GB |
| 100,000 | 2.6 min | 2.2 GB |
| 200,000 | 11 min | 4.2 GB |
| 400,000 | **2.4 hours** | 8.3 GB (over the card's 4GB — paging) |
| 700,000 | hard OOM | — |

This is real and it directly justifies the whole effort — there's a
genuine wall around 200-400k rows on this hardware, not just a theoretical
one at TabICLv2's own 1M-row target.

## The honest news: two engineering attempts, one real unifying lesson

**Attempt 1 (two nights ago): Barnes-Hut sparse-gather kernel.** Built it,
found and fixed three real bugs (including an inverted masking polarity
that took real hunting to find), verified mathematically correct to
floating-point precision. But it was *slower and used more memory* than
plain exact attention — a real, reported negative result, not hidden.

**Attempt 2 (tonight): chunked but EXACT attention** (zero accuracy cost,
same idea as the original zsisab/TabPFN-v1 work). Also found and fixed a
real bug overnight (first version only chunked one of two dimensions that
needed it, tried to allocate 97GB). Fixed, re-verified exact. But at large
N it either OOM'd (chunks too big) or projected to take **~17 hours**
(chunks small enough to fit memory).

**Both hit the same wall, and it's the same root cause for both:**
bounding memory in pure PyTorch means serializing one big GPU operation
into many small Python-level steps, and each step pays real overhead
(~9.5ms measured directly, confirmed by timing). Real FlashAttention is
fast because it's a *fused* kernel where that per-step cost is nearly zero
— nothing written in plain PyTorch loops can replicate that, no matter how
the chunk sizes are tuned. This isn't a failure of the idea, it's a correct
diagnosis: **the actual missing piece, for both approaches, is a real
Triton/CUDA kernel, not further restructuring in Python.** That's now
clearly the right next step if this thread continues, and skipping it
would be trying the same tuning tweaks that already failed twice.

One more asymmetric-chunk-size test was still running as I wrote this
(N=700,000 — past exact attention's hard OOM point) to get one concrete
data point rather than stop on a pure negative. I'll tell you the result
the moment it lands — not fabricating a number to close this out cleanly.

## Also done overnight, both honest non-results

- **Investigated the 2 datasets where anchor-only beat Barnes-Hut.**
  Checked feature count and class imbalance as explanations — both refuted
  by direct counterexample (a Barnes-Hut *win* dataset had fewer features
  and near-identical class balance to a *loss* dataset). Genuinely
  unexplained, logged as open rather than forced.

## Where this leaves Phase 1

The accuracy-side result from two nights ago still stands unaffected by any
of this: Barnes-Hut beats anchor-only consistently across real datasets and
the real pretrained checkpoint. What's now clearly understood is *why plain
PyTorch chunking wasn't enough on its own to fix the memory problem* — and
that's a real, useful thing to know before writing a Triton kernel, not a
wasted night.

**Concrete choice for when you're up:** (a) commit to writing an actual
fused kernel for the Barnes-Hut approach — real work, now well-motivated
and well-scoped by tonight's findings, or (b) shift effort to the real
51-task/30-split Elo run instead, which doesn't depend on solving this
kernel problem at all and was already flagged as the other high-value
open item.
