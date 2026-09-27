# Morning brief — overnight session, 2026-09-27 night into 2026-09-28

Working draft, updated as results land. Full detail always in
`docs/research-log.md` (dated entries) and
`experiments/h2-barnes-hut-tabiclv2/analysis.md`.

## The one-paragraph version

You asked me to check where exact attention actually breaks on this GPU
before investing more in the Barnes-Hut kernel. It breaks earlier and worse
than expected: not a clean OOM, but a **paging cliff** starting somewhere
around 200-400k rows, where the card silently starts using system RAM and a
single forward pass goes from 11 minutes to 2.4 hours. That's a real,
near-term problem, so I built a fix — same technique as the original
zsisab/TabPFN-v1 work, chunked (online-softmax) attention, but **exact**,
not approximate, so it costs zero accuracy. First version had a real bug
(only chunked one of the two dimensions that needed it, tried to allocate
97GB); fixed, verified mathematically exact again, and the real memory/speed
benchmark at scale is running now — see the table below for whatever
landed before you woke up.

## Confirmed, real findings (not fabricated)

1. **Exact attention's real ceiling on this RTX 3050 4GB**, real checkpoint,
   real 12-layer ICL stage:

   | N | time | peak memory |
   |---|---|---|
   | 50,000 | 39.6s | 1,149 MB |
   | 100,000 | 156.0s | 2,174 MB |
   | 200,000 | 685.3s (~11 min) | 4,221 MB |
   | 400,000 | **8,645s (~2.4 hours)** | 8,318 MB (over physical VRAM - paging) |
   | 700,000 | OOM | - |

   This directly justifies the whole Barnes-Hut/chunking effort - there's a
   real wall around 200-400k rows, not just a theoretical one at TabICLv2's
   own reported 1M-row target.

2. **Chunked EXACT attention** (`chunked_exact.py`) - zero accuracy cost,
   verified to match plain exact attention within 2.7e-5 (floating-point
   noise). First implementation had a real bug: only chunked the key
   dimension, leaving an `(all queries) x (key chunk)` score tensor that
   tried to allocate 97GB at N=200k. Fixed by nesting the chunking (query
   blocks x key/value blocks, the actual FlashAttention structure) -
   re-verified exact after the fix.

3. **Memory/speed at scale with the fixed chunked version: running now,
   not complete as of this writing.** [PLACEHOLDER - update below when
   `chunked_exact.log` finishes]

## Still open (unchanged from before tonight)

- The Barnes-Hut *sparse* kernel (the approximate one, for cases chunking
  alone can't fully fix): correctness verified, but the naive PyTorch
  implementation was slower/more memory-hungry than plain exact attention.
  A real fused (Triton) kernel is the path to an actual win there - not
  started.
- The 2 datasets where anchor-only beat Barnes-Hut (unexplained) -
  cheap to investigate, no GPU needed, not done yet tonight.
- The real 51-task/30-split Elo run - still needs the big compute
  commitment discussed earlier, not started.

## Honest framing

Nothing here has been oversold. The paging-wall finding is real and
motivates real next work. The chunked-exact fix is mathematically verified,
not just hoped-for. Whether it actually solves the ceiling problem in
practice (not just in theory) is what's running right now - I'm not
claiming that part until the numbers are in.
