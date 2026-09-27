# Is prior-fitting a zero-shot model from scratch feasible on a 4GB RTX 3050?

See `research/landscape.md` for sources. This is the actual math, not a
"simulation" — every number here is either measured (from your machine) or
cited (from a real paper), and every derived number shows its derivation.

## The compute budget that already worked once

The original 2022 TabPFN — which already beat XGBoost/CatBoost/RandomForest
on small tables zero-shot — was prior-fit with:

- 9.2M synthetic datasets, capped at **N ≤ 1024** rows during fitting
- 12-layer transformer
- 18,000 batches × 512 datasets/batch
- **20 hours on 8× RTX 2080 Ti** = 160 GPU-hours (single-GPU-equivalent)

That's the benchmark to reason from, not TabPFN v2's 2-week/8-GPU run or
LimiX-2's 400M-param run — those targeted a much bigger regime (10k rows,
500 features, generalist). We're asking a different, smaller question: can
we reach *160-GPU-hour-class* compute on *this* laptop.

## Hardware comparison (real specs, not guessed)

| | RTX 2080 Ti (orig. TabPFN) | RTX 3050 4GB Laptop (this machine) |
|---|---|---|
| FP32 TFLOPS | ~13.4 | ~7.2 |
| VRAM | 11 GB | 4 GB |
| Memory bandwidth | ~616 GB/s | ~192 GB/s |
| Tensor cores | 2nd-gen | 3rd-gen (better FP16/BF16/INT8 throughput per core) |

Rough per-GPU throughput ratio: the 3050 is **roughly 2-3x slower** in raw
compute, and has **~1/3 the VRAM and ~1/3 the memory bandwidth**. VRAM is
the binding constraint, not raw FLOPs — it caps batch size and max sequence
length (N per synthetic dataset) directly.

## The actual math

Original recipe: 160 GPU-hours across 8 GPUs in parallel (20h wall-clock).
Running the *identical* recipe on 1 GPU at ~1/2.5x the throughput, serially
instead of 8-way parallel:

```
160 GPU-hours × 2.5 (slower chip) ≈ 400 "3050-hours"
400 hours / 24 = ~16.7 days of continuous, unattended compute
```

That's the naive estimate for reproducing the *exact* 2022 recipe. It ignores
that VRAM will force smaller batches (more gradient steps, not necessarily
more wall-clock if you also shrink the model — see below), and ignores
thermal throttling on a laptop chip pinned at 100% for over two weeks, which
is a real risk, not a rounding error.

**Conclusion: "a few days" is optimistic. "Two to three weeks of background
compute" is the honest number for matching the original TabPFN's scale
outright.** That's not a rejection of the idea — two weeks of a laptop
running unattended overnight/while idle is a real, payable cost, just not
the "days" hoped for.

## Getting back toward "days": scope down, don't just wait longer

The lever that actually works is not "wait longer," it's **train a smaller,
narrower model that needs less of that 9.2M-dataset budget to converge**,
using ideas the field has already validated (cited in `landscape.md`), not
invented for this document:

1. **Cap N lower than 1024** (say 256-512) and use fewer layers (6-8, not
   12). Attention cost is superlinear in N, so this is the single biggest
   lever — cutting effective context in half can cut per-step cost by more
   than half.
2. **Specialize instead of generalize** — Mitra's own reported niche is
   "<5,000 rows, <100 features" and it's genuinely competitive there at
   72-75M params. A much smaller model narrowly targeting, say, "≤500 rows,
   ≤50 features, binary/multiclass classification only" needs to model a far
   smaller function class than a generalist. Smaller target ⇒ fewer
   synthetic tasks needed to cover it ⇒ less compute. This is the same logic
   as choosing a research topic scoped to what's provable, not the whole
   field.
3. **Mixed synthetic priors (Mitra's finding), not more of one prior** — diversity
   of data-generating mechanisms improved generalization more than raw
   dataset count did, in their ablations. This is a *free* change (better
   sampling recipe, not more compute) — implement several distinct
   generators (linear SCMs, tree-based SCMs, mixture-of-Gaussians clusters,
   at least one with genuine categorical/discrete structure) instead of one.
4. **Self-supervised auxiliary loss on top of real data (TabDPT's finding)** —
   mixing in a modest amount of real tabular data (e.g. the already-downloaded
   TabZilla-168 suite sitting in `datasets/`) with a column-masking
   self-supervised objective reportedly speeds convergence and improves
   downstream accuracy vs synthetic-only. Free real training signal you
   already have on disk.
5. **Bake row-compression into pretraining, don't bolt it on after** — this
   is where our own `zsisab/engine.py` chunked online-softmax kernel earns
   its keep: use the *same* mechanism, but as part of the model trained from
   scratch (like TabICL's row-transformer does), not patched onto a frozen
   backbone post-hoc. Trained jointly, the model's weights adapt to the
   compression instead of an already-fixed backbone approximating a mechanism
   it was never trained for — this closes the accuracy gap Track A's k-means
   anchors can only narrow, not close. It also directly reduces *training*
   VRAM, which is the actual bottleneck on this machine.
6. **bf16 mixed precision + gradient checkpointing** — standard, free
   (Ampere tensor cores support it natively), roughly halves activation
   memory, letting you afford either a bigger batch or a longer N within the
   4GB budget.

Combining (1)+(2) alone plausibly cuts required synthetic-task count and
per-step cost by something like 4-8x versus the unscoped original recipe —
enough to bring the wall-clock estimate from ~17 days down to roughly
**5-10 days** of unattended background compute. That's an estimate with
real assumptions stated above, not a measurement — the only way to know the
real number is to actually run a short calibration (see below), not to trust
this arithmetic blindly.

## What this can honestly claim, and what it can't

**Can't claim:** overall #1 on TabArena, or competing with 400M-parameter
LimiX-2 or TabPFN-3.5 trained on datacenter clusters. That gap is real and
this doesn't close it.

**Can honestly claim, if it works:** a small (single-digit-million to
low-tens-of-millions parameter), narrowly-specialized, from-scratch
zero-shot model, trained entirely on a 4GB consumer laptop in roughly a
week, that is competitive *within its declared niche* (small row/feature
counts) — a genuinely different, defensible, citable result even without
touching the overall leaderboard top. That mirrors exactly what Mitra
already demonstrated is a real, publishable niche, just at a much smaller
training budget.

## Before committing a week of compute: run a calibration first

Don't start the full run blind. Cheap, fast checks that turn the estimates
above into real numbers, each takes minutes to hours, not days:

1. Time 100 synthetic-dataset generation + 50 training steps at the target
   scoped-down size (N=256-512, 6-8 layers) on this GPU. Extrapolate real
   steps/hour instead of the FLOPs-ratio guess above.
2. Confirm the scoped model + chosen batch size actually fits in 4GB with
   bf16 + gradient checkpointing before queuing a multi-day run.
3. Train for a few hours, evaluate on a handful of small TabZilla datasets
   already in `datasets/`, and sanity-check it's learning *something*
   (beats a trivial baseline) before trusting it to run unattended for days.

That calibration is the actual next action, not writing more theory. It's
the only thing that turns the estimate above from arithmetic into a decision
you can trust with a week of your machine's time.

## Update: more compute is actually available — revised math

Two more resources came up: possible access to a colleague's **24GB RTX
3090 Ti**, and **Kaggle's free tier** (~30 GPU-hours/week, historically
offered as 2×T4 — each T4 has 16GB VRAM, more than the 11GB 2080Ti the
original TabPFN was trained on; verify current quota/accelerator options on
Kaggle before relying on this, they change the offering over time).

Revised estimates for the *scoped-down* recipe (smaller N cap, fewer
layers, mixed priors — same scoping as above, roughly 4-8x cheaper than the
unscoped original):

| Resource | Rough wall-clock for the scoped recipe | Needs permission? |
|---|---|---|
| This laptop (RTX 3050, 4GB) | ~5-10 days continuous | No |
| Kaggle free tier (~30 GPU-hrs/week, T4-class) | ~1-2 calendar weeks, spread across sessions with checkpointing (session length is capped, historically ~9-12h continuous — plan for resume-from-checkpoint, not one unbroken run) | **No** |
| 24GB RTX 3090 Ti (colleague's) | ~1.5-3 days continuous (roughly 2.5-3x this laptop's throughput, and no VRAM-driven batch-size penalty) | Yes |

**Sequencing recommendation, cheapest-first:** run the calibration (above)
and initial scoped training on the laptop + free Kaggle quota — needs no
one's permission, costs nothing, and is what actually tells you whether the
recipe converges to something better than a trivial baseline at all. Only
ask a colleague for the 3090 Ti once you have that real evidence in hand —
asking for a favor backed by "here's a partial run that's already beating
CatBoost on our target niche" is a very different conversation than asking
on the strength of a theory doc.
