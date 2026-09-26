# Research decision log

One entry per decision, newest first. This is the running record of choices
made and why — see `docs/research-direction.md` for the overall strategy
those choices sit inside.

## 2026-09-26 — Real web research on TabArena SOTA + from-scratch feasibility math

**Context:** user clarified they understand large-scale synthetic
pretraining can't be matched, but want a math-grounded case for whether
training a small zero-shot model *from scratch* is even feasible on their
actual hardware (4GB RTX 3050 laptop) — and asked for real literature
research, not more guessing. Wrote up in `research/landscape.md` (sourced
survey of current TabArena SOTA — LimiX-2 #1 at 1935 Elo, not 2000+; the
actual fit-time budget is 3600s not the 30s the fabricated doc claimed) and
`research/from-scratch-feasibility.md` (the actual compute-budget math).

**Important finding that changes our framing:** a Sept 2025 paper, "Chunked
TabPFN" ([arXiv:2509.00326](https://arxiv.org/pdf/2509.00326)), already does
*exact* (not approximate) chunked/tiled attention for TabPFN at scale,
training-free, zero accuracy delta. This directly overlaps ZS-ISAB's "fix
TabPFN's OOM problem" pitch — and beats it on accuracy, since it doesn't
approximate. ZS-ISAB's remaining honest differentiator is throughput: O(N·M)
compute vs Chunked TabPFN's O(N²), i.e. faster at extreme row counts at a
controlled accuracy cost — not "we solved the memory problem" (already done,
better, by someone else).

**Feasibility conclusion:** reproducing the original 2022 TabPFN's exact
prior-fitting recipe (160 GPU-hours on 2080Ti-class hardware) on this laptop
serially is ~17 days, not "a few days." Scoping the model down (smaller N
cap, fewer layers, specialize to a narrow regime the way Mitra already does
for <5,000-row tables, mixed synthetic priors, bf16) plausibly brings that
to ~5-10 days — an estimate, not a measurement. Recommended next action is a
short calibration run (minutes-to-hours) to replace the estimate with a real
number before committing a week of the machine's time. Not yet run.

## 2026-09-26 — Implemented Track A: coreset anchor refinement

**Decision:** replaced ZS-ISAB's random inducing-point sampling with a
chunked k-means refinement (`zsisab/engine.py::_refine_inducing_points`),
default `refine_iters=1`, exposed as `inject_zsisab(refine_iters=...)`.
`refine_iters=0` reproduces the old exact behavior.

**Why this over Track B:** smallest diff that reuses the existing eval
scripts (`benchmarks/run_tab_arena.py`, `run_tfm_arena.py`) and existing
zsisab code path — no new infra, no training loop, no new dependency (pure
torch, chunked so it stays O(N*M) and scales to the million-row regime the
project already targets). Track B (lean prior-fitting) needs a real training
budget and is a bigger commitment; worth doing once Track A's numbers are in
and if they justify it.

**Verified so far:** `zsisab/test_engine.py` — synthetic-data self-check
only (confirms the clustering logic itself is correct: reduces distortion,
backward-compatible at `refine_iters=0`, no shape/NaN issues). This is
*not* a TabArena/TabZilla accuracy result yet.

**Status / next action:** not yet benchmarked against real data. Next step
is running `benchmarks/run_tab_arena.py` (or `run_tfm_arena.py`) with
`refine_iters=0` vs `refine_iters=1` (and maybe `3`) on a handful of
datasets to get a real accuracy/time comparison — that result is what
replaces the stale README claims. Not run yet because it needs real
wall-clock GPU/CPU time, not something to fire off silently; flagging here
so the next session picks it up instead of re-deciding from scratch.

## 2026-09-26 — Retired `scratch_architecture_analysis.md`

**Decision:** moved to `scratch/architecture_analysis_UNVERIFIED.md`
(gitignored), out of the real research trail.

**Why:** described "S3T2/SPARK/DART" results with no corresponding
implementation anywhere in the repo — fabricated, not reproducible. Decided
not to build on it or treat its numbers as a baseline.
