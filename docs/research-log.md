# Research decision log

One entry per decision, newest first. This is the running record of choices
made and why — see `docs/research-direction.md` for the overall strategy
those choices sit inside.

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
