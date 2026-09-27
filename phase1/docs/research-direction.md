# Research direction: ZS-ISAB going forward

Written 2026-09-26. See `docs/research-log.md` for the running log of which
choice got picked and why, updated as work progresses. Replaces `scratch_architecture_analysis.md` (moved to
`scratch/architecture_analysis_UNVERIFIED.md`, gitignored) — that doc
described "S3T2/SPARK/DART" results with no corresponding code anywhere in
the repo. Nothing here reuses those numbers.

## The two goals are in tension

Goal as stated: reach #1 on TabArena (~2000+ Elo), zero-shot, **without**
pretraining on millions of synthetic datasets, while using less compute than
everything else.

Every model currently near the top of TabArena's zero-shot/foundation-model
tier (TabPFNv2, TabICL, TabDPT, Mitra, ...) gets there specifically *because*
of large-scale synthetic prior-fitting — the pretraining is the mechanism
that gives them in-context generalization to unseen tabular tasks at all.
"Zero-shot and competitive" and "no large-scale synthetic pretraining" are
not two independent knobs — the first is currently a downstream effect of the
second. I'm not aware of any published zero-shot tabular model that reaches
that tier without it, and I don't have a mechanism to promise otherwise. So:
literal #1 with zero prior-fitting is not a plan I can respect and hand you
— it would need a real algorithmic breakthrough, not an engineering pass.

What's actually available to you, using the codebase you already have:

## Track A — compute/accuracy Pareto frontier (real, buildable now)

This is the genuine differentiator ZS-ISAB already has: nobody on the
leaderboard optimizes for *rows-per-second-per-GB-VRAM at fixed accuracy*.
Current implementation picks its M=512 inducing points via **random seeded
sampling** (`zsisab/engine.py`) — the known weak point of ISAB (Lee et al.
2019): random anchors throw away information a data-dependent summary
would keep.

Concrete next step (small, testable, no training-loop changes needed):
replace random anchor sampling with a cheap coreset/k-means-style summary of
each incoming chunk before it hits the inducing-point attention. This is a
few hours of work against existing code, and the existing benchmark scripts
(`benchmarks/run_tab_arena.py`, `benchmarks/run_tfm_arena.py`) already give
you a real accuracy-vs-baseline comparison — no new infra required.

This targets a defensible, narrow claim: *best accuracy-per-VRAM among
zero-shot tabular models at extreme row counts* — not overall #1, but a real
point nobody else is contesting.

## Track B — lean prior-fitting (moves accuracy, still cheap)

Instead of "millions" of synthetic datasets, prior-fit on a much smaller
synthetic-task budget (thousands, not millions — feasible in single-digit
GPU-hours) *on top of* the ZS-ISAB chunked attention, so the prior-fitting
itself inherits the memory savings. This won't reach frontier-model Elo, but
it's a real, honest way to move ZS-ISAB's accuracy up from "matches vanilla
TabPFN with less memory" toward "measurably better than vanilla TabPFN,"
which is a publishable result even without beating GPT-of-tabular-data
incumbents.

`create_synthetic_dataset.py` already exists in the repo — worth checking
what it currently generates before building this out further.

## What I won't do

Produce another document with invented benchmark numbers. Every number that
goes in the paper or README from here should trace to an actual run of
`benchmarks/run_tab_arena.py` / `run_tfm_arena.py` you or I execute.

## Suggested next concrete step

Pick Track A first — it's the smaller diff, reuses existing eval scripts,
and gives you a real, current number to replace the stale README claims
(whose backing raw-result zips are no longer in the repo). Say the word and
I'll implement the coreset anchor swap and run it against a handful of
TabZilla datasets for a real before/after comparison.
