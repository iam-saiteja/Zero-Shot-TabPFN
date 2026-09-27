# Real TabArena eval harness (borrowed 2026-09-27)

Copied from `Documents/tabarena` (your fork of `iam-saiteja/tabarena`,
branch `add-zsisab`, preserved and pushed there before that folder was
deleted per your instruction — full history is on GitHub if anything here
needs re-checking against it).

- `tabarena/` — the actual `tabarena` package (`packages/tabarena` in the
  source repo): real model connectors for every model in
  `research/landscape.md` (TabICL, Mitra, TabPFN-3, LimiX, TabDPT, ...) plus
  your own `models/zsisab` and `models/zstabfm` integrations, and a working
  `models/s3t2` implementation (the S3T2 that `scratch/architecture_analysis_UNVERIFIED.md`
  in this repo described — it isn't fabricated, it exists as real code there,
  I just didn't know about this repo when I called it unverified).
- `bencheval/` — the real evaluation/leaderboard/Elo computation library
  this harness uses.
- `tabarena_tiny_results_reference/` — a real local eval run mixing genuine
  cached official TabArena baselines (from the project's S3/R2 result cache)
  with a local ZSTABFM run, on an 18-task "tiny" subset (not the full
  51-task suite). Kept as reference: `tabarena_leaderboard.csv` there shows
  ZSTABFM at **Elo 1866.8, average rank ~7.4** — behind TABFM (1944.6) and
  TabFM+ (1935.3) in that run. A commit message in the source history
  claimed this run "achieve[d] #1 Rank" — the CSV in the same results folder
  doesn't support that; noted here so the discrepancy isn't lost.

## Not yet done

Not installed or test-run in this repo yet (needs its own dependency set —
likely heavier than `.venv-tabicl`, given `bencheval`/`tabarena` pull in
AutoGluon). Next step before using this for real: `uv venv .venv-harness`,
install `packages/tabarena` and `packages/bencheval` in editable mode, and
confirm a small eval still reproduces the reference numbers above before
trusting any new result from it.
