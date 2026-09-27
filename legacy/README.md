# Legacy

Superseded material, kept for history, not deleted. Nothing here is
maintained or trusted as current. See `docs/research-log.md` at the repo
root for why each pivot happened.

- `tabarena_submission/` — TabArena submission scaffold built around the old
  TabPFN v1 + `zsisab` wrapper. Useful as a template for a future real
  submission, once there's a model worth submitting.
- `benchmarks/` — one-off scripts written to validate v1 zsisab's claims
  (several literally named `verify_claims*.py`). Superseded because those
  claims can't be re-verified (backing raw-result zips were removed) — see
  `docs/research-log.md`, 2026-09-26 cleanup entry.
- `paper/`, `paper-draft-isab/` — old paper drafts (LaTeX + Markdown) for the
  v1-era ZS-ISAB work. Numbers in them are the same unverified claims above;
  don't cite them.
- `assets/`, `tfm_leaderboard.json` — plots/data backing the old paper and
  README claims.
- `run_official_tabarena_lite.py`, `run_official_beyondarena_lite.py` — v1
  zsisab TabArena-Lite runners.

`zsisab/` at the repo root is *not* here — it's still the active pilot
testbed (see `experiments/h1-barnes-hut-real-activations/`), kept live
because H1's results depend on it.
