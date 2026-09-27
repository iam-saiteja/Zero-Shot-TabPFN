# ZS-ISAB / TabArena research

**Author:** Thanniru Sai Teja ([@iam-saiteja](https://github.com/iam-saiteja))

Research project targeting a fast, consumer-GPU-runnable tabular foundation
model. This README states current, honest status — see `docs/research-log.md`
for the full decision history (why every pivot happened) and `findings.md`
for the running research narrative.

## Current status (2026-09-27)

**What's validated:** on TabPFN v1 (`tabpfn==0.1.11`), a Barnes–Hut-style
attention approximation — exact attention over the few nearby row-clusters
per query, mass-weighted summaries for the rest — cut the fidelity gap to
exact attention to 0.43x of plain k-means inducing points, losing only
0.33pp accuracy vs exact (vs 3.6–4.9pp for the original random/k-means
anchor schemes). Details: `experiments/h1-barnes-hut-real-activations/`.
Speed is **not yet demonstrated** — the current implementation is a
correctness check, not a fast kernel.

**What's next:** TabPFN v1 is obsolete (2022); it was only ever a cheap
testbed for the attention method. The two active workstreams:

1. **Make an existing open, permissively-licensed model consumer-fast.**
   Target: **TabICLv2** (BSD-3/Apache-2.0, beats RealTabPFN-2.5 untuned on
   TabArena, but needs ~50GB GPU memory at 1M rows). Port the validated
   attention method to its dataset-wise ICL stage. Why this model, and why
   not LimiX-2 (current #1) or TabPFN-3: `research/model-choice.md` — both
   are under non-commercial / research-only licenses.
2. **Try to move the Elo needle**, via fine-tuning (not full pretraining —
   out of reach on the available hardware). Unproven; see
   `research/from-scratch-feasibility.md` and `research/finetune-existing-models.md`.

**What this project does not claim:** current TabArena #1 is ~1935 Elo
(LimiX-2), a number that moves as the benchmark lives on. An attention
approximation cannot score above exact attention on the same weights — it
buys speed and memory, not accuracy. Nothing here has been benchmarked
against the real TabArena leaderboard yet.

## Repository layout

- `zsisab/` — the validated attention-approximation code (chunked online-softmax,
  k-means/Barnes–Hut anchor refinement), patched into TabPFN v1 as a cheap
  testbed. Not the final target — see status above.
- `research/` — landscape survey, model choice, feasibility math, and the
  brainstorm that produced the current lead idea.
- `experiments/` — locked protocols + results for each tested hypothesis.
- `docs/research-log.md` — dated decision log, most recent first.
- `findings.md`, `research-state.yaml` — running project-memory files.
- `legacy/` — superseded v1-era code, benchmarks, and paper drafts. Kept for
  history; see `legacy/README.md` for what's there and why it's retired.

## Setup

```
uv venv .venv           # TabPFN v1 testbed (zsisab/, experiments/h1-*)
uv pip install -r requirements.txt

uv venv .venv-tabicl     # TabICLv2 work
uv pip install --python .venv-tabicl torch --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-tabicl tabicl pandas openml
```
