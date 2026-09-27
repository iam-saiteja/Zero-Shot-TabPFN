# Research decision log

One entry per decision, newest first. This is the running record of choices
made and why — see `docs/research-direction.md` for the overall strategy
those choices sit inside.

## 2026-09-27 — Repo reorg into legacy/; model choice locked to TabICLv2

User asked for two parallel workstreams (make an open, permissively-licensed
model consumer-fast; separately try fine-tuning toward a better Elo), and to
clean the repo: move superseded v1-era material into `legacy/`, verify the
license before committing to a model (their "major focus"), and push the
result straight to `main` under their own authorship only (no co-author
line, per their explicit instruction, overriding the default attribution
convention for this push).

**License check (`research/model-choice.md`):** LimiX-2 (current #1) and
TabPFN-3/3.5 are both non-commercial / research-only licensed — excluded.
TabICLv2 (BSD-3/Apache-2.0) and Mitra (Apache-2.0) are the fully open
options. **Chosen: TabICLv2** — strongest fully-open model, and it has a
real unsolved efficiency gap (~50GB GPU at 1M rows) that the validated H1
attention method should address directly, since its dataset-wise ICL stage
has the same one-token-per-row structure H1 was tested on. Mitra kept as
fallback for the fine-tuning track (smaller, already small-table-specialized).

**Repo reorg:** moved `tabarena_submission/`, `benchmarks/`, `paper/`, the
old paper draft, `tmlr-style-file-main/`, `tabzilla/`, `assets/`,
`tfm_leaderboard.json`, and the old TabArena-Lite runner scripts into
`legacy/` (git history preserved via `git mv` where tracked). All of it was
built around the v1 zsisab claims that can't be re-verified (2026-09-26
entry). README rewritten to state current status honestly instead of the
old unverified benchmark table. `zsisab/` stays at the repo root — it's
still the active pilot testbed H1 depends on.

**Set up `.venv-tabicl`** (via `uv`, per user preference) with `tabicl` +
torch+cu121 for the port work.

## 2026-09-27 — Base model must move off TabPFN v1; TabICLv2 is the port target

User (rightly) flagged that v1 is obsolete. Checked the current field:
- **TabPFN-3** (53M cls params, 24 layers) already ships row-chunking + multi-query
  attention: ~7GB KV cache at 1M rows, sub-second inference on one GPU. So "scale to
  1M rows on one GPU" is no longer an open niche. Weights are research/internal-eval
  only (no commercial or production use, derivatives included), so it is a poor base
  for something meant to be released.
- **TabICLv2** is open including pretraining code, beats RealTabPFN-2.5 on TabArena
  untuned, but needs ~50GB GPU (with disk offload) at 1M rows, and its dataset-wise ICL
  stage is O(n²) in rows. Pretraining cost ~24.5 H100-days (Stage 1 alone ~20), so
  from-scratch pretraining on consumer GPUs is out of reach; fine-tuning/porting is not.
- Its dataset-wise ICL stage is one token per row, the same structure as v1, so the
  Barnes–Hut module ports there directly. The consumer-GPU gap (50GB at 1M rows) is a
  real, open problem.

**Decision:** keep the v1 pilot only as a cheap test of whether embeddings cluster; then
port the attention module to TabICLv2's dataset-wise stage. Elo caveat stays: an
approximation cannot exceed exact attention, so this buys speed/memory, not Elo. Elo
gains would have to come from fine-tuning (open question). Current #1 is ~1935, so 2K+
is not supported by any evidence found.

## 2026-09-27 — Structured brainstorm; Barnes–Hut attention chosen as lead idea

Ran the ideation frameworks (`research/ideas.md`). Two facts surfaced: our base
model is **TabPFN v1** (`tabpfn==0.1.11`), not the v2+/Mitra models on today's
leaderboard, so any Elo claim needs a port; and TabArena's 3,600s fit budget is
mostly unused by foundation models.

Ran a synthetic attention-approximation sim (`research/sim_anchor_attention.py`).
Results (attention-output error, not accuracy): Track A's k-means anchors cut
error 2–4x vs random anchors (validates the committed change mechanistically);
log-count mass weighting alone is not a free win; adding exact near-field
expansion over the top-t clusters per query (Barnes–Hut style) cut error a
further 5–15x, at ~M + t·N/M rows touched per query.

**Decision:** pilot Barnes–Hut attention on real TabPFN v1 activations (no
training, laptop, ~1 week) before any fine-tuning track. Kill if real
activations don't cluster. Mitra episodic fine-tune stays as a parallel cheap
track. Not yet run on real activations.

## 2026-09-26 — Found a much cheaper path: fine-tune an existing checkpoint, not train from scratch

User asked about fine-tuning + quantizing an existing model instead of
prior-fitting from scratch. Checked it properly — see
`research/finetune-existing-models.md`. Three findings that change the plan:

1. **[TabTune](https://github.com/Lexsi-Labs/TabTune)** (MIT, active) already
   implements inference + full fine-tune + LoRA/PEFT + episodic ICL
   fine-tuning across 16 tabular FMs including Mitra, TabPFN, TabICL. This
   closes the "no training infra exists" gap from the previous entry — we
   don't need to build a training loop to get started.
2. Full fine-tuning beats LoRA on accuracy/convergence speed for TabPFNv2-scale
   models per ["On Finetuning Tabular Foundation Models"](https://arxiv.org/html/2506.08982v2);
   LoRA is a memory-fit tool for us (4GB), not an accuracy trick.
3. For tabular FMs specifically, weight quantization (QLoRA-style) barely
   helps since the models are already small — **FP8 attention
   quantization** is the tabular-specific lever, with a measured 1.7x
   speedup / no accuracy loss on TabPFN-v3 & TabICLv2 (["Attention Quantization
   for Tabular FMs"](https://arxiv.org/abs/2609.13031)).

**Decision: try this before committing to from-scratch pretraining or
asking for the 3090 Ti.** Mitra is already the exact checkpoint for our
already-chosen small-table niche — fine-tuning it (once, on a corpus
disjoint from the eval set, to keep this a legitimate zero-shot entry, not a
per-dataset "tuned" one) is a few hours of work using existing tooling, not
days of custom training infra. Not yet run.

## 2026-09-26 — Checked RLCD; decided against RL for the training process

User asked whether Jev's actual training method — RLCD (Reinforcement
Learning for Calibrated Decisions) — would be better than the supervised
prior-fitting process proposed above. Checked [arXiv:2609.29429](https://arxiv.org/abs/2609.29429).

**Decision: no, stick with supervised meta-training (cross-entropy on known
synthetic labels), not RL.** RLCD's use of RL is a workaround for adapting
an already-pretrained *generative* LLM, where there's no direct gradient
from "was the probability right" back through sampled text — that's the
same reason RLHF needs RL. Our situation is different: we're training a
transformer from scratch on synthetic (table, label) pairs where the label
is known exactly, so cross-entropy loss already gives a direct gradient and
already is a proper scoring rule (i.e. already optimizes calibration
directly). Adding an RL loop here would solve a problem we don't have.
Corroborating evidence: a related paper found in the same search reports
plain RL fine-tuning actually *leaves LLMs overconfident* unless calibration
is specially corrected for — RL isn't inherently "the calibrated one."

## 2026-09-26 — More compute available; checked "Jev/JEPA" lead; no training infra yet

**Compute revised:** possible colleague access to a 24GB RTX 3090 Ti, plus
Kaggle's free ~30 GPU-hrs/week (T4-class, 16GB), sit alongside the 4GB
laptop. Updated `research/from-scratch-feasibility.md` with a resource
table — recommendation is calibrate/iterate on the free options (laptop +
Kaggle) first, ask for the 3090 Ti only once a scoped run shows real signal.

**Checked the "Jev / TypeSafe AI" lead:** it's real, but it's a
general-purpose schema-constrained decision model for agentic pipelines, not
tabular-specific research to build on. While searching, found the actually
relevant, days-old paper: "A JEPA Recipe for Tabular Foundation Models"
([arXiv:2609.25541](https://arxiv.org/abs/2609.25541)) — tried
representation-space prediction for tabular FMs, and it **underperformed
plain value prediction while costing more compute**. Logged in
`research/landscape.md` so we don't spend a week rediscovering that.

**Open question, not yet answered:** do we have real training
infrastructure for from-scratch prior-fitting? No — `zsisab/engine.py` is
inference-time only (patches a frozen pretrained TabPFN), and
`create_synthetic_dataset.py` is a single `sklearn.make_classification` call,
nowhere near the diverse-SCM-prior generator the actual recipe needs. A real
training loop, a proper synthetic-prior generator, and an eval harness for
this are all unbuilt. That's the real next lift if we move past calibration.

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

## 2026-09-27 — Borrowed the real eval harness from Documents/tabarena; that project retired locally

User pointed out a second local project, `Documents/tabarena` — their fork
of the real upstream AutoGluon `tabarena` repo (`iam-saiteja/tabarena`,
branch `add-zsisab`), not a throwaway test. It had real infrastructure
ISAB-r lacked: the actual `tabarena`/`bencheval` evaluation packages with
real cached official TabArena baselines, plus working (committed, not just
brainstormed) implementations of `zsisab`, `zstabfm`, and **S3T2** —
correcting what I said earlier about S3T2 being purely fabricated; it
exists as real code there, I just didn't know that repo existed.

**Found an overclaim worth flagging plainly:** a commit there
("feat(zstabfm): ... achieve #1 Rank on TabArena Leaderboard") is not
supported by the results sitting next to it in the same repo — the actual
`tabarena_leaderboard.csv` from that run shows ZSTABFM at Elo 1866.8,
average rank ~7.4, behind TABFM (1944.6) and TabFM+ (1935.3), on an 18-task
subset, not the full 51-task suite. Recorded honestly in `harness/README.md`
rather than repeated at face value.

**Decision (user's choice):** keep ISAB-r as the primary project, but
borrow the real harness rather than continue with ISAB-r's rough
OpenML-CC18 approximation (`legacy/benchmarks/run_tab_arena.py`). Copied
`packages/tabarena` and `packages/bencheval` into `harness/`, plus a
reference results snapshot. Per explicit instruction: committed and pushed
the source repo's uncommitted work to its GitHub remote first (nothing
lost), then deleted `Documents/tabarena` entirely, including its `.venv`.

**Not yet done:** the harness isn't installed/tested in this repo's
environment yet — it needs its own dependency set (AutoGluon-heavy, likely
too large for `.venv-tabicl`). That's the next real step before H2's
TabICLv2 port work can be measured against genuine cached TabArena numbers
instead of approximations.

Also removed `.venv` (old TabPFN v1 testbed env) from this repo per
explicit request — `.venv-tabicl` is now the only environment here.
Reinstall `.venv` from `requirements.txt` only if `zsisab`/H1 needs
re-running.
