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

## 2026-09-27 — Restructured into phase1/ + phase2/; deleted disposable clutter; license reconsidered

User: no proper gitignore, too many venvs/loose folders at root, wanted only
two visible folders (`phase1/`, `phase2/`) with everything sorted into one
or deleted. Also asked about `data/comparison_results_30.csv` open in their
IDE — confirmed those were failed runs from `legacy_v1era/benchmarks/`.

**Deleted outright** (all gitignored, never in git history, so nothing
lost): `AutogluonModels/` (100K runtime cache), `catboost_info/` (runtime
log), `tabzilla/` (vendored clone, explicit "not needed"), `scratch/`
(one-off parse scripts + the already-superseded UNVERIFIED analysis doc),
`.qodo/` (unrelated editor tool cache), `datasets/` (395MB — turned out to
be mislabeled AutoGluon run dumps under `toy_binary`/`toy_regression`, not
real data, fully regenerable). `data/`'s two small result CSVs were
preserved into `phase1/legacy_v1era/data/` rather than deleted, since the
user had one open and was actively referencing it.

**Moved into `phase1/`**: `legacy/` (renamed `legacy_v1era/`), `zsisab/`,
`experiments/`, `research/`, `docs/` (this file), `findings.md`,
`research-state.yaml`, `harness/`, `create_synthetic_dataset.py`,
`requirements*.txt`, `to_human/`. Fixed the two H1 pilot scripts
(`pilot.py`, `analyze.py`) to resolve paths from `__file__` instead of
assuming repo-root cwd, since the move breaks the old relative paths.

**`.venv-tabicl`** recreated under `phase1/.venv-tabicl` (deleting and
`uv venv` + `uv pip install`ing again was faster than trying to relocate a
venv with hardcoded interpreter paths - uv's package cache made the
reinstall near-instant, no re-download).

**`.gitignore` simplified** to 3 lines (`phase1/.venv-tabicl/`,
`phase1/.venv/`, `__pycache__/`) now that everything else disposable is
just gone rather than hidden.

**License framing corrected:** user pointed out this is non-commercial
research, not a product — re-checked the actual license text and both
LimiX-2 and TabPFN-3 explicitly permit research/evaluation use, they only
forbid commercial/production use. Revised `research/model-choice.md`:
TabICLv2 stays the Phase 1 target because its result stays freely reusable
by anyone (no license inherited restriction), but repeating the validated
method on LimiX-2 (actual #1) or TabPFN-3 is now a planned, licensed-for-this
follow-up once H2 works, not an excluded option.

`phase2/` created as an explicit placeholder — not started, per instruction
to work on Phase 1 only for now.

## 2026-09-27 — H2 smoke test passed: patch mechanism verified on TabICLv2

Patched `ICLearning.tf_icl` (instance-level, not the shared class - `tf_col`/
`tf_row` provably untouched via object-identity check) to replace exact
`k=v=q[:train_size]` attention with k-means coreset anchors (the anchor half
of H1's method; near-field exact expansion not yet ported - needs
`attn_mask`-based per-query masking, a real next step, not done yet).

All 4 locked sanity checks + the fidelity gate passed on an untrained,
random-init TabICL. Caught and fixed a real gotcha before trusting the
result: the first run used TabICL's default `zero_init=True`, which zeros
the attention output projection, making the divergence check pass
trivially (0.0000 diff) regardless of whether the patch was even correct.
Reran with `zero_init=False`; the real number is 0.0084 mean abs diff
(7.2% relative) - same order of magnitude as H1's real numbers, not
inflated or trivial.

**This is a mechanism check, not an accuracy or speed result** - untrained
weights, tiny config, anchor-only (no near-field yet). Details and honest
scope: `experiments/h2-barnes-hut-tabiclv2/analysis.md`. Next: run this
same patch against TabICLv2's real pretrained checkpoint.

## 2026-09-27 — Real checkpoint result: anchor-only insufficient, near-field is mixed

Ran the tf_icl patch against TabICLv2's actual pretrained checkpoint (real
weights, real OpenML data, single dataset/split - exploratory, not H1-level
rigor yet). Anchor-only prediction agreement with exact attention: 80-94%
across M=16-128, worse than H1's TabPFN v1 numbers - confirms near-field
expansion is necessary, not optional, at this model's scale/depth (12 ICL
blocks vs whatever H1 used on TabPFN v1).

Implemented near-field expansion (`patch_nearfield.py`, mask-based, still
O(train_size) per query - correctness check, not a speed win yet, same
honest sequencing as H1). Result is genuinely mixed: prediction agreement
improves substantially (82-84% -> 92-93%), but the raw softmax-fidelity
metric barely moves and is sometimes marginally worse. Recorded both
numbers rather than reporting only the one that looks good - see
`experiments/h2-barnes-hut-tabiclv2/analysis.md` for the full table and
three candidate (unconfirmed) explanations for the disagreement.

**Not yet done, in order:** multi-seed/multi-dataset repeat of this at
H1's rigor before trusting these numbers further; investigate the
fidelity-vs-agreement disagreement; a real sparse kernel (this near-field
version still doesn't save compute, only proves correctness); real
TabArena numbers via `harness/`.

## 2026-09-27 — Found and fixed the cause of the mixed near-field result

Investigated the fidelity-vs-agreement disagreement from the previous entry
(the disciplined next step, before scaling up to more seeds/datasets on a
possibly-broken measurement). Root cause: `log_counts` (real rows per
cluster) was only used to pick which clusters get exact near-field
treatment, never added as an attention-logit bias for the surviving
monopole entries - so a cluster standing in for 50 rows got the same
softmax weight as 1 row. Fixed in `patch_nearfield.py`.

**After the fix:** fidelity drops 3-6x vs anchor-only (into H1's 0.02-0.06
range on real TabPFN v1 data), now roughly monotonic in `t` as theory
predicts, and both metrics (fidelity, prediction agreement) move together
instead of disagreeing. Full before/after table in
`experiments/h2-barnes-hut-tabiclv2/analysis.md`.

**Still not done, unchanged from before:** multi-seed/multi-dataset repeat
at H1's rigor, a real sparse kernel (still O(train_size) per query, masked
not gathered), and real TabArena numbers via `harness/`.

## 2026-09-27 — H2 confirmed across 8 real datasets: Barnes-Hut beats anchor-only, 16/16

Ran the (bug-fixed) near-field patch across the same 10 OpenML datasets + 2
seeds H1 used, against the real TabICLv2 checkpoint (`sweep.py`). First run
failed on a real bug in the sweep script itself (batch dim not squeezed
before `accuracy_score` - all 20 runs failed identically on the same shape
error, fixed in one line, rerun).

2 of 10 datasets (adult, jm1) give NaN fidelity - root cause confirmed:
this rig bypasses TabICLClassifier's normal input normalization, and those
two datasets have raw feature magnitudes up to ~1.37M, overflowing the
softmax. Test-rig artifact, not a method flaw - excluded from stats, not
hidden.

**Across the remaining 8 datasets x 2 seeds = 16 comparisons: Barnes-Hut
beats anchor-only on fidelity in all 16, zero exceptions.** Mean fidelity
0.050 vs 0.111, mean accuracy loss vs exact -1.75pp vs -6.19pp. Strongest
result in the project so far - real checkpoint, real data, multi-dataset,
bug-fixed, mechanistically understood (the log-count bias fix from the
previous entry is exactly what made this consistent). Full table:
`experiments/h2-barnes-hut-tabiclv2/analysis.md`.

**Still open:** fix the rig to use real preprocessing (kills the NaN
cases), a real sparse kernel (still no speed win shown), and the actual
TabArena protocol via `harness/` instead of a custom pilot script.

## 2026-09-27 — Real TabArena-tiny dataset test; stopped short of an Elo number, on purpose

Ran on the actual 13 classification datasets from the real
`harness/tabarena_tiny_results_reference` (genuine TabArena dataset names,
not another arbitrary list) - 2 skipped on a data-loading bug (raw string
columns not caught by dtype detection, not a method issue).

**Internal result (valid):** Barnes-Hut beats anchor-only on 9/11 datasets,
same pipeline both sides. Real, but not the clean 16/16 the earlier
`sweep.py` showed - 2 real exceptions (Fitness_Club, MIC), not explained
away, logged as open.

**Stopped before computing an Elo number, deliberately:** cross-checked our
own vanilla (exact attention) against the real cached `TABICLV2 (default)`
baseline for the same datasets - several diverge substantially (MIC: 0.469
official vs 0.099 ours; Fitness_Club: 0.191 vs 0.352; maternal_health_risk:
0.354 vs 0.524; seismic-bumps: 0.214 vs 0.303). Our pipeline isn't
equivalent to real TabICLv2 usage yet: crude preprocessing, `_train_forward`
instead of the real `forward_with_cache` inference path (the KV-cache gap
flagged earlier - now confirmed to matter beyond just production realism),
and an ad hoc split instead of TabArena's real fold protocol. Merging with
the official cache and computing Elo now would produce a number that looks
authoritative but rests on a baseline mismatch. Full writeup:
`experiments/h2-barnes-hut-tabiclv2/results/tabarena_tiny_readme.md`.

**Real remaining gap before an Elo number is trustworthy:** route through
`TabICLClassifier`'s actual preprocessing + `forward_with_cache`, and match
TabArena's real fold protocol. Not a quick fix - a real chunk of work,
flagged for a decision rather than pushed through silently.

## 2026-09-27 — Built the real inference-path integration; corrected a wrong assumption along the way

Traced the exact call chain properly (per instruction to "build that
properly"): `predict_proba`'s default (`kv_cache=False`) never touches
`forward_with_cache` - it calls `Encoder.forward`, the method already
patched and validated. Built `patch_kvcache.py` for the real KV-cache path
anyway (real, correct, kept as infrastructure for `kv_cache=True` users),
but the actual fix for the baseline-mismatch problem was simpler: run
through real preprocessing + the real `predict_proba()` ensemble path
instead of the `_train_forward` bypass.

**Result:** mean gap to the official cached TabICLv2 baseline dropped from
an untrustworthy mismatch to 0.049 across 13 real TabArena-tiny datasets -
most now genuinely close, one real unresolved outlier (MIC, 0.378 gap).
Barnes-Hut vs vanilla via the real API: modest real cost (~1.1pp mean error
increase), the expected shape for an approximation. Full detail:
`experiments/h2-barnes-hut-tabiclv2/analysis.md`.

**This is now trustworthy enough to compute a real Elo number against the
official cached leaderboard - the next step, not yet done.**

## 2026-09-27 — Computed a real Elo number via bencheval; flagged it as not trustworthy yet

Merged our real-API results with the genuine cached official baselines for
13 real TabArena-tiny datasets and ran the actual `bencheval` Elo computation
(not a hand-rolled metric) - a real infrastructure milestone.

**Caught before reporting it as a result:** our own vanilla TabICLv2 run
scored ~200 Elo above the official cached run of the identical model
(1482.9 vs 1284.4), with 95% CIs that barely overlap. Same red-flag category
as the MIC outlier earlier - two things that should agree don't. Root cause:
13 single-split tasks vs TabArena's real 51-task/30-split protocol gives
enormous Elo variance (CI widths 200-400 here vs 100-150 for methods with
the real protocol's sample behind them) - not a computation bug, an expected
consequence of the tiny sample.

**Decision: infrastructure is real and works; the specific number (our
vanilla ranks 3rd/87) is not reported as a result** - would repeat the
overclaiming pattern this project has tried to avoid throughout. Full
writeup: `experiments/h2-barnes-hut-tabiclv2/results/elo_readme.md`.

**What a trustworthy number needs:** the real 51-task suite with something
closer to the actual 30-split protocol - an order-of-magnitude bigger
compute commitment than this session's work, flagged as a decision point
rather than started automatically.

## 2026-09-27 — Built the sparse kernel: correctness verified, speed/memory win not achieved (honest)

Built a real per-query gather-based Barnes-Hut attention kernel
(`sparse_kernel.py`) instead of the masked-over-dense approach used
everywhere else. Found and fixed 3 real bugs while validating it against the
already-proven dense version (via a hooked capture of dense's actual
attn_mask, not guessing): clustering done in the wrong (normalized) space,
a missing `ssmax_layer` application, and - the actual root cause of the
failure - an inverted far-field masking polarity. After all three fixes:
exact match (1e-6) to the validated dense version.

**Benchmarked speed/memory on the same RTX 3050 4GB this whole project has
run on: the sparse kernel is slower and uses more memory than both plain
exact attention and the dense-masked version, at every tested scale, and
OOMs earlier than exact attention does.** Root cause: plain exact attention
already benefits from an efficient (likely FlashAttention-style) SDPA path
when no custom mask is supplied (near-flat memory 671->785MB across a 40x
increase in N); introducing any custom float attn_mask forces a slower,
more memory-hungry fallback, and my hand-rolled unfused gather/einsum
implementation adds further real overhead on top of that. Full numbers and
reasoning: `experiments/h2-barnes-hut-tabiclv2/analysis.md`.

**Decision: report this honestly as correctness-yes/performance-no, not as
a win.** A real performance win needs an actual fused kernel (Triton) or a
restructuring that avoids triggering SDPA's slow custom-mask path entirely
(e.g. two separate unmasked calls instead of one masked combined call) -
both real, separate next steps, not done here.

## 2026-09-28 (overnight, autonomous) — Found exact attention's REAL ceiling: paging, not just OOM

Ran `exact_ceiling_check.py` on the real checkpoint (12 ICL blocks, real
embed/icl dims), pushing `tf_icl`'s plain exact attention from 50k to 1.5M
rows on this 4GB RTX 3050, to answer: does exact attention actually have a
nearby problem on this hardware, or was the whole Barnes-Hut effort chasing
something only relevant at scales this GPU could never reach anyway?

| N | time | peak memory |
|---|---|---|
| 50,000 | 39.6s | 1,149 MB |
| 100,000 | 156.0s | 2,174 MB |
| 200,000 | 685.3s (~11 min) | 4,221 MB |
| 400,000 | **8,645s (~2.4 hours)** | 8,318 MB |
| 700,000 | OOM | - |

**This answers last night's open question decisively: yes, there's a real,
nearby problem.** Peak memory at N=400k (8.3GB) exceeds the card's physical
4GB - Windows/CUDA was paging into system RAM rather than cleanly failing,
which is why time exploded non-linearly (200k->400k is a 2x row increase
but a ~12.6x time increase, not the ~4x the earlier scaling trend would
predict). Hard OOM lands at 700k, but the *practical* ceiling - where it
stops being usable at all - is much lower, somewhere around 200-300k rows.

**This changes the priority call from earlier:** the Barnes-Hut/sparse
kernel line of work is justified after all - there's a real wall around
200-400k rows on this exact hardware, not just a theoretical one at
TabICLv2's own reported 1M-row target.

**Decision (see next entry): pursue a chunked EXACT (not approximate)
attention first**, before more Barnes-Hut kernel work. Reasoning: the
failure mode just found is a peak-memory problem, not fundamentally an
accuracy-vs-compute tradeoff problem - a chunked online-softmax
accumulator (same technique already validated in `zsisab/engine.py` for
TabPFN v1, and matching the real "Chunked TabPFN" prior art in
`research/landscape.md`) can cap peak memory with ZERO accuracy cost,
unlike Barnes-Hut which trades accuracy for it. If chunking alone pushes
the practical ceiling past 400k, that's a bigger, safer, easier win than
finishing the sparse Barnes-Hut kernel - and Barnes-Hut would then only be
needed for whatever gap remains beyond what chunking alone can reach.

## 2026-09-28 (overnight, autonomous) — Chunked exact attention: real bug found and fixed

First implementation of `chunked_exact.py` only chunked the key/value
dimension, not the query dimension - left an `(all T queries) x (key
chunk)` intermediate score tensor, which at N=200k with chunk_size=16384
tried to allocate 97.69 GiB (crashed immediately, caught before it was
mistaken for a fundamental limitation). Fixed with proper nested chunking
(query blocks x key/value blocks - the actual FlashAttention structure).
Re-verified: matches plain exact attention within 2.7e-5 (still genuinely
exact, zero accuracy cost). Real memory/speed benchmark at scale (200k-1.5M
rows, small bounded chunk sizes) launched, not yet complete.

## 2026-09-28 (overnight, autonomous) — Investigated the 2 anchor-only-wins exceptions: genuinely unexplained

Checked two candidate explanations for why anchor-only beat Barnes-Hut on
Fitness_Club and MIC (both times this comparison has been run): low feature
count causing degenerate k-means clustering, and severe class imbalance.
Both refuted by direct counterexample: phoneme (a clear Barnes-Hut win) has
*fewer* features (5) than Fitness_Club (6) and nearly identical class
balance (70.7/29.3 vs 69.7/30.3) - if either theory were right, phoneme
should behave like Fitness_Club, not oppositely.

**Not resolved.** Logged honestly as a genuine open question rather than
forcing an explanation that doesn't survive its own counterexample. One
partial, unconfirmed observation worth keeping in mind: on Fitness_Club,
anchor-only's AUC (0.817) is actually *better* than exact attention's own
(0.648) - i.e. the "approximation" isn't just noisier, it's doing
something exact attention itself doesn't do well on this dataset. Whether
that's a real regularization effect or a one-off artifact of a single
train/test split isn't known - would need repeated splits to check, not
done tonight.

## 2026-09-28 (morning) — Chunked-exact: same fundamental wall as the sparse kernel

The full overnight benchmark (chunk_size=query_chunk_size=2048) never
produced output because it was catastrophically slow, not hung - killing it
lost the buffered output (SIGKILL doesn't flush stdout), so a timed
sub-test was run instead: 20,000 rows, 1200 total chunk-steps, took 11.4s ->
**~9.5ms overhead per step**. Extrapolated to N=1.5M at that chunk size:
~6.4M total steps -> ~17 hours. Confirmed by testing large symmetric chunks
(65536) instead: immediately OOM'd trying to allocate 128GB (query_chunk x
key_chunk x heads x 4 bytes grows fast when both are large).

**Root cause, and it's the same one behind the sparse Barnes-Hut kernel's
failure two nights ago:** chunking exact attention in pure PyTorch bounds
memory but does not reduce the O(N^2) total work - it serializes what would
otherwise be one large parallel GPU op into many small ones, each paying
real Python-loop + CUDA-launch overhead. Real FlashAttention gets its speed
from a fused kernel where that per-tile overhead is near-zero; a
Python-level loop over PyTorch ops cannot replicate that. **Both attempts
this session hit this same wall - it is not a coincidence, it is the
correct explanation for why neither delivered a speed win**, and it points
at the same real next step for both: an actual Triton/CUDA kernel, not
further pure-PyTorch restructuring.

One asymmetric-chunk test (chunk_k=2048, chunk_q=8192, N=700,000 - past
exact attention's hard OOM point) is running in the background as of this
writing to get one real data point before finalizing the brief.

## 2026-09-29 — Session restarted; real TabArena run resumed from checkpoint

Previous session ended abruptly (process killed), stopping the real
TabArena v0.1 run and the leftover 700k chunked-exact test. Real progress
had survived to disk: 1,122 rows (374 dataset-repeat-fold combinations x 3
methods) across 14 of 38 classification datasets, written incrementally by
the script's resumable design. Resumed cleanly using the same script's
built-in skip-already-done logic - no wasted recomputation. The 700k
chunked-exact test was not resumed (superseded - the Triton-kernel-vs-Elo
question was already decided in favor of the Elo run before the session
ended).
