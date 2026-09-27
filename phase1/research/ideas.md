# Research idea brainstorm (2026-09-27)

Frameworks used: tension hunting (F3), what-changed (F5), boundary probing
(F6), cross-pollination (F4), composition (F9), then converge filters
(F10 explain-it, F1 problem-first, F7 simplicity, feasibility).

## Facts that constrain the ideas

- Our base model is **TabPFN v1** (`tabpfn==0.1.11` pinned in `requirements.txt`;
  `zsisab/` patches its `TransformerEncoderLayer`). That is the 2022 model, not the
  v2/2.5/3.5 or Mitra/LimiX models on today's leaderboard. Any Elo claim needs a
  port to a current architecture; the *method* transfers (the O(N²) axis is
  row-attention in all of them), the *code* doesn't.
- TabArena gives 3,600s of fit time; foundation models use seconds. Unused budget.
- Prior art already owns: exact chunking (Chunked TabPFN), FP8 attention
  (Kübler et al.), row compression in-architecture (TabICL), JEPA (negative), RL (unneeded).

## Candidates (diverge)

| # | Idea | Lens | Verdict |
|---|---|---|---|
| 1 | **Barnes–Hut attention**: exact near-field over the few clusters a query attends to, mass-weighted cluster summaries for the far field, training-free | F4 physics (N-body/FMM) + F6 | **Pursue** (sim below) |
| 2 | Compression-aware fine-tuning: fine-tune a TFM with anchored attention on so weights adapt to it | F9 | Pursue *after* #1 shows the base scheme is worth adapting |
| 3 | Episodic real-data fine-tune of Mitra (see `finetune-existing-models.md`) | F5 | Parallel track; likely small gain, cheap to test |
| 4 | Distill a big open TFM into a small fast student using teacher labels on unlimited synthetic tables | F3 perf↔efficiency | Park: check TabDPT-Turbo / TabPFN distillation first for prior art |
| 5 | Merge domain-specific fine-tunes of one base (task arithmetic) | F9 | Park: speculative, needs #3 first |
| 6 | Use the unused 3,600s: per-dataset fine-tune | F5 | Different (tuned) leaderboard tier, not zero-shot |
| 7 | Fuse FP8 attention with our chunked kernel | F9 | Engineering, incremental, not a research claim |
| 8 | Training-free predictor from theory (no pretraining) | F2 | Drop: nothing suggests it reaches the frontier |
| 9 | Study *where* TFMs lose to GBDTs (N, features, cardinality) | F6 | Cheap analysis, feeds #1/#2 targeting |

## Converge

**#1 passes all filters:**
- Two sentences: *TFMs must either subsample context or pay O(N²) at large N. Barnes–Hut attention keeps the few nearby clusters exact and summarizes the rest by mass, with a provable per-cluster error bound (error ≤ exp(‖q‖·r_c/√d) − 1 for cluster radius r_c), needing no retraining.*
- Problem-first: large-N tables on modest hardware; who: anyone whose table exceeds a TFM's context.
- Simplicity: the core is ~15 lines (the sim); no new parameters beyond M and t.
- Feasible: pilot needs no training, runs on the laptop.

## Simulation result (synthetic, `research/sim_anchor_attention.py`)

N=8000 keys from an imbalanced 24-cluster mixture, 5 seeds, mean relative L2
error of attention output vs exact (lower = better). Selected rows, M=64:

| Method | sharp=0.5 | sharp=1.0 | sharp=2.0 |
|---|---|---|---|
| exact attention on M subsampled rows | 0.226 | 0.257 | 0.310 |
| ISAB random anchors (current zsisab) | 0.091 | 0.091 | 0.132 |
| ISAB k-means anchors (Track A, committed) | 0.033 | 0.021 | 0.063 |
| monopole (k-means + log-count only) | 0.029 | 0.039 | 0.082 |
| **Barnes–Hut (t=4 clusters exact)** | **0.002** | **0.004** | **0.014** |

Reads:
1. Track A is mechanistically justified: k-means anchors cut error 2–4x vs random.
2. Mass weighting alone is **not** a free win (worse than plain k-means at sharp=1, 2). Don't sell it.
3. The near-field expansion is what buys the 5–15x further reduction.
4. Cost honesty: per query, Barnes–Hut touches ~M + t·N/M rows (≈ 570 here vs ISAB's 64 and exact's 8000), so it sits between ISAB and exact on speed.
5. **Limits:** synthetic clean clusters favor clustering; error is attention-output error, not accuracy; keys here are not real TabPFN activations. This is a hypothesis-supporting sim, not a result.

## Pilot (about 1 week, laptop, no training)

1. Hook real per-layer Q/K/V from TabPFN v1 on ~10 datasets with N=2k–10k (exact attention is affordable there, giving ground truth).
2. Measure layerwise attention error and end-task accuracy delta vs exact for: random anchors, k-means anchors, Barnes–Hut at several (M, t).
3. Gate: if Barnes–Hut closes most of the accuracy gap to exact at <=1/4 the attention FLOPs, port to a v2-style model (Mitra) and consider #2. If real activations don't cluster, kill it.

Strongest objection: real embeddings may not cluster tightly, so near-field
expansion needs large t. Pilot step 1 answers this directly.
