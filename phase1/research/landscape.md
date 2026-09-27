# TabArena landscape (as of 2026-09-26)

Real, sourced facts gathered via web search — not internal knowledge, since
this is a living benchmark and internal knowledge is stale. Re-check before
trusting old numbers here; TabArena moves.

## Current leaderboard

- **#1 overall: LimiX-2**, 400M params, Elo **1935**. Uses a "Contextual
  Mechanism Network" pretrained with context-conditional masked modeling on
  SCM-generated synthetic data — learns joint feature+label structure rather
  than a fixed column-then-row pipeline. [arXiv:2609.17488](https://arxiv.org/abs/2609.17488)
- **TabPFN-3.5**: Elo **1910** (strongest variant). [arXiv:2609.17895](https://arxiv.org/pdf/2609.17895)
- Also near the top: TabFM, EXAONE Tabular, Mitra-v2, TabICLv2, RealTabPFN-2.5.
- Regression-only sub-leaderboard: **TabFM leads at 2019 Elo**; Xiaomi-TabLDM
  2nd at 1900.
- **Correction to our own target:** current #1 is ~1935, not 2000+. The
  benchmark is "living" — new entries push the bar; whatever number we chase
  will already be stale by the time anything here is trained. Target the
  frontier, not a fixed number.

Source: [TabArena leaderboard (HF Space)](https://huggingface.co/spaces/TabArena/leaderboard), [intelligentliving.co](https://www.intelligentliving.co/limix-2-tops-tabarena-tabular-ai/)

## Benchmark mechanics (this matters for planning)

- 51 IID tasks (binary/multiclass classification + regression), 10-repeat
  3-fold CV, up to 30 train/val/test splits per dataset.
- **Fit time limit: 3,600s (1 hour) per bagged run**, not the 30s the old
  (fabricated, now-deleted) `scratch_architecture_analysis.md` claimed.
- Reference protocol: 8-fold bagging + refit, run on **one node with 24 CPU
  cores and one NVIDIA RTX Pro 6000 GPU** — TabArena's own reference
  hardware, not the machine a model was trained on.
- **Implication for us:** the 4GB laptop constrains *training/prototyping*
  only. If ZS-ISAB (or anything else) becomes an actual TabArena submission,
  inference for scoring runs on TabArena's own hardware, with a fairly
  generous 1-hour budget. Don't over-constrain the design for inference-time
  VRAM unless the goal is also "usable by people who only have consumer
  hardware," which is a separate, real, but different claim.

Source: [TabArena paper (NeurIPS 2025)](https://arxiv.org/html/2506.16791v1)

## What the top models actually do differently

| Model | Key mechanism | Pretraining data | Notes |
|---|---|---|---|
| TabPFN (orig., 2022) | Transformer prior-fitting on synthetic SCM-style data | 9.2M synthetic datasets, N≤1024 during fitting | 20h on 8×RTX 2080Ti = **160 GPU-hours total**. [OpenReview PDF](https://openreview.net/pdf?id=cp5PvcI6w8_) |
| TabPFN v2 | Same paradigm, scaled up, N≤10k / 500 features | — | ~2 weeks on 8×RTX 2080Ti. [Prior Labs](https://priorlabs.ai/tabpfn-2) |
| TabICL | 3-stage: column-transformer → row-transformer → dataset-wise ICL transformer | synthetic, up to 60K samples during training, scales to 500K at inference | The column/row split is the architectural trick that caps attention cost — built into pretraining, not bolted on after. [arXiv:2502.05564](https://arxiv.org/abs/2502.05564) |
| TabDPT | ICL + **retrieval-based context selection** + self-supervised (column-masking) auxiliary loss, trained on **real** data too | real + synthetic | Retrieval narrows the effective context per query instead of compressing to fixed anchors. [NeurIPS PDF](https://www.cs.toronto.edu/~mvolkovs/NeurIPS2025_TabDPT.pdf) |
| Mitra / Mitra-v2 | 12-layer, 72-75M params, 2D (row+column) attention, no positional encoding, **mixed synthetic priors** (multiple different generating mechanisms) | synthetic only, diverse | Explicitly strongest on small tables (<5,000 rows, <100 features) — a **specialist niche claim**, not a generalist one, and it's competitive at that scale. [Mitra-v2 report](https://arxiv.org/pdf/2609.04540) |
| LimiX-2 | Contextual Mechanism Network + CCMM pretraining | synthetic (SCM) | Current #1; single model does classification+regression+imputation. [arXiv:2609.17488](https://arxiv.org/abs/2609.17488) |

## Important prior art directly overlapping our own work

**"Chunked TabPFN: Exact Training-Free In-Context Learning" (Sept 2025,
[arXiv:2509.00326](https://arxiv.org/pdf/2509.00326))** — tiles TabPFN's
attention over the training set using an online-softmax accumulator,
**training-free**, and is **mathematically exact** (zero accuracy delta vs
vanilla full attention) rather than an approximation. No inducing points, no
ISAB, no compression — it just makes the exact computation memory-safe.

This matters directly for us: it already solves "TabPFN OOMs on big
datasets," and it solves it *without* the accuracy cost ZS-ISAB pays for
using M=512 inducing points instead of the full N rows. **ZS-ISAB's "we fix
TabPFN's OOM problem" framing is no longer the differentiator — that's
already published, and exact beats approximate on accuracy.**

What ZS-ISAB (and Track A's better anchors) still legitimately owns: Chunked
TabPFN is memory-safe but still **O(N²) compute** — every query still
attends over all N training rows. ZS-ISAB's O(N·M) compute (M≪N) is a real
speed advantage at extreme row counts, at a controlled accuracy cost. The
honest positioning is **"fastest at extreme scale, small accuracy trade-off"**
vs Chunked TabPFN's **"exact, but still O(N²) time."** That's a real,
narrower, currently-uncontested claim — not "we fixed TabPFN," but "we're
the throughput option once you're past the point where exact attention is
still fast enough to matter."

## A very recent negative result worth knowing before chasing it

**"A JEPA Recipe for Tabular Foundation Models"** ([arXiv:2609.25541](https://arxiv.org/abs/2609.25541),
posted days before this was written) tried applying LeCun's joint-embedding
predictive architecture — predicting in representation space instead of raw
cell values — to tabular foundation model pretraining. Getting it to not
collapse required real engineering (value head reads the encoder field,
EMA target). Even after fixing collapse, **it underperformed the plain
value-prediction baseline** (32 wins / 70 losses on classification, 8/24 on
regression, across 147 datasets) **and cost more compute to get there**
(1.42x more training steps, 1.66x more wall-clock).

Relevant because it was raised as "read about Jev from TypeSafe AI" for
inspiration — worth being precise about what that actually is: **TypeSafe
AI's "Jev"** is a real, shipped product, but it's a general-purpose
schema-constrained decision model for agentic/LLM pipelines (declare a
closed set of possible answers, the model is architecturally prevented from
returning anything outside it), not a tabular foundation model, and not
open research to build on directly. Conceptually it's TabPFN's own
"single forward pass, closed prediction schema" paradigm applied to a
different problem (structured decisions instead of table rows) — so if
anything, our field already had the idea Jev is built on, not the reverse.

The JEPA paper above is the actually-relevant, actually-recent tabular
research in this space — and its answer is "we tried representation-space
prediction for tabular FMs, it didn't help." Worth knowing before spending a
week of compute chasing the same idea. Tabular cell values are already a
low-dimensional, mixed discrete/continuous signal — much of JEPA's value in
vision/robotics comes from avoiding the cost of predicting high-dimensional
raw pixels, a problem tabular data doesn't have in the first place, which is
a plausible reason the trick doesn't transfer.

## What Jev is actually trained with: RLCD — and why it doesn't apply here

Jev's own training method is real and has a name: **RLCD (Reinforcement
Learning for Calibrated Decisions)**, described (thinly — TypeSafe hasn't
published the full recipe) in
[arXiv:2609.29429](https://arxiv.org/abs/2609.29429). The one substantive
line available: "RLCD trains a model to return calibrated decisions instead
of generated text. Calibrated means that among decisions assigned
probability p, a fraction close to p is correct."

This is solving a different problem than ours: RLCD adapts an **already
pretrained generative LLM's** decision behavior via RL — RL is the standard
tool there because the LLM produces answers by sampling/generation, so
there's no direct gradient from "was the probability right" back to the
model without policy-gradient machinery (the same reason RLHF uses RL at
all). We're not adapting a pretrained generator; we're training a
transformer **from scratch** on synthetic (table, label) pairs where the
correct label is known exactly. That objective is already directly
differentiable — cross-entropy loss on the true label **is** a proper
scoring rule, which is precisely what makes a model calibrated. There's no
missing gradient RL needs to route around, so introducing an RL loop would
add variance and complexity for a problem supervised learning already solves
directly. This is exactly the same training process TabPFN, TabICL, TabDPT
and Mitra all already use (in-context learning via supervised meta-training
across synthetic tasks), and it's not incidental — it's the correct tool for
"labels are known, loss is differentiable."

Worth noting the field's own evidence points the other way, too: a related
paper found in this same search, "Balancing Classification and Calibration
Performance in Decision-Making LLMs via Calibration Aware Reinforcement
Learning" ([alphaXiv 2601.13284](https://www.alphaxiv.org/abs/2601.13284)),
reports that plain RL fine-tuning (RLVR) **leaves LLMs overconfident** —
calibration has to be specially corrected for, it isn't a free side-effect
of using RL. So "RL" and "calibrated" aren't the same claim; RLCD earns
calibration through its specific reward design, not from RL itself.
