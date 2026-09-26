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
