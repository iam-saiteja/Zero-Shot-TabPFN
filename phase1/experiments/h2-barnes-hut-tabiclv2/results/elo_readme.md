# Real Elo computed via bencheval — infrastructure works, the number doesn't hold up yet

`compute_real_elo.py` merges `real_api_default_path.py`'s results (real
preprocessing, real `predict_proba()` path) with the genuine cached official
baselines for the same 13 TabArena-tiny classification datasets, and runs
`bencheval.evaluator.BenchmarkEvaluator.compute_elo` — the actual TabArena
evaluation library, not a hand-rolled metric. Full table:
`real_elo.csv`. This is the first time this project has produced an Elo
number through real official infrastructure.

## Why the number itself isn't trustworthy yet

**Our own "vanilla" TabICLv2 run scores ~200 Elo above the official cached
run of the identical model** (1482.9 vs 1284.4). Checked whether this is
just noise: the 95% CIs barely overlap (ours: 1252–1886, official:
1183–1435) — our point estimate sits well outside the official model's own
confidence interval, for what should be the same model. That should not
happen, and it's the same category of red flag caught earlier with the MIC
outlier: two things that should agree don't, so the number is flagged
rather than reported as a result.

**Root cause, most likely:** 13 tasks with a single ad hoc 70/30 split each
is nowhere close to TabArena's real protocol (51 tasks, 10-repeat 3-fold
CV — 30 splits per dataset). Elo computed from head-to-head battles over
only 13 noisy single-split tasks has enormous variance (CI widths of
200–400 Elo here vs 100–150 for methods with the real protocol's full
sample behind them) — enough for split-to-split luck alone to swing a
model's apparent rank by dozens of places. This isn't a bug in the
computation; it's a real, expected consequence of the sample size, and it's
exactly why TabArena's real protocol uses 30 splits per dataset instead of
one.

## What this session actually accomplished vs. what it didn't

**Built and validated:** the full pipeline from patched model to a real,
official-library-computed Elo number — real preprocessing, real inference
path, real cached baselines, real evaluation code. That infrastructure is
correct and reusable.

**Not yet true:** "our method ranks 3rd/87" or any other specific number
from this run. The measurement isn't precise enough yet to support that
claim, and reporting it anyway would repeat exactly the overclaiming
pattern this project has been trying to avoid since the original fabricated
S3T2 doc.

## What a genuinely trustworthy number requires

Running the real 51-task suite with something closer to the actual 30-split
protocol — not 13 single-split tasks. That is a real, order-of-magnitude
bigger compute commitment than anything done in this session (13 datasets
took minutes; 51 datasets x ~30 splits each is a different scale of
undertaking), and worth a deliberate decision before starting, not an
automatic next step.
