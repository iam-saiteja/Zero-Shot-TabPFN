# Project Brief: MSA-TabPFN — A Tabular Foundation Model with MiniMax Sparse Attention

**Read this whole document before writing any code. It is the project context, the plan, and the working agreement in one place. Execute phase by phase. Do not skip ahead. Pause and check in with the user at the end of every phase before starting the next one.**

---

## 1. Mission

Build a tabular foundation model in the TabPFN family — pretrained once on synthetic data, never retrained on the user's actual data, single forward pass at inference — but replace its attention mechanism with **MiniMax Sparse Attention (MSA)**. The goal is to find out whether MSA's block-selection mechanism, originally built for ordered text, can transfer to permutation-invariant tabular rows and let the model scale past TabPFN's ~10K-row ceiling without losing the accuracy that made TabPFN beat trees at small scale.

Then benchmark it honestly against tuned XGBoost, CatBoost, and LightGBM on datasets from 10K rows up to several million rows.

**This is real research. There are three valid outcomes (see Section 6), and two of the three are "we found out it doesn't work and here's exactly why." That is not a failure condition for this project — it's a deliverable. Don't let the agent or the user drift into only optimizing for "it beats trees."**

---

## 2. Background (why this is worth doing)

- Tree ensembles (XGBoost/CatBoost/LightGBM) have dominated tabular ML for ~20 years.
- TabPFN (2022, v2 in *Nature* 2025) broke that for small datasets: pretrained on millions of synthetic datasets, it reads a new dataset in one forward pass and beats tuned tree ensembles. But attention cost is quadratic in row count, so it caps out around 10K rows.
- Three independent 2024–2025 lines of work attacked the scaling problem: faster attention (TabICL, TabFlex), retrieval of relevant rows (LoCalPFN), and bagging-style stability fixes. Each fixed one piece and left another broken.
- Multiple 2025 papers (ConTextTab, MultiTab, a late-2025 hardware-cost benchmark) independently concluded: nobody has yet combined "retrieval" and "better attention" into one native, differentiable mechanism. Trees still win at scale, in production, today.
- MiniMax Sparse Attention (June 2026) does exactly that fusion — for text. An Index Branch scores blocks of context and selects which matter per query; exact attention then runs only on selected blocks. It's MIT-licensed and open-sourced (`MiniMax-AI/MSA` on GitHub).
- Nobody has ported it to tabular data yet. That's the gap this project fills.

---

## 3. The core technical bet

> If we swap TabPFN's full attention for MSA's two-branch block-sparse attention, and solve the "what is a block when rows have no order" problem, the model should scale to 100K–millions of rows while keeping TabPFN's accuracy advantage.

The single hardest open question in this entire project: **MSA's block selection was designed and trained on sequential, ordered text. Tabular rows are permutation-invariant — there is no "next row."** Every phase below exists in service of answering this one question. Everything else (kernels, benchmarks, tree baselines) is supporting infrastructure.

---

## 4. Ground rules / working agreement for the agent

1. **Checkpoint after every phase.** Summarize what was built, what the numbers showed, and explicitly state whether we're tracking toward Failure Shape 1, 2, or 3, or toward success — then wait for the user before continuing.
2. **Don't hardcode GPU assumptions.** MSA's released kernels target NVIDIA SM100 hardware specifically. Detect what's actually available at the start of Phase 0 and confirm with the user which path to take (see Section 5, Phase 0).
3. **Implement the block-definition strategies as swappable modules**, not as one hardcoded choice — this is the actual research variable, not an implementation detail to lock in early.
4. **Keep an ablation discipline from day one**: every claim of "MSA helped" must be backed by a same-conditions run with MSA swapped out for the TabFlex-style linear attention baseline. No exceptions, even under time pressure.
5. **Log honestly.** If the indexer is selecting effectively-random blocks (Failure Shape 1 signature), say so immediately rather than continuing to tune hyperparameters hoping it improves.
6. Code style: Python, no inline comments unless asked, keep functions concise and readable over clever.

---

## 5. Phased build plan

### Phase 0 — Environment & repo setup
**Objective:** working dev environment, all reference code cloned and runnable.
- Detect available GPU(s) and CUDA version. Confirm with user: do we have SM100-class hardware (for the optimized MSA kernel) or do we build a plain-PyTorch reference implementation of MSA's mechanism first (recommended default — gets us to a research answer faster; optimized kernel is a stretch goal, not a blocker)?
- Clone/reference: TabPFN's official repo (Prior Labs), TabICL, TabFlex, and `MiniMax-AI/MSA` (search for current canonical links — don't hardcode stale URLs).
- Set up a clean Python environment (PyTorch, the tree libraries, an experiment tracker).
- **Exit criteria:** can run vanilla TabPFN v2 on a toy dataset and reproduce a published number, even approximately.

### Phase 1 — Baseline reproduction
**Objective:** a trustworthy benchmark harness before touching MSA at all.
- Stand up tuned XGBoost, CatBoost, LightGBM baselines on a handful of standard small datasets.
- Stand up TabPFN v2 on the same datasets, confirm it beats the trees (the known result) — this validates the harness, not the hypothesis.
- **Exit criteria:** numbers roughly match literature. If they don't, fix the harness before moving on — don't carry a broken baseline forward.

### Phase 2 — Synthetic pretraining data pipeline
**Objective:** the engine that generates millions of fake datasets for pretraining.
- Implement (or adapt from TabPFN's open code) the prior-fitted-network generator: sample structural causal models, generate (X, y) pairs.
- Validate output shapes and label distributions are sane across classification and regression.
- **Exit criteria:** can generate an arbitrary number of synthetic datasets of arbitrary row count on demand.

### Phase 3 — MSA backbone integration (the heart of the project)
**Objective:** a tabular transformer backbone with MSA's Index Branch + sparse Main Branch in place of full attention.
- Strip MSA's mechanism out of its text-specific packaging: keep the two-branch idea (cheap block-scorer → exact attention on selected blocks only), drop anything that assumes token order.
- Implement at least two competing strategies for "what is a block" on unordered rows, as swappable modules:
  - **A — naive/random blocking**: fixed-size random groupings, no learned structure. This is the sanity-check floor.
  - **B — similarity-clustered blocking**: group rows by a cheap similarity measure (e.g., embedding distance or k-means) before the indexer scores blocks.
  - *(Optional C if A/B both show signal)* — a fully learned, jointly-trained block assignment.
- Wire this into the TabPFN-style backbone in place of full attention.
- **Exit criteria:** model trains without diverging on small synthetic datasets, and produces predictions (accuracy doesn't need to be good yet — just need a working forward/backward pass).

### Phase 4 — Pretraining run
**Objective:** actually pretrain the swapped-attention model on the synthetic pipeline from Phase 2.
- Train at a scale you can afford; this does not need to match a 109B-parameter LLM budget — TabPFN-scale models are far smaller.
- Instrument the indexer: log what fraction of selected blocks look meaningfully different from random selection. This single metric is your earliest signal on Failure Shape 1.
- **Exit criteria:** a trained checkpoint, plus the indexer-behavior log.

### Phase 5 — Evaluation at scale
**Objective:** the actual experiment.
- Benchmark on datasets spanning small (sanity check vs. published TabPFN numbers) up through 100K–several million rows (TabZilla-hard, TabReD, large OpenML sets).
- Metrics: accuracy/AUC, wall-clock inference time, peak memory.
- Compare against: vanilla TabPFN v2 (will fail/cap out at scale — expected), TabFlex, TabICL if feasible, and the tuned tree baselines from Phase 1.
- **Exit criteria:** a results table covering all methods across all dataset sizes.

### Phase 6 — Ablations
**Objective:** isolate whether MSA specifically is doing the work.
- Same model, same training, swap MSA back out for TabFlex-style linear attention. Re-run the same evaluation.
- Compare blocking strategies A vs. B (vs. C) against each other.
- **Exit criteria:** can answer, with evidence, "is MSA the active ingredient, or would any attention swap have done this?"

### Phase 7 — Failure-mode diagnosis & scoping
**Objective:** figure out which of the three outcomes in Section 6 actually happened, and scope the claim honestly.
- If results are mixed across dataset types, characterize *which* kinds of data favor MSA vs. trees (local structure vs. global signal) — this is Failure Shape 3's deliverable.
- **Exit criteria:** a clear, evidenced statement of what this method does and doesn't do.

### Phase 8 — Write-up
**Objective:** the paper.
- Architecture description, with explicit detail on how the row-ordering problem was handled (this is the novel contribution regardless of which outcome occurred).
- Full results tables, ablations, and — required, not optional — the honest failure cases / scoped limitations.

---

## 6. Success and failure criteria (decide which paper this is, don't pretend otherwise)

| Outcome | Signature | What we publish |
|---|---|---|
| **Success** | MSA-backed model matches/beats tuned trees at 100K–millions of rows | First tabular foundation model closing the scale gap; full ablation proving MSA (not just any swap) drove it |
| **Failure 1** | Indexer selects near-random blocks; no accuracy gain over TabFlex baseline | Documented evidence that MSA's selection mechanism needs sequence order to function — tells the field what must be solved first |
| **Failure 2** | Transfers fine, scales fine, but plateaus at the same "relatively solid" level as TabFlex/TabICL | The bottleneck is the synthetic pretraining prior, not attention — redirects the field |
| **Failure 3** | Wins on local-structure data (sensor/spatial/medical), loses on global-signal data (financial/ad-click) | A scoped taxonomy: when to use sparse-attention tabular PFNs vs. trees |

---

## 7. Immediate next action for the agent

Start Phase 0. Report back: what GPU/CUDA environment is actually available, and which reference repos were successfully cloned and verified runnable. Stop and wait for the user before Phase 1.
