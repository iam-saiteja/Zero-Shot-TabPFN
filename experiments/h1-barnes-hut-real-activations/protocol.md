# H1: Barnes-Hut attention closes the ISAB fidelity gap on real TabPFN v1 (LOCKED before running)

**Hypothesis.** On real tabular data, k-means anchors beat random anchors (H1a), and adding exact
near-field expansion over the top-t clusters per query (Barnes-Hut) beats both (H1b), measured by
end-task fidelity to exact attention. Mechanism: attention mass concentrates on a few nearby clusters;
exact there + mass-weighted summaries elsewhere leaves little error. Synthetic sim (research/sim_anchor_attention.py)
predicts 2-4x (H1a) and a further 5-15x (H1b).

**Setup.** tabpfn==0.1.11 (bundled v1 checkpoint), N_ensemble_configurations=4, 10 OpenML datasets
(<=100 features, <=10 classes), train subsample N=3000, test 500, 2 seeds. Same TabPFNClassifier fit; only
the encoder-layer self-attention is swapped. Clusters/anchors live in row-embedding space (as in zsisab/engine.py);
cluster mean key/value are exact because W_k/W_v are linear.

**Modes.** exact (reference) | isab_random | isab_kmeans (3 Lloyd iters) | bh (kmeans + top-t exact near-field,
mass-weighted monopoles for the rest). M in {32,64,128}, t=4 (also t=8 at M=64).

**Metrics.** PRIMARY: mean |p_mode - p_exact| over test rows/classes (fidelity). SECONDARY: accuracy, log-loss vs labels.
Cost proxy: rows touched per query = M + t*N/M (bh), M (isab), N (exact).

**Sanity checks before trusting anything.** (1) patched exact == vanilla TabPFN probs (max diff < 1e-4);
(2) all modes finite; (3) exact accuracy above majority baseline on most datasets.

**Predictions / gate.** Proceed to a v2-style port only if: bh(M=64,t=4) median fidelity error <= 0.5x isab_kmeans(M=64)
on >=7/10 datasets AND mean accuracy delta vs exact within 0.5pp. Kill if real embeddings do not cluster
(bh no better than 1.5x over isab_kmeans). Anything else found is EXPLORATORY.
