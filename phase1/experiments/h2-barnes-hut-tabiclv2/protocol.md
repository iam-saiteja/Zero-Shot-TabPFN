# H2: Port Barnes-Hut attention to TabICLv2's dataset-wise ICL stage (LOCKED before running)

**Where, exactly** (verified by reading `tabicl` 2026 source, not the paper text):
`tabicl._model.learning.ICLearning.tf_icl` is a `tabicl._model.encoders.Encoder`
built from `tabicl._model.layers.MultiheadAttentionBlock` blocks. When called
with `train_size=N`, each block does exact attention with
`k = v = q[..., :N, :]` (`layers.py:432-437`) — the query is the full
train+test sequence, keys/values are only the `N` training rows. This is the
O(n^2)-in-rows stage (the ~50GB@1M-rows problem in `research/model-choice.md`),
structurally identical to what H1 validated on TabPFN v1's
`isinstance(src_mask, int)` branch. Confirmed `use_rope=False` here (defaults
to False, `ICLearning.__init__` doesn't override it) — rows are an unordered
set, no positional encoding to worry about, so H1's method (no positional
adjustment needed) ports directly.

Not the same as `tf_col` (column embedding, already uses ISAB/
`InducedSelfAttentionBlock` — a different, already-bounded axis, columns
capped at 100) or `tf_row` (row interaction, also over columns). Patch must
touch only `tf_icl`'s blocks, not `tf_col`/`tf_row`'s, which share the same
`MultiheadAttentionBlock` class.

**Hypothesis.** Instance-level monkeypatching each block in `tf_icl.blocks`
(not the shared class) to replace `k=v=q[:train_size]` with Barnes-Hut
anchors (k-means clusters of the training rows *as seen by that block*,
recomputed per-block like H1 did per-layer; top-t clusters kept exact,
mass-weighted monopoles for the rest) preserves fidelity to exact attention
at a fraction of the rows touched, matching H1's real-data result (median
fidelity error 0.43x of plain k-means anchors, -0.33pp accuracy vs exact at
M=64,t=4 on TabPFN v1).

**This run is a smoke test, not a benchmark.** Scope: verify the patch is
wired correctly (shapes, finiteness, doesn't touch tf_col/tf_row, and
approaches vanilla output as M -> train_size) on synthetic data. It does
NOT measure real accuracy/speed on real TabArena data — that needs the
`harness/` eval (bigger next step, not this run).

**Sanity checks.** (1) vanilla twice == deterministic; (2) patched with
M >= train_size (i.e. M-cap never binds) == vanilla exactly (fallback path);
(3) tf_col/tf_row outputs bit-identical to vanilla when only tf_icl is
patched (proves the patch is properly scoped); (4) all finite.

**Gate to call this a working port (not yet a result):** all 4 sanity
checks pass, and patched-vs-vanilla fidelity error at M=64,t=4 is in the
same order of magnitude as H1's real-data number (0.02-0.06), not >0.2
(which would mean the port broke something H1 didn't have, e.g. rope
interaction or the block-sharing scope).
