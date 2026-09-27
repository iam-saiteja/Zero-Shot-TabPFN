# H2 analysis (smoke test only, per protocol.md's stated scope)

All 4 locked sanity checks + the gate passed:
1. Vanilla deterministic.
2. `M >= train_size` fallback == vanilla exactly (bit-identical).
3. `tf_col` (column embedding, already ISAB) provably untouched — same
   `forward` object identity before/after patching `tf_icl`.
4. Patched output: correct shape, all finite.
Gate: patched-vs-vanilla mean abs diff 0.0084 (relative 7.2%) at M=32,
train_size=300 — same order of magnitude as H1's real-data numbers
(0.02-0.06), not the >0.2 that would mean something broke.

**One thing caught and corrected, not hidden:** the first run used
`zero_init=True` (TabICL's default) — the untrained model's attention
output projection starts at exactly zero, making attention a no-op
regardless of what the patch does. That run reported a suspicious "0.0000
diff" that would have passed even with a broken patch. Reran with
`zero_init=False` to get a real signal; 0.0084 above is that rerun.

## What this does and doesn't show

**Shows:** the patch is correctly scoped (only `tf_icl`, verified by object
identity, not just by construction) and produces finite, shape-correct
output whose divergence from exact attention is in the same range H1
measured on real data and a real trained model.

**Does not show:** any real accuracy or speed number. This is an untrained,
randomly-initialized, tiny model (`embed_dim=32`, 2 ICL blocks vs the real
checkpoint's larger config) — nowhere near TabICLv2's actual pretrained
weights. Also only implements the coreset-anchor half of Barnes-Hut (see
`code/patch.py`'s scope note) — near-field exact expansion needs
`attn_mask`-based masking (each query needs a different near-field row set,
which a single shared k/v tensor can't express) and is the next real step,
not yet built.

## Next steps, in order

1. Load TabICLv2's actual pretrained checkpoint (not random init) and rerun
   this same patch — first real signal on whether it holds on trained
   weights, still without touching real data.
2. Extend `patch.py` with the near-field exact-expansion half via
   `attn_mask`, matching what H1 actually validated (not just the anchor
   half).
3. Only then: real datasets, via `harness/` for genuine cached TabArena
   comparisons instead of another custom pilot script.
