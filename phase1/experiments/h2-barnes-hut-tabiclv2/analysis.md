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

## Update: real checkpoint + near-field expansion (`real_checkpoint_check*.py`)

Real pretrained TabICLv2 (`tabicl-classifier-v2-20260212.ckpt`, 12 ICL
blocks, embed_dim 128), real data (OpenML phoneme, same dataset H1 used),
600 train / 200 test rows. Still bypasses the sklearn cache path (direct
`_train_forward` call) and still a single dataset/split, no seed averaging
— exploratory, not confirmatory like H1.

**Anchor-only is clearly insufficient here** — worse than H1's TabPFN v1
numbers:

| M | pred-agreement with exact |
|---|---|
| 16 | 0.805 |
| 32 | 0.820-0.825 |
| 64 | 0.835-0.840 |
| 128 | 0.940 |

**Near-field expansion (`patch_nearfield.py`, mask-based, not yet a fast
kernel) helps prediction agreement, but not the raw fidelity metric — a
genuinely mixed result, not spun either way:**

| Variant | fidelity (mean abs softmax diff) | pred-agreement |
|---|---|---|
| anchor-only, M=32 | 0.149 | 0.825 |
| barnes-hut, M=32, t=2/4/8 | 0.157 / 0.145 / 0.148 | 0.920 / 0.930 / 0.925 |
| anchor-only, M=64 | 0.127 | 0.840 |
| barnes-hut, M=64, t=2/4/8 | 0.164 / 0.150 / 0.152 | 0.895 / 0.925 / 0.930 |

Near-field pushes prediction agreement up ~9-10 points (82-84% -> 92-93%)
but the softmax-fidelity number barely moves and is sometimes marginally
worse. Two metrics disagreeing like this means something real is going on
that isn't understood yet — possibilities, not conclusions: (a) fidelity
(mean abs diff over the whole softmax vector) may be dominated by small
shifts in low-probability classes that don't flip any decision, while
agreement (argmax) captures what actually matters for accuracy; (b) 12
independently-reclustered layers may compound noise differently than H1's
setup; (c) single dataset/split — could just be this dataset. Not resolved,
flagged honestly rather than picking whichever metric looks better.

**t doesn't monotonically help** (t=2 worse than t=4 at both M) — plausibly
single-run noise given no seed averaging yet, not a claim either way.

### Found and fixed the cause: monopole entries were undercounted

Root cause of the fidelity/agreement disagreement above: `log_counts` (how
many real rows each cluster summarizes) was only used to pick which
clusters get exact near-field treatment (the `topk` call) — it was never
actually added as a bias to the real attention logits for the surviving
monopole entries. So a cluster standing in for 50 real rows was attended to
with the same softmax weight as a single row: systematically wrong
probability mass, explaining why fidelity didn't move while argmax
agreement (which survives even with wrong relative mass, as long as the
biggest class stays biggest) did. Fixed by adding `log(count)` as an
additive bias on the monopole positions in `attn_mask` (matching the design
already validated as the right one in `research/sim_anchor_attention.py`'s
"monopole" variant — this was an implementation gap, not a design error).

**After the fix, same setup as above:**

| Variant | fidelity | pred-agreement |
|---|---|---|
| anchor-only, M=32 | 0.171 | 0.830 |
| barnes-hut, M=32, t=2/4/8 | 0.064 / 0.045 / 0.034 | 0.935 / 0.970 / 0.960 |
| anchor-only, M=64 | 0.134 | 0.840 |
| barnes-hut, M=64, t=2/4/8 | 0.055 / 0.049 / 0.025 | 0.940 / 0.960 / 0.965 |

Both metrics now agree and move together: fidelity drops 3-6x versus
anchor-only (into the same 0.02-0.06 range H1 measured on real TabPFN v1
data), and it's now roughly monotonic in `t` (M=64: 0.055 -> 0.049 -> 0.025
as t=2 -> 4 -> 8), which is what theory predicts and the pre-fix numbers
did not show. This is a real, mechanistically-understood result, not a
metric-picking exercise.

## Next steps, in order

1. **Not done:** repeat with H1's rigor — multiple seeds, multiple datasets
   — before trusting any of the numbers above as more than a first look.
2. Investigate the fidelity-vs-agreement disagreement rather than pick the
   metric that looks better.
3. A real sparse/gather kernel — this near-field version still computes
   scores over all `train_size` rows and masks, so it does not yet save
   compute, only tests correctness.
4. Only then: real datasets via `harness/` for genuine cached TabArena
   comparisons instead of another custom pilot script.
