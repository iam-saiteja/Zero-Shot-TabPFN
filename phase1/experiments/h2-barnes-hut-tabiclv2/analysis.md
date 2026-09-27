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

## Confirmatory sweep: same 10 datasets + 2 seeds as H1, post-fix (`sweep.py`)

M=64, t=4, anchor-only vs Barnes-Hut, against the real pretrained TabICLv2
checkpoint. 2 of 10 datasets (adult, jm1) produced NaN fidelity — root
cause confirmed, not a mystery: this rig calls `_train_forward` directly on
raw feature values (bypassing `TabICLClassifier`'s normal input
normalization), and those two datasets have raw features up to ~1.37M,
which overflows the softmax. Artifact of the test rig, not the method —
excluded from the stats below rather than silently included or hidden.

**Across the remaining 8 real datasets x 2 seeds = 16 comparisons:**

| | mean fidelity | median fidelity | mean agreement | mean acc_delta vs exact |
|---|---|---|---|---|
| anchor-only | 0.111 | 0.104 | 0.873 | -6.19pp |
| Barnes-Hut (t=4) | 0.050 | 0.045 | 0.945 | -1.75pp |

**Barnes-Hut beats anchor-only on fidelity in 16/16 pairs, zero
exceptions.** Roughly halves the fidelity error on average and cuts the
mean accuracy loss vs exact attention from -6.2pp to -1.75pp. This is now a
real, multi-dataset, multi-seed, bug-fixed, mechanistically-understood
result on an actual pretrained SOTA-adjacent model — the strongest evidence
in this project so far that the method (validated on TabPFN v1 in H1)
generalizes to a current model.

**Still true, unchanged:** single train/test split per dataset-seed (not
H1's repeated-fold rigor), still a masked-not-gathered implementation (no
speed win demonstrated yet), still not the real TabArena protocol via
`harness/`, and the two NaN datasets need the rig fixed to go through
proper preprocessing before they can be included.

## Next steps, in order

1. Fix the test rig to use TabICLClassifier's real preprocessing (resolves
   the adult/jm1 NaNs, and gets closer to how the model is actually used).
2. A real sparse/gather kernel — this near-field version still computes
   scores over all `train_size` rows and masks, so it does not yet save
   compute, only tests correctness.
3. Real datasets via `harness/` for genuine cached TabArena comparisons
   instead of another custom pilot script — the real test.

## Built the real inference-path integration ("build that properly")

**Correction of an earlier assumption:** traced the exact call chain and found
`predict_proba`'s DEFAULT config (`kv_cache=False`, which is what every test
so far used) never touches `forward_with_cache` at all — it calls
`self.model_(...)` -> `TabICL.forward` -> `Encoder.forward`, the same method
`patch.py`/`patch_nearfield.py` already patch and validated. The KV-cache
path (`forward_with_cache`) is a separate, opt-in performance feature
(`kv_cache=True`) for reusing cached K/V across repeated `predict_proba`
calls on the same fit. Built `patch_kvcache.py` for that path anyway (real,
correct, instance-scoped patch of the store/use cache phases with per-query
near-field masking against cached centroids) since the work was already
mostly done when the mistake was caught — kept as real infrastructure for
when someone actually uses `kv_cache=True`, not the fix for the baseline
mismatch.

**The actual fix:** run through real preprocessing and the real
`predict_proba()` ensemble path (`real_api_default_path.py`), applying the
already-validated `patch_tf_icl_bh` to `clf.model_` after a normal `.fit()` —
instead of the crude `_train_forward` + raw-feature bypass every earlier H2
script used.

**Result: mean gap to the official cached TabICLv2 baseline dropped from a
wild, untrustworthy mismatch to 0.049** across the same 13 real
TabArena-tiny datasets. Most are now genuinely close (blood-transfusion:
0.0002, hazelnut: 0.0039, qsar-biodeg: 0.0046, website_phishing: 0.0089,
diabetes: 0.0178, seismic-bumps: 0.0188, anneal: 0.0265) — within plausible
range of a single split vs the official multi-fold average. **MIC remains a
real, unexplained outlier** (0.378 gap) — not resolved, flagged rather than
averaged away.

**Barnes-Hut vs vanilla, via the real API:** a modest, real accuracy cost —
mean error increase ~1.1pp across 13 datasets, 2/13 where it actually did
slightly better (plausibly noise), no catastrophic failures. This is the
expected shape for an approximation: real cost, not a wash and not a
disaster.

**This is now a foundation trustworthy enough to compute a real Elo number
against the official cached leaderboard** — the next concrete step, not yet
done.
