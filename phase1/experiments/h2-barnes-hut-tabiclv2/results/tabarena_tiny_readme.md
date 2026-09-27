# TabArena-tiny eval: internal result is real, Elo comparison is not (yet)

Ran on the 13 classification datasets from the real
`harness/tabarena_tiny_results_reference` (the actual TabArena-tiny dataset
names, not another arbitrary list). 2 skipped on a real data-loading bug in
`tabarena_tiny_eval.py` (`Is-this-a-good-customer` has a raw string column
like `'male'`, `Marketing_Campaign` has a date string — my dtype-detection
missed them; not a method issue, just an unhandled column type).

## Internal comparison (valid): Barnes-Hut vs anchor-only

Same pipeline for both, so this comparison is apples-to-apples even though
the absolute numbers below aren't leaderboard-comparable yet (see next
section). **Barnes-Hut beats anchor-only on 9/11 datasets** — real, but not
the clean 16/16 the earlier `sweep.py` run showed. Two real exceptions:
`Fitness_Club` (0.183 anchor vs 0.393 barnes-hut — anchor much better) and
`MIC` (0.174 vs 0.199 — anchor better). Not explained away — could be
dataset-specific (both have fewer effective clusters worth splitting at
this train size, unconfirmed), could be single-split noise (this run has no
repeated splits, unlike the 2-seed sweep). Open question, not resolved.

## Why no Elo number: our "vanilla" doesn't match the official cached baseline

Checked our own vanilla (exact attention, same pipeline) against the real
cached `TABICLV2 (default)` row for the same datasets:

| dataset | official cached error | our vanilla error |
|---|---|---|
| Fitness_Club | 0.191 | 0.352 |
| MIC | 0.469 | 0.099 |
| anneal | 0.032 | 0.016 |
| blood-transfusion-service-center | 0.255 | 0.272 |
| credit-g | 0.211 | 0.190 |
| diabetes | 0.155 | 0.212 |
| hazelnut-spread-contaminant-detection | 0.007 | 0.035 |
| maternal_health_risk | 0.354 | 0.524 |
| qsar-biodeg | 0.064 | 0.066 |
| seismic-bumps | 0.214 | 0.303 |
| website_phishing | 0.204 | 0.221 |

Several are close (anneal, credit-g, qsar-biodeg, website_phishing), but
MIC, Fitness_Club, maternal_health_risk, and seismic-bumps diverge a lot —
MIC especially (0.469 vs 0.099, our number looks suspiciously good, which is
itself a red flag, not a win). This means our pipeline isn't equivalent to
real TabICLv2 usage: this rig (a) uses crude preprocessing (raw
`.astype(float32).fillna(-1)`, blunt categorical codes) instead of
`TabICLClassifier`'s real preprocessing, (b) calls `_train_forward` directly
instead of the real `predict_proba` path (`forward_with_cache` — see H2's
`docs/research-log.md` entry on the KV-cache gap), and (c) uses one ad hoc
70/30 split, not TabArena's actual fold protocol.

**Decision: not computing an Elo/leaderboard number from this.** Merging our
rows with the official cache and running `bencheval.compute_elo` right now
would produce a number that looks authoritative but rests on a baseline
mismatch — exactly the kind of unverified-claim pattern this whole project
has been trying to avoid (see the original `scratch_architecture_analysis.md`
episode, `docs/research-log.md`, 2026-09-26). Stated honestly instead of
computed and reported anyway.

## What's actually needed before a trustworthy Elo number is possible

Two real gaps, not small ones:
1. Route the patch through `TabICLClassifier`'s real preprocessing and its
   real inference path (`forward_with_cache`), not `_train_forward` — this
   is the deferred KV-cache integration noted earlier, now confirmed to
   matter for more than just "production realism."
2. Match (or get close to) TabArena's actual fold protocol, not an ad hoc
   single split.

This is a real, separate chunk of engineering work, not a quick fix — worth
a decision on how much further to invest before doing it.
