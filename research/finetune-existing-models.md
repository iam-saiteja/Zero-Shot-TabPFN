# Fine-tuning an existing checkpoint instead of training from scratch

Sources: [On Finetuning Tabular Foundation Models](https://arxiv.org/html/2506.08982v2),
[Attention Quantization for Tabular Foundation Models](https://arxiv.org/abs/2609.13031),
[TabTune (GitHub)](https://github.com/Lexsi-Labs/TabTune).

## Short answer to "can we fine-tune + quantize an existing model": yes,
## and it's probably a better first move than from-scratch pretraining.

Three real findings change the recommendation from
`research/from-scratch-feasibility.md`:

**1. There's already an MIT-licensed library that does exactly this.**
[TabTune](https://github.com/Lexsi-Labs/TabTune) supports inference *and*
fine-tuning (full fine-tune, LoRA/PEFT, episodic meta-learning fine-tuning,
and native provider APIs) across 16 tabular foundation models including
TabPFN, TabICL, **Mitra**, TabDPT, LimiX, and TabFM. This directly closes
the "no training infrastructure exists yet" gap logged in
`docs/research-log.md` — we don't need to write a training loop or a
synthetic-prior generator to get started; we need to pick a checkpoint and
call `pipeline.fit(X, y)` with `tuning_strategy="finetune"`.

**2. LoRA/quantization matters less than expected here, for a specific
reason.** "On Finetuning Tabular Foundation Models" found **full
fine-tuning is the most practical approach for TabPFNv2** — it converges
faster and matches or slightly beats PEFT/LoRA on accuracy. PEFT only wins
when memory is the binding constraint. That's still us (4GB), so PEFT/LoRA
is still worth having available, but it's a memory-fit tool, not an
accuracy trick — don't expect it to help beyond what it's for.

**3. The tabular-specific quantization lever is different from the LLM
playbook.** "Attention Quantization for Tabular Foundation Models" (Sept
2026) found **weight quantization (the QLoRA playbook) gives limited
benefit for tabular FMs because the models are already small** — there
isn't much weight memory to save. What actually helps: quantizing the
**attention computation itself** (Q/K/V to FP8), which got a real, measured
**1.7x speedup with no accuracy loss on TabPFN-v3 and TabICLv2** via a
custom Triton kernel. If we need more headroom on the 4GB card, this is the
better-targeted lever than generic weight quantization — and it's
conceptually adjacent to our own `zsisab/engine.py` chunked attention kernel,
so there's a plausible combination here (chunked + FP8 attention) worth
trying before writing anything from scratch.

## Why this fits what we already decided, better than starting over

We already picked "specialize into the small-table niche" as the
realistic angle (see `research/from-scratch-feasibility.md`, citing Mitra's
own reported strength at <5,000 rows / <100 features). **Mitra is exactly
that checkpoint, already trained, already available**
([autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier)
on Hugging Face). Fine-tuning it further — rather than prior-fitting a new
small model from zero to try to reach the same niche — starts from
something already competitive instead of from nothing. That's strictly less
compute for a real shot at the same target.

## The honest catch: this has to stay a one-time fine-tune, not per-dataset

TabArena's leaderboard distinguishes zero-shot models from tuned ones (the
original README table you had labeled this exact column: "Zero-Shot /
Tuned"). Fine-tuning **once**, on a broad corpus the model didn't see during
its own pretraining (e.g. a chunk of the TabZilla-168 datasets already
sitting in `datasets/`, or a curated diverse small-table corpus), then
evaluating **zero-shot on held-out TabArena tasks it never trained on** —
that's still a legitimate zero-shot entry, the same way Mitra itself is
"pretrained once, applied zero-shot everywhere." Fine-tuning **per target
dataset at eval time** would instead be a "tuned" entry, competing in a
different, more crowded tier (alongside HPO'd XGBoost etc.) — a different,
weaker claim than the one the research direction has been aiming for. Keep
the fine-tuning corpus and the eval corpus disjoint, or this quietly becomes
a different (less interesting) project.

## Recommended next step

1. `pip install tabtune`, load `autogluon/mitra-classifier` in inference mode
   first — get a real, current baseline number on a few held-out TabZilla
   datasets before touching any weights. (Still haven't replaced the stale
   README claims with a real measured number — this does that for free as a
   side effect.)
2. Fine-tune (full fine-tune first, per the paper's finding; fall back to
   LoRA only if it doesn't fit in 4GB) on a *different* subset of datasets
   than step 1's eval set.
3. Re-evaluate on the same held-out set from step 1. That delta is the
   first real, honest number in this whole research thread.

This is a few hours of work, not days — worth doing before committing to
either the from-scratch pretraining plan or asking for the 3090 Ti.
