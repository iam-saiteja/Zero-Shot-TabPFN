# Which model to build on (decided 2026-09-27, revised same day)

**Revision:** initially excluded LimiX-2 and TabPFN-3 outright for having
non-commercial/research-only licenses. That was too strict — this is
non-commercial research, not a product, and both licenses explicitly permit
exactly that:

| Model | License | Research/eval use? | Commercial/production use? |
|---|---|---|---|
| **LimiX-2** (current #1, 1935 Elo) | StableAI LimiX Non-Commercial License v1.0 | Allowed (non-commercial) | Not allowed |
| **TabPFN-3 / 3.5** | `tabpfn-3-license-v1.0` | Explicitly allowed — "testing, evaluation, and internal benchmarking" | Not allowed, model/derivatives/outputs |
| **TabICLv2** | BSD-3-Clause / Apache-2.0 | Allowed | Allowed |
| **Mitra / Mitra-v2** | Apache-2.0 | Allowed | Allowed |

So all four are usable for this project. The real, remaining tradeoff isn't
legal — it's what a derivative built on a non-commercial base can do later:

- Build on **TabICLv2/Mitra**: the result (code, any improved weights) is
  freely reusable by anyone who reads the eventual paper — no restriction on
  reproducing or building further on it, commercial or not.
- Build on **LimiX-2/TabPFN-3**: fine for the research and the paper itself,
  but the derivative inherits the non-commercial restriction — nobody
  (including future us) could turn the result into a product or a freely
  redistributable open release without renegotiating licensing. Submitting
  to TabArena's leaderboard for evaluation is still fine either way (that's
  exactly the "internal benchmarking" both licenses call out).

**Decision: TabICLv2 stays the primary target**, specifically because the
research goal includes publishing an open, freely reusable result — not
because LimiX-2/TabPFN-3 are off-limits. Reasons beyond the license:

1. It's the strongest fully-open model — beats RealTabPFN-2.5 untuned on
   TabArena (per `research/landscape.md`).
2. It has a genuine, unsolved efficiency problem that matches Stage 1's ask
   directly: pretraining used H100s (~24.5 GPU-days) and inference at 1M
   rows needs **~50GB GPU memory** even with disk offloading. That is not
   consumer-GPU-accessible. Making it consumer-runnable is a real,
   uncontested contribution, not a re-hash of what TabPFN-3 already solved
   for itself (TabPFN-3's row-chunking + multi-query attention gets 1M rows
   under ~7GB — but that model's license excludes us).
3. Its "dataset-wise ICL" stage is the one with **O(n²) cost in rows** —
   structurally the same one-token-per-row attention pattern the H1 pilot
   already validated Barnes–Hut attention against on TabPFN v1. The method
   should port directly; TabICLv2's own report doesn't mention inducing
   points, ISAB, or clustering, so this angle looks genuinely unclaimed there.
4. Pretraining code being open (unlike TabPFN-3) means Stage 2 (fine-tune
   toward a better Elo) is technically possible here — closed weights alone
   (Mitra) would only allow fine-tuning, not touching the pretraining prior
   itself.

**Mitra stays as the fallback** for Stage 2 specifically: it's smaller
(72–75M vs whatever TabICLv2's dataset-wise stage totals — not disclosed),
already Apache-2.0, and already specialized to the small-table niche, so if
TabICLv2 fine-tuning proves too expensive for the available hardware, Mitra
is the cheaper fallback with a real chance of a measurable, if narrow, win.

## Honest scope note

This decision is about *which model to adapt*, not a promise about Elo.
Nothing here changes the math in `research/from-scratch-feasibility.md`:
current #1 is ~1935 Elo (a number that itself will keep moving), an
attention approximation cannot score above exact attention on the same
weights, and TabICLv2's own pretraining cost (~24.5 H100-days) is out of
reach for full reproduction on the hardware available here. Stage 1
(efficiency) is the well-supported claim. Stage 2 (Elo) is the open,
unproven one — see `docs/research-log.md` for what's actually been tried.

## Follow-up worth doing once H2 works on TabICLv2

Once Barnes–Hut attention is validated on TabICLv2's dataset-wise ICL stage,
repeating it on **LimiX-2 (actual current #1)** or **TabPFN-3** (which
already ships its own row-chunking/multi-query answer to the same problem)
is a stronger research result than TabICLv2 alone — "this also works on the
literal #1 model" is a better paper claim than "this works on an open
model." Both are licensed for exactly this (non-commercial research/
evaluation). Keep as a planned follow-up, not the first target, since
TabICLv2 is the one whose result stays freely shareable.
