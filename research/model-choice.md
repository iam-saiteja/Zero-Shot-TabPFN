# Which model to build on (decided 2026-09-27)

Requirement from the user: open, permissively licensed (no non-commercial /
research-only restriction), strong current TabArena standing. Checked the
license of every model in `research/landscape.md`'s table directly:

| Model | License | Verdict |
|---|---|---|
| **LimiX-2** (current #1, 1935 Elo) | StableAI LimiX Non-Commercial License v1.0 — no commercial use | **Excluded** |
| **TabPFN-3 / 3.5** | `tabpfn-3-license-v1.0` — research/internal-eval only, no commercial or production use of model, derivatives, or outputs | **Excluded** |
| **TabICLv2** | Code: BSD-3-Clause / Apache-2.0 (the `forecast` module). No commercial restriction. Pretraining code, inference code, and weights all released. | **Chosen** |
| **Mitra / Mitra-v2** | Apache-2.0, fully open (AutoGluon) | Viable secondary |

**Decision: TabICLv2** is the primary target for both stages. Reasons beyond
the license:

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
