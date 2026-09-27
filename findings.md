# Findings (project memory; update after each outer loop)

## Current understanding
- Base model in repo is TabPFN v1 (tabpfn==0.1.11), not v2+/Mitra; Elo claims need a port.
- Synthetic sim: k-means anchors cut attention error 2-4x vs random; near-field expansion a further 5-15x (synthetic, error-only).
- Top TabArena Elo ~1935 (LimiX-2, 400M params); exact chunked attention (Chunked TabPFN) already exists.

## Lessons and constraints
- JEPA for tabular FMs: negative result in the literature. RL not needed (supervised CE is already a proper scoring rule).
- Log-count mass weighting alone is not a free win.
- Keep fine-tuning corpus disjoint from eval sets (else it is not a zero-shot entry).

## Open questions
- Do real TabPFN embeddings cluster well enough for near-field expansion to help? (H1 answers this)
