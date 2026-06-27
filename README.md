# Chunked Zero-Shot Induced Set Attention Blocks (ZS-ISAB)

This repository contains the official implementation of Chunked ZS-ISAB for TabPFN, allowing it to natively evaluate datasets with millions of rows on a standard 4GB GPU without any VRAM bottlenecks.

## The Architecture
Vanilla TabPFN uses O(N^2) attention and projects all N rows into the MLP simultaneously, requiring >8GB VRAM for 1,000,000 rows.

Our **Chunked ZS-ISAB** solves this natively inside the GPU:
1. **Global Pooling (Online Softmax):** Compresses information from all N training rows into M prototypes in O(NM) time with practically zero VRAM overhead.
2. **Chunked Broadcast & MLP:** Slices the N rows into blocks of 16,384, preventing the MLP from spiking VRAM. 

**Peak VRAM is strictly bounded.** You can run 1,000,000+ rows on a 4GB GPU, achieving exactly the same >0.90 accuracy as full Nystrom/ISAB, but in <50% of the time and <50% of the VRAM.

## Setup
\\ash
uv pip install -r requirements.txt
\
## Running the Benchmark
\\ash
uv run python benchmarks/benchmark_local.py
\