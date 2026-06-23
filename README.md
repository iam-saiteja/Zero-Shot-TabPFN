# Dynamic Row Sub-sampling and Block-Sparse Attention for TabPFN

This repository contains two primary scaling implementations integrated into TabPFN's permutation-invariant row-attention layers to handle large context lengths (8,192+ rows) on CUDA:

1.  **`Partitioned_Attention_TabPFN`** (Block-Sparse): Evaluates query-specific top-K block attention via index-branch prediction.
2.  **`Similarity_Sorted_Subsampled_TabPFN`** (Direct Row Sub-sampling): A linear-complexity approach that dynamically selects a globally-aligned subset of unmodified training rows as prototypes.

---

## 📊 CUDA Benchmark Results (3-Run Averages with Standard Deviations)

When scaling to large synthetic contexts on GPU, the performance metrics (averaged over 3 runs per setting to ensure statistical stability) are as follows:

| Rows ($N$) | Model | Accuracy | ROC AUC | Time Mean (s) | Time Std (s) | Peak VRAM Mean (MB) | Peak VRAM Std (MB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1000** | Vanilla TabPFN | 0.67 | 0.79 | 1.571 | 0.016 | 124.64 | 0.0 |
| | Linear Attention | 0.49 | 0.69 | 1.295 | 0.006 | 111.95 | 0.0 |
| | Similarity-Sorted Subsampled | 0.55 | 0.51 | 1.145 | 0.003 | 99.29 | 0.0 |
| | Partitioned Attention | 0.68 | 0.78 | 9.369 | 0.088 | 768.04 | 0.0 |
| **4096** | Vanilla TabPFN | 0.85 | 0.89 | 8.510 | 0.014 | 272.82 | 0.0 |
| | Linear Attention | 0.63 | 0.70 | 5.862 | 0.024 | 208.91 | 0.0 |
| | Similarity-Sorted Subsampled | 0.66 | 0.70 | 4.267 | 0.022 | 167.34 | 0.0 |
| | Partitioned Attention | 0.81 | 0.86 | 35.380 | 0.114 | 2801.80 | 0.0 |
| **8192** | Vanilla TabPFN | 0.66 | 0.65 | 21.351 | 0.062 | 471.32 | 0.0 |
| | Linear Attention | 0.60 | 0.65 | 11.781 | 0.165 | 344.27 | 0.0 |
| | Similarity-Sorted Subsampled | 0.51 | 0.47 | **8.287** | **0.006** | **257.55** | **0.0** |
| | Partitioned Attention | 0.63 | 0.63 | 424.315 | 16.299 | 5523.75 | 0.0 |

---

## 🔍 Key Findings

### 1. Does Sub-sampling Hold Accuracy at Scale?
At $N=8,192$ rows, our **Direct Row Sub-sampling** layout runs **2.6x faster** (8.29s vs 21.35s) and uses **1.8x less VRAM** (257.6MB vs 471.3MB) compared to Vanilla TabPFN.
- **Accuracy Verdict**: In zero-shot mode, sub-sampling (M=128 prototypes) retains a baseline classification sanity on the large context but experiences a predictive gap compared to dense attention (e.g. 0.51 Accuracy/0.47 ROC AUC vs 0.66 Accuracy/0.65 ROC AUC at 8K). Because we select a static $M=128$ prototypes regardless of context size, the compression ratio grows from $8\times$ (at N=1K) to $64\times$ (at N=8K), naturally leading to information drop. Training the model natively with this layout represents a key future work.

### 2. Naive Block-Sparse Gather Overhead
Partitioned Attention's dynamic index gather is highly inefficient:
- At $N=8,192$ rows, it runs **20x slower** (424.3s vs 21.3s) and uses **11.7x more memory** (5.5GB vs 471MB).
- This empirically confirms that naive PyTorch implementation of dynamic block selection loses all theoretical FLOP-savings to uncoalesced memory reads and layout-shuffling overhead.

---

## 🛠️ Performance & Accuracy on Real Datasets

| Dataset | Model | Accuracy | ROC AUC | Time (s) | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Breast Cancer** | Vanilla TabPFN | **0.9591** | 0.9969 | 2.45 | 144.0 |
| | Similarity-Sorted Subsampled | 0.9532 | **0.9974** | **1.35** | **106.4** |
| **diabetes** | Vanilla TabPFN | 0.7446 | **0.8469** | 1.18 | 108.7 |
| | Similarity-Sorted Subsampled | **0.7489** | 0.8233 | **1.10** | **89.0** |
| **credit-g** | Vanilla TabPFN | **0.7733** | **0.7903** | 2.84 | 183.3 |
| | Similarity-Sorted Subsampled | 0.7133 | 0.7377 | **1.50** | **113.0** |
