# Dynamic Row Sub-sampling and Block-Sparse Attention for TabPFN

This repository contains two primary scaling implementations integrated into TabPFN's permutation-invariant row-attention layers to handle large context lengths (8,192+ rows) on CUDA:

1.  **`Partitioned_Attention_TabPFN`** (Block-Sparse): Evaluates query-specific top-K block attention via index-branch prediction.
2.  **`Similarity_Sorted_Subsampled_TabPFN`** (Direct Row Sub-sampling): A linear-complexity approach that dynamically selects a globally-aligned subset of unmodified training rows as prototypes.

---

## 📊 CUDA Scaling Results ($N = 8,192$ rows)

When scaling to large synthetic contexts on GPU, the performance metrics are as follows:

| Model | Complexity | Time (s) | Peak VRAM (MB) | Note / Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Vanilla TabPFN** | $O(N^2)$ | 18.22 | 471.3 | Baseline |
| **Linear Attention** | $O(N)$ | 9.39 | 344.3 | High accuracy loss |
| **Similarity-Sorted Subsampled (Ours)** | $O(N \cdot M)$ | **6.70** | **257.6** | **Fastest, lowest VRAM** |
| **Partitioned Attention (Block-Sparse)** | $O(N \cdot \text{blk\_kv})$ | **241.31** | **5523.75** | **Highly inefficient gather overhead** |

### Why is Partitioned Attention Slow?
Even though it reduces the theoretical FLOP count, the exact block gather step:
```python
flat_k_idx = batch_indices * (H * S_k_padded) + head_indices * S_k_padded + gather_indices
k_gathered = k_flat[flat_k_idx]
```
requires a massive, non-contiguous memory gather operation across batches, heads, queries, and tokens. On GPU, this triggers uncoalesced memory reads, and on CPU, it lacks SIMD vectorization. Consequently, it runs **13x slower** and uses **11x more memory** than Vanilla TabPFN.

---

## 🛠️ Performance & Accuracy on Real Datasets

| Dataset | Model | Accuracy | ROC AUC | Time (s) | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Breast Cancer** | Vanilla TabPFN | **0.9591** | 0.9969 | 2.89 | 144.0 |
| | Similarity-Sorted Subsampled | 0.9532 | **0.9974** | **1.19** | **106.4** |
| **diabetes** | Vanilla TabPFN | 0.7446 | **0.8469** | 1.11 | 108.7 |
| | Similarity-Sorted Subsampled | **0.7489** | 0.8233 | **1.04** | **89.0** |
| **credit-g** | Vanilla TabPFN | **0.7733** | **0.7903** | 2.40 | 183.3 |
| | Similarity-Sorted Subsampled | 0.7133 | 0.7377 | **1.33** | **113.0** |

*Note: There remains a zero-shot accuracy gap of ~5.6% on `credit-g`. Fine-tuning the projection layers to adapt to the sub-sampling mechanics remains a direction for future work.*
