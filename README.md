# Permutation-Invariant Sparse & Induced Set Attention for Tabular Foundation Models

This repository contains the implementation of **linear-complexity $O(N)$ row-attention** layers integrated into TabPFN's permutation-invariant tabular classification backbone. By replacing the default dense $O(N^2)$ row-wise self-attention, we enable TabPFN to scale to large context lengths (8,192+ rows) without out-of-memory (OOM) failures or execution bottlenecks on CPU and memory-constrained GPUs.

---

## 🚀 Key Achievements & Results

Our primary method, **`Similarity_Sorted_ISAB_TabPFN`** ($M=128$), bridges the gap between scalability and zero-shot predictive performance:

### 1. Classification Performance (Zero-Shot)

| Dataset | Model | Accuracy | ROC AUC | Time (s) | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Breast Cancer** | Vanilla TabPFN | **0.9591** | 0.9969 | 1.98 | 144.0 |
| | Similarity-Sorted ISAB | 0.9532 | **0.9973** | **1.10** | **106.4** |
| **diabetes** | Vanilla TabPFN | 0.7446 | **0.8469** | 1.05 | 108.7 |
| | Similarity-Sorted ISAB | **0.7489** | 0.8233 | **0.99** | **89.0** |
| **credit-g** | Vanilla TabPFN | **0.7733** | **0.7903** | 2.33 | 183.3 |
| | Similarity-Sorted ISAB | 0.7133 | 0.7377 | **1.28** | **113.0** |

### 2. Large Dataset Scaling (Synthetic SCM Datasets)

| Rows ($N$) | Model | Time (s) | Peak VRAM (MB) | Scaling vs Vanilla |
| :---: | :--- | :---: | :---: | :---: |
| **1000** | Vanilla TabPFN | 1.36s | 124.6 | 1.0x (Baseline) |
| | Similarity-Sorted ISAB | **0.99s** | **99.3** | **1.4x Faster** |
| **4096** | Vanilla TabPFN | 7.11s | 272.8 | 1.0x (Baseline) |
| | Similarity-Sorted ISAB | **3.39s** | **167.3** | **2.1x Faster** |
| **8192** | Vanilla TabPFN | 17.76s | 471.3 | 1.0x (Baseline) |
| | Similarity-Sorted ISAB | **6.46s** | **257.6** | **2.7x Faster, 1.8x Lower VRAM** |

---

## 🛠️ Key Architectural Innovations

1. **Globally Aligned Row Permutations**: Standard multi-head token sorting or row partitioning sorts features independently per column. For TabPFN's column-wise attention blocks, this scrambles feature correspondences. We resolved this by computing a Z-score normalized similarity projection across columns, averaging the projection scores, and applying a **single, globally aligned row permutation (`perm_base`)** across all columns.
2. **Direct Row Sub-sampling**: Averaging clusters of rows to build prototypes acts as a low-pass filter, blurring decision boundaries and degrading zero-shot accuracy. We solved this by using **Direct Row Sub-sampling**: selecting actual, unmodified training rows as prototypes.
3. **Distribution-Spanning linspace Indexing**: Instead of spacing indices using integer division (`N // M`), which misses the tail of the sorted elements, we use `torch.linspace(0, N - 1, steps=M).long()`. This guarantees that selected prototypes span the entire data distribution.

---

## 📁 Repository Structure

*   `tabpfn_msa.py`: Contains the sparse attention architectures:
    *   `AlongColumnAttentionISAB`: Sub-sampled, globally aligned Induced Set Attention (O(N) complexity).
    *   `AlongColumnAttentionTwoPass`: Soft-clustering based two-pass prototype attention.
    *   `AlongColumnAttentionMSA`: Block-sparse row attention using `MiniMaxSparseAttentionPyTorch`.
*   `evaluate.py`: The evaluation framework running real-dataset classifications and CPU scaling benchmarks.
*   `msa_pytorch.py`: PyTorch fallback for block-sparse attention mechanics.

---

## 💻 How to Run the Evaluation

To evaluate accuracy and scaling performance:

```bash
.venv\Scripts\python evaluate.py
```
