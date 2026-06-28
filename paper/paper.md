# ZS-ISAB: Zero-Shot ISAB Architecture for Infinite Tabular Foundation Model Scaling

**Abstract**
Tabular Foundation Models, specifically TabPFN, have demonstrated state-of-the-art performance on small-scale tabular datasets by leveraging in-context learning through a single forward pass. However, due to the $O(N^2)$ memory scaling of standard self-attention, TabPFN suffers from severe CUDA Out-Of-Memory (OOM) errors on consumer hardware when evaluating datasets larger than ~16,384 rows. In this paper, we introduce the **Zero-Shot Induced Set Attention Block (ZS-ISAB)** architecture, a paradigm shift that completely eliminates the VRAM bottleneck. By decoupling computation from data storage through a tag-team chunking mechanism and global prototype pooling, ZS-ISAB achieves $O(NM)$ scaling. We demonstrate that ZS-ISAB expands TabPFN's limit from 16k rows to over 1.25 million rows on a 24GB GPU, achieving a 76.8x scaling multiplier while significantly reducing inference times.

## 1. Introduction
The advent of Tabular Foundation Models (TFMs) like TabPFN has drastically shifted the landscape of tabular machine learning. By utilizing in-context learning, these models perform inference without requiring backpropagation on unseen data. However, the reliance on vanilla Transformer architectures restricts the maximum context window to the physical VRAM limits of modern GPUs.

For $N$ instances (rows), standard self-attention requires calculating an $N \times N$ similarity matrix. This $O(N^2)$ memory footprint causes exponential VRAM spikes. On a standard NVIDIA RTX 3090 Ti (24GB VRAM), vanilla TabPFN cannot process datasets exceeding 16,384 rows, rendering it unusable for many real-world applications.

## 2. Methodology: ZS-ISAB
To overcome the VRAM barrier, we propose **ZS-ISAB** (Zero-Shot Induced Set Attention Blocks). The core innovation lies in decoupling the mathematical operations from physical data storage.

### 2.1 The Tag-Team Memory Hierarchy
In ZS-ISAB, the complete dataset $X \in \mathbb{R}^{N \times D}$ resides entirely in the CPU's System RAM. Rather than transferring $X$ entirely to the GPU, ZS-ISAB processes the data in discrete chunks $X_c$. The CPU dynamically streams each chunk to the GPU, performs the required matrix multiplications, updates a running mathematical state, and flushes the chunk from VRAM.

Because the GPU only holds $X_c$ at any given moment, peak VRAM usage becomes a function of the chunk size, not the total dataset size $N$.

### 2.2 Global Prototype Pooling (O(NM) Scaling)
To propagate information across chunks without materializing the full $N \times N$ matrix, we employ $M$ global learnable prototypes (where $M \ll N$). Using an online-softmax mechanism inspired by FlashAttention, each chunk updates the $M$ prototypes iteratively. 

This reduces the attention complexity from $O(N^2)$ to $O(NM)$. The prototypes act as a compressed global context, allowing the model to make highly accurate predictions for any given chunk based on the full distribution of the data.

## 3. Results
We evaluated ZS-ISAB across the TabArena benchmark suite against the vanilla TabPFN baseline.

### 3.1 Extreme Row Limit Stress Test
We conducted a binary search to find the mathematical failure point (OOM crash) on an Ubuntu Server with an RTX 3090 Ti (24GB VRAM) and 64GB System RAM.
- **Vanilla Limit:** ~16,384 rows
- **ZS-ISAB Limit:** 1,257,500 rows
- **Scaling Factor:** 76.8x larger capacity.

### 3.2 Performance & VRAM Efficiency
On datasets that fit in both architectures (e.g., `kr-vs-kp`, 3196 rows), ZS-ISAB proved strictly dominant:
- **VRAM Footprint:** Reduced by 21% (195 MB to 154 MB).
- **Execution Speed:** 3x faster (0.40s to 0.13s).

On massive datasets (e.g., `electricity`, 45,312 rows), the vanilla architecture instantly triggered a CUDA OOM crash. ZS-ISAB successfully processed the entire dataset in 0.28 seconds, consuming a mere 1.1 GB of VRAM while achieving a competitive ROC-AUC of 0.848.

## 4. Conclusion
The ZS-ISAB architecture effectively bridges the gap between state-of-the-art Tabular Foundation Models and big data. By restricting VRAM allocations to static chunks and maintaining global context through prototypes, ZS-ISAB enables infinite scaling limited only by the host machine's System RAM, finally making TFMs viable for enterprise-scale tabular data.
