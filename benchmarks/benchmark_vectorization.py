import os
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time

# 1. DYNAMIC SYSTEM PATH INCLUSION
# Dynamically locate the project root relative to the directory containing this script.
# This prevents crashes when the repository is moved or executed in different environments.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. DEVICE SELECTION
# Dynamically choose between CUDA and CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running vectorization benchmark on device: {device}")

# 3. NAIVE LOOP IMPLEMENTATION
# This version computes chunk means using a sequential python loop over M chunks.
# It suffers from significant Python CPU-side overhead, kernel launch latency, and
# poor GPU utilization because it performs M small slices and averages.
def _chunk_means_loop(train_rows, M):
    Bc, N, E = train_rows.shape
    perm = torch.randperm(N, device=train_rows.device)
    chunk_size = max(1, N // M)
    protos = []
    for i in range(M):
        s, e = i * chunk_size, min((i + 1) * chunk_size, N)
        if s >= N: s, e = 0, chunk_size
        protos.append(train_rows[:, perm[s:e]].mean(dim=1))
    return torch.stack(protos, dim=1)

# 4. VECTORIZED IMPLEMENTATION (Zero-Shot ISAB Core Optimization)
# This vectorized chunk-averaging implementation avoids loops completely.
# It slices the permutation array down to the nearest multiple of M (M * chunk_size),
# gathers all elements at once using advanced indexing (avoiding launch overhead),
# and uses a view/reshape followed by a single mean operation across the chunk dimension.
def _chunk_means_vectorized(train_rows, M):
    Bc, N, E = train_rows.shape
    device = train_rows.device
    perm = torch.randperm(N, device=device)
    chunk_size = max(1, N // M)
    num_elements = M * chunk_size
    selected_perm = perm[:num_elements]
    # Single gather operation over the permutations: [Bc, M * chunk_size, E]
    gathered = train_rows[:, selected_perm]
    # Reshape to group chunks together: [Bc, M, chunk_size, E]
    gathered = gathered.view(Bc, M, chunk_size, E)
    # Average across the third dimension (chunk_size) to yield M prototype averages in a single step
    return gathered.mean(dim=2)

Bc, N, E = 8, 398, 128
M = 128
x = torch.randn(Bc, N, E, device=device)

# Warmup iterations to compile/cache operations (essential for accurate PyTorch benchmarks)
for _ in range(10):
    _ = _chunk_means_loop(x, M)
    _ = _chunk_means_vectorized(x, M)

if device == "cuda":
    torch.cuda.synchronize()
    
# Benchmark Loop Implementation
t0 = time.time()
for _ in range(100):
    _ = _chunk_means_loop(x, M)
if device == "cuda":
    torch.cuda.synchronize()
t_loop = time.time() - t0

# Benchmark Vectorized Implementation
t0 = time.time()
for _ in range(100):
    _ = _chunk_means_vectorized(x, M)
if device == "cuda":
    torch.cuda.synchronize()
t_vec = time.time() - t0

print(f"Loop implementation (100 runs): {t_loop:.6f}s")
print(f"Vectorized implementation (100 runs): {t_vec:.6f}s")
print(f"Vectorization Speedup: {t_loop / t_vec:.1f}x")



# 5. Generate Vectorization Speedup Plot
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("assets", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.barplot(x=['Sequential Loop', 'Vectorized Pipeline'], y=[t_loop, t_vec], palette=['#e74c3c', '#2ecc71'])
plt.title(f'Chunk-Averaging Execution Time (100 runs)\nSpeedup: {t_loop/t_vec:.1f}x')
plt.ylabel('Time (seconds)')
for i, v in enumerate([t_loop, t_vec]):
    plt.text(i, v + (t_loop*0.02), f'{v:.4f}s', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('assets/vectorization_speedup.png', dpi=300)
plt.close()
print("Saved assets/vectorization_speedup.png")
