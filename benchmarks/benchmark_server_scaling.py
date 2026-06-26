import time
import torch
import psutil
import gc
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

# --- PYTORCH COMPATIBILITY PATCH FOR TABPFN 0.1.11 ---
import typing
import torch.nn.modules.transformer
torch.nn.modules.transformer.Optional = typing.Optional
# ---------------------------------------------------

from tabpfn import TabPFNClassifier

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return psutil.Process().memory_info().rss / (1024 ** 3)

def generate_synthetic_data(n_samples, n_features=20):
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X.numpy(), y.numpy()

def run_extreme_scaling():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Extreme Scaling Benchmark on {device}")
    
    # Enable ZS-ISAB
    from zsisab.wrapper import inject_zsisab_into_tabpfn
    inject_zsisab_into_tabpfn(num_prototypes=128)
    
    # Scale from 65k to 8.3 Million
    row_counts = [65536, 262144, 1048576, 4194304, 8388608]
    times = []
    vrams = []
    
    clf = TabPFNClassifier(device=device, N_ensemble_configurations=1)
    
    for N in row_counts:
        print(f"\\nEvaluating N = {N:,} rows...")
        clear_gpu()
        
        try:
            X, y = generate_synthetic_data(N)
            
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            clf.fit(X, y)
            _ = clf.predict_proba(X[:10]) # Force inference graph evaluation
            
            end_time = time.time()
            exec_time = end_time - start_time
            peak_vram = get_vram_usage()
            
            times.append(exec_time)
            vrams.append(peak_vram)
            
            print(f"Success: {exec_time:.2f}s, VRAM: {peak_vram:.2f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM on {N} rows")
            else:
                print(f"Error: {str(e)}")
            break
            
    # Save raw data
    np.savez("server_scaling_results.npz", row_counts=row_counts[:len(times)], times=times, vrams=vrams)
    
    # Plotting
    os.makedirs("assets", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Context Sequence Length N (Rows)')
    ax1.set_ylabel('Execution Time (seconds)', color=color)
    ax1.plot(row_counts[:len(times)], times, marker='o', color=color, linewidth=2, label="Time")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Peak VRAM Allocation (GB)', color=color)
    ax2.plot(row_counts[:len(vrams)], vrams, marker='s', color=color, linewidth=2, linestyle='--', label="VRAM")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('ZS-ISAB True $\mathcal{O}(N)$ Scaling Limit (RTX 3090 Ti, 24GB VRAM)')
    fig.tight_layout()
    plt.savefig('assets/server_scaling_log.png', dpi=300)
    print("\\nGenerated assets/server_scaling_log.png")

if __name__ == "__main__":
    run_extreme_scaling()
