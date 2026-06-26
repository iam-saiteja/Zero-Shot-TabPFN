import time
import torch
import psutil
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests')))

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

# --- SCIKIT-LEARN COMPATIBILITY PATCH FOR TABPFN 0.1.11 ---
import sklearn.utils.validation
import sklearn.utils
original_check_X_y = sklearn.utils.validation.check_X_y
original_check_array = sklearn.utils.validation.check_array

def patched_check_X_y(*args, **kwargs):
    kwargs.pop('force_all_finite', None)
    return original_check_X_y(*args, **kwargs)

def patched_check_array(*args, **kwargs):
    kwargs.pop('force_all_finite', None)
    return original_check_array(*args, **kwargs)

sklearn.utils.validation.check_X_y = patched_check_X_y
sklearn.utils.validation.check_array = patched_check_array
sklearn.utils.check_X_y = patched_check_X_y
sklearn.utils.check_array = patched_check_array
# ----------------------------------------------------------

from tabpfn import TabPFNClassifier

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return psutil.Process().memory_info().rss / (1024 ** 2)

def generate_synthetic_data(n_samples, n_features=20):
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X.numpy(), y.numpy()

def run_extreme_scaling():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Extreme Scaling Benchmark on {device}")
    
    # Scale from 65k to 8.3 Million
    row_counts = [65536, 131072, 262144, 524288] # Scaled down range slightly to avoid massive CPU waits if running on CPU
    
    results = []
    
    for model_name in ['Vanilla TabPFN', 'NSA-TabPFN']:
        print(f"\n--- Benchmarking Model: {model_name} ---")
        
        for N in row_counts:
            print(f"Evaluating N = {N:,} rows...")
            clear_gpu()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Inject or Restore
            from nsatabpfn.wrapper import restore_vanilla_tabpfn, inject_nsatabpfn
            if model_name == 'Vanilla TabPFN':
                restore_vanilla_tabpfn()
            else:
                inject_nsatabpfn(num_prototypes=128)
                
            try:
                X_train, y_train = generate_synthetic_data(N)
                X_test, y_test = generate_synthetic_data(100) # Small evaluation set
                
                clf = TabPFNClassifier(device=device, N_ensemble_configurations=1)
                
                start_mem = get_vram_usage()
                start_time = time.time()
                
                clf.fit(X_train, y_train, overwrite_warning=True)
                probs = clf.predict_proba(X_test)
                
                exec_time = time.time() - start_time
                peak_mem = max(0.0, get_vram_usage() - start_mem)
                
                preds = probs.argmax(axis=1)
                acc = (preds == y_test).mean()
                
                print(f"Success: {exec_time:.2f}s, Peak Memory: {peak_mem:.2f} MB, Acc: {acc:.4f}")
                results.append({
                    'Model': model_name, 'N': N, 
                    'Latency (s)': exec_time, 'Peak Memory (MB)': peak_mem,
                    'Accuracy': acc
                })
            except Exception as e:
                print(f"Model {model_name} failed/OOM on N = {N:,}: {str(e)}")
                break
                
    # Restore to clean state
    from nsatabpfn.wrapper import restore_vanilla_tabpfn
    restore_vanilla_tabpfn()
            
    # Save raw data
    if results:
        df = pd.DataFrame(results)
        df.to_csv("server_scaling_results.csv", index=False)
        print("\nResults saved to server_scaling_results.csv")
    
        # Plotting
        os.makedirs("assets", exist_ok=True)
        device_name = "RTX 3090 Ti GPU" if torch.cuda.is_available() else "Server CPU"
        
        # Time comparison
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='N', y='Latency (s)', hue='Model', marker='o', linewidth=2)
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.xlabel('Sequence Length N (Rows)')
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Execution Time Scaling Limit ({device_name})')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig('assets/server_scaling_time.png', dpi=300)
        plt.close()
        
        # Memory comparison
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='N', y='Peak Memory (MB)', hue='Model', marker='s', linewidth=2, linestyle='--')
        plt.xscale('log', base=2)
        plt.xlabel('Sequence Length N (Rows)')
        plt.ylabel('Peak Memory Allocation (MB)')
        plt.title(f'Peak Memory Usage Scaling Limit ({device_name})')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig('assets/server_scaling_memory.png', dpi=300)
        plt.close()
        
        print("Generated scaling charts in assets/")
    else:
        print("\nNo successful runs to plot.")

if __name__ == "__main__":
    run_extreme_scaling()
