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

# Append workspace path to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- COMPATIBILITY PATCHES ---
import typing
import torch.nn.modules.transformer
torch.nn.modules.transformer.Optional = typing.Optional

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
# -----------------------------

from tabpfn import TabPFNClassifier
from nsatabpfn.wrapper import restore_vanilla_tabpfn, inject_nsatabpfn

def get_hardware_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return psutil.Process().memory_info().rss / (1024 ** 2)

def generate_synthetic_data(n_samples, n_features=20):
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X.numpy(), y.numpy()

def run_unlimited_single_process_scaling():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw_name = get_hardware_name()
    
    print("=" * 70)
    print("NSA-TABPFN EXTREME SCALING BENCHMARK SUITE")
    print("=" * 70)
    print(f"Target Hardware : {hw_name}")
    print(f"Device Active   : {device.upper()}")
    print("Dataset Profile : Synthetic Tabular Classification (20 features, binary labels)")
    
    if not torch.cuda.is_available():
        print("\n" + "!" * 70)
        print("WARNING: CUDA is NOT active in PyTorch. The benchmark is running on CPU.")
        print("To run on the RTX 3090 Ti GPU, reinstall PyTorch matching your CUDA 12.2 driver:")
        print("  uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121")
        print("!" * 70 + "\n")
        
    print("=" * 70)
    
    configs = [
        ('NSA-TabPFN (M=64)', 'NSA-TabPFN', 64),
        ('NSA-TabPFN (M=128)', 'NSA-TabPFN', 128),
        ('NSA-TabPFN (M=256)', 'NSA-TabPFN', 256),
        ('Vanilla TabPFN', 'Vanilla TabPFN', 128)
    ]
    
    results = []
    
    for label, model_name, M in configs:
        print(f"\n=== Benchmarking Configuration: {label} ===")
        
        N = 1024
        while True:
            print(f"Evaluating N = {N:,} rows...")
            
            # 1. Clean memory before execution
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 2. Inject patch or restore vanilla
            if model_name == 'Vanilla TabPFN':
                restore_vanilla_tabpfn()
            else:
                inject_nsatabpfn(num_prototypes=M)
                
            # 3. Run evaluation inside try/except block
            try:
                X_train, y_train = generate_synthetic_data(N)
                X_test, y_test = generate_synthetic_data(100)
                
                clf = TabPFNClassifier(device=device, N_ensemble_configurations=1)
                
                start_mem = get_vram_usage()
                start_time = time.time()
                
                clf.fit(X_train, y_train, overwrite_warning=True)
                probs = clf.predict_proba(X_test)
                
                exec_time = time.time() - start_time
                peak_mem = max(0.0, get_vram_usage() - start_mem)
                
                preds = probs.argmax(axis=1)
                acc = (preds == y_test).mean()
                
                print(f"-> Success: {exec_time:.2f}s, Peak Memory: {peak_mem:.2f} MB, Acc: {acc:.4f}")
                results.append({
                    'Model': label, 'N': N, 
                    'Latency (s)': exec_time, 'Peak Memory (MB)': peak_mem,
                    'Accuracy': acc, 'Hardware': hw_name, 'Status': 'Success'
                })
                
                # Free memory immediately
                del clf, X_train, y_train, X_test, y_test, probs
                
                # Double sequence length
                N *= 2
                
            except Exception as e:
                err_msg = str(e)
                print(f"-> Configuration {label} hit resource ceiling/OOM on N = {N:,}: {err_msg}")
                results.append({
                    'Model': label, 'N': N, 
                    'Latency (s)': np.nan, 'Peak Memory (MB)': np.nan,
                    'Accuracy': np.nan, 'Hardware': hw_name, 'Status': f'Failed/OOM ({err_msg})'
                })
                
                # Clear references on exception
                try:
                    del clf, X_train, y_train, X_test, y_test
                except:
                    pass
                break
                
    # Restore to clean state
    restore_vanilla_tabpfn()
            
    # Save raw data
    if results:
        df = pd.DataFrame(results)
        df.to_csv("server_extreme_scaling_results.csv", index=False)
        print("\nResults saved to server_extreme_scaling_results.csv")
    
        # Plotting
        os.makedirs("assets", exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        colors = {
            'Vanilla TabPFN': '#e74c3c', 
            'NSA-TabPFN (M=64)': '#3498db', 
            'NSA-TabPFN (M=128)': '#2ecc71', 
            'NSA-TabPFN (M=256)': '#9b59b6'
        }
        
        # Time comparison
        plt.figure(figsize=(10, 6))
        df_plot_time = df.dropna(subset=['Latency (s)'])
        sns.lineplot(data=df_plot_time, x='N', y='Latency (s)', hue='Model', marker='o', linewidth=2.5, palette=colors)
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.xlabel('Sequence Length N (Rows)')
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Dynamic Inference Latency Scaling Limit ({hw_name})')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig('assets/server_extreme_scaling_time.png', dpi=300)
        plt.close()
        
        # Memory comparison
        plt.figure(figsize=(10, 6))
        df_plot_mem = df.dropna(subset=['Peak Memory (MB)'])
        sns.lineplot(data=df_plot_mem, x='N', y='Peak Memory (MB)', hue='Model', marker='s', linewidth=2.5, linestyle='--', palette=colors)
        plt.xscale('log', base=2)
        plt.xlabel('Sequence Length N (Rows)')
        plt.ylabel('Peak Memory Allocation (MB)')
        plt.title(f'Dynamic Peak Memory Allocation Scaling Limit ({hw_name})')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig('assets/server_extreme_scaling_memory.png', dpi=300)
        plt.close()
        
        print("Generated scaling charts in assets/")
    else:
        print("\nNo successful runs to plot.")

if __name__ == "__main__":
    run_unlimited_single_process_scaling()
