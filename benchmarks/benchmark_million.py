import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 1. DYNAMIC SYSTEM PATH INCLUSION
# Dynamically locate the project root relative to the directory containing this script.
# This prevents crashes when the repository is moved or executed in different environments.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set the environment token for TabPFN authentication
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_WDvw1MHEYQRQz8NKJBMqEFoink8X-sagyYRMKWM8Vo4"

import tabpfn.architectures.tabpfn_v2 as tabpfn_v2
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
import tabpfn.architectures.tabpfn_v2_6 as tabpfn_v2_6
from tabpfn_msa import AlongColumnAttentionTwoPass
from tabpfn import TabPFNClassifier
from zsisab.data_generator import generate_scm_dataset
from evaluate import preprocess_dataset, clear_gpu
from sklearn.metrics import accuracy_score, roc_auc_score

def run_isab_benchmark(size):
    # Ensure a clean slate on GPU memory before running the benchmark
    clear_gpu()
    
    # 2. Zero-Shot ISAB HYPERPARAMETER & ARCHITECTURE PATCHING
    # We dynamically patch the TabPFN attention layers to use our AlongColumnAttentionTwoPass (Zero-Shot ISAB).
    # num_prototypes (M): Sets the number of compressed induce points.
    class TempISAB(AlongColumnAttentionTwoPass):
        def __init__(self, *args, **kwargs):
            kwargs["num_prototypes"] = 128  # Use 128 prototypes for large scale
            super().__init__(*args, **kwargs)
            
    # Patch all the versioned module definitions of along-column attention to use the Zero-Shot ISAB implementation
    tabpfn_v2.AlongColumnAttention = TempISAB
    tabpfn_v2_5.AlongColumnAttention = TempISAB
    tabpfn_v2_6.AlongColumnAttention = TempISAB
    
    # Relax strict checking of state dict when loading weights because Zero-Shot ISAB layers
    # contain additional hyperparameters/buffers not present in the vanilla checkpoints.
    original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
    tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)
        
    try:
        # Generate a synthetic causal dataset using Structural Causal Models
        X, y = generate_scm_dataset(num_samples=size+100, num_features=10, task_type="classification", num_classes=2, random_state=42)
        X, y = preprocess_dataset(X, y)
        X_train, X_test = X[:size], X[size:]
        y_train, y_test = y[:size], y[size:]
        
        # 3. DEVICE SELECTION
        # Dynamically determine device: GPU is highly recommended for N=1M to avoid CPU timeout/slowness.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print(f"[WARNING] Running N={size} on CPU. This might take a very long time!")
            
        from tabpfn.constants import ModelVersion
        
        # 4. ignore_pretraining_limits
        # TabPFN has built-in constraints that raise exceptions for datasets larger than its pre-training limit.
        # Setting ignore_pretraining_limits=True allows zero-shot execution on arbitrary context sizes.
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, 
            device=device, 
            ignore_pretraining_limits=True
        )
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        preds = clf.predict(X_test)
        elapsed = time.time() - start_time
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs[:, 1])
        
        return elapsed, peak_vram, acc, auc, "Success"
        
    except torch.cuda.OutOfMemoryError as oom:
        # Clear cache immediately on OOM
        clear_gpu()
        return 0.0, 0.0, 0.0, 0.0, f"CUDA OOM Error: {str(oom)}"
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0, str(e)
    finally:
        # Restore original architecture files by reloading modules, ensuring clean slate for subsequent benchmarks
        import importlib
        importlib.reload(tabpfn_v2)
        importlib.reload(tabpfn_v2_5)
        importlib.reload(tabpfn_v2_6)
        clear_gpu()

# Run the benchmark on sizes scaling up to 1 Million rows
sizes = [65536, 131072, 262144, 524288, 1048576]
print("=========================================")
print("MILLION-ROW SCALING BENCHMARK (Zero-Shot ISAB)")
print("=========================================")

results_mil = []
for size in sizes:
    t, vram, acc, auc, status = run_isab_benchmark(size)
    if status == "Success":
        print(f"N={size}: Acc={acc:.4f} AUC={auc:.4f} | Time={t:.1f}s VRAM={vram:.1f}MB")
        results_mil.append({'Rows': size, 'Time': t, 'VRAM': vram})
    else:
        print(f"N={size} failed: {status}")
        break

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if results_mil:
    df_mil = pd.DataFrame(results_mil)
    plt.figure(figsize=(10, 6))
    plt.plot(df_mil['Rows'], df_mil['Time'], marker='o', color='blue', label='Time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Million-Row Scaling Benchmark: Time vs Rows (Log-Log)')
    plt.xlabel('Number of Rows (N)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('assets/million_row_scaling.png', dpi=300)
    plt.close()
    print("Saved assets/million_row_scaling.png")


