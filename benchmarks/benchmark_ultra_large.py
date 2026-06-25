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

def evaluate_variant(X_train, X_test, y_train, y_test, variant="vanilla"):
    # Ensure a clean slate on GPU memory before running the benchmark
    clear_gpu()
    
    # 2. Zero-Shot ISAB HYPERPARAMETER & ARCHITECTURE PATCHING
    if variant == "isab":
        class TempISAB(AlongColumnAttentionTwoPass):
            def __init__(self, *args, **kwargs):
                kwargs["num_prototypes"] = 128  # Use 128 prototypes for large scale
                super().__init__(*args, **kwargs)
        # Patch the versioned along-column attention to use the Zero-Shot ISAB implementation
        tabpfn_v2.AlongColumnAttention = TempISAB
        tabpfn_v2_5.AlongColumnAttention = TempISAB
        tabpfn_v2_6.AlongColumnAttention = TempISAB
        
        # Relax strict checking of state dict when loading weights because Zero-Shot ISAB layers
        # contain additional hyperparameters/buffers not present in the vanilla checkpoints.
        original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
        tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)
    else:
        # Reload architecture files to revert monkey-patching and restore vanilla full-attention behavior
        import importlib
        importlib.reload(tabpfn_v2)
        importlib.reload(tabpfn_v2_5)
        importlib.reload(tabpfn_v2_6)
        
    try:
        from tabpfn.constants import ModelVersion
        
        # 3. DEVICE SELECTION & ignore_pretraining_limits
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set ignore_pretraining_limits=True. Vanilla attention fails/errors on N >= 16384 without this,
        # because TabPFN checks if context length exceeds its pretraining limit (typically 10,000).
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, 
            device=device,
            ignore_pretraining_limits=True
        )
        
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        elapsed = time.time() - start_time
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        return elapsed, peak_vram, "Success"
        
    except torch.cuda.OutOfMemoryError as oom:
        clear_gpu()
        return 0.0, 0.0, f"CUDA OOM Error: {str(oom)}"
    except Exception as e:
        return 0.0, 0.0, str(e)
    finally:
        # Restore original architecture files to ensure clean slate for subsequent iterations
        import importlib
        importlib.reload(tabpfn_v2)
        importlib.reload(tabpfn_v2_5)
        importlib.reload(tabpfn_v2_6)
        clear_gpu()

# Run the comparative benchmark on sizes N=8K, 16K, and 32K
sizes = [8192, 16384, 32768]
print("=========================================")
print("ULTRA-LARGE SCALING BENCHMARK (N=8K..32K)")
print("=========================================")
for size in sizes:
    X, y = generate_scm_dataset(num_samples=size+100, num_features=10, task_type="classification", num_classes=2, random_state=42)
    X, y = preprocess_dataset(X, y)
    X_train, X_test = X[:size], X[size:]
    y_train, y_test = y[:size], y[size:]
    
    t_v, vr_v, s_v = evaluate_variant(X_train, X_test, y_train, y_test, variant="vanilla")
    t_i, vr_i, s_i = evaluate_variant(X_train, X_test, y_train, y_test, variant="isab")
    
    status_v = f"{t_v:.2f}s ({vr_v:.1f}MB)" if s_v == "Success" else f"Failed: {s_v}"
    status_i = f"{t_i:.2f}s ({vr_i:.1f}MB)" if s_i == "Success" else f"Failed: {s_i}"
    print(f"N={size}: Vanilla={status_v} | Zero-Shot ISAB={status_i}")

