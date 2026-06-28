import time
import torch
import gc
import sys
import os
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from tabpfn import TabPFNClassifier
from zsisab.wrapper import inject_zsisab, restore_vanilla_tabpfn

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return psutil.Process().memory_info().rss / (1024 ** 2)

def generate_synthetic_data(n_samples, n_features=20, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X.numpy(), y.numpy()

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def run_1_million():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 1_000_000
    
    print("=" * 60)
    print(f"1 MILLION ROW STRESS TEST (N={N})")
    print("=" * 60)
    
    try:
        # Pre-allocate to see if it even fits in CPU RAM
        print("Generating 1M rows...")
        X_train, y_train = generate_synthetic_data(N)
        X_test, y_test = generate_synthetic_data(500, seed=99)
        print("Data generated successfully!")
        
        clear_gpu()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Use high hyperparameters as user requested
        inject_zsisab(num_prototypes=1024, chunk_size=16384)
        
        clf = TabPFNClassifier(device=device, N_ensemble_configurations=1)
        start_mem = get_vram_usage()
        start_time = time.time()
        
        print("Starting fit & predict...")
        clf.fit(X_train, y_train, overwrite_warning=True)
        probs = clf.predict_proba(X_test)
        
        exec_time = time.time() - start_time
        peak_mem = max(0.0, get_vram_usage() - start_mem)
        
        preds = probs.argmax(axis=1)
        acc = (preds == y_test).mean()
        
        print(f"ZS-ISAB (1 Million) | Acc: {acc:.4f} | VRAM: {peak_mem:7.2f} MB | Time: {exec_time:5.2f}s")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == '__main__':
    run_1_million()
