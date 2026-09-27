import openml
import time
import torch
import gc
import sys
import os
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

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
from zsisab.wrapper import inject_zsisab

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return psutil.Process().memory_info().rss / (1024 ** 2)

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def run_benchmarks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # OpenML CC18 Datasets (representative subset of binary classification datasets)
    dataset_ids = [
        31, # credit-g
        37, # diabetes
        1461, # bank-marketing
        1464, # blood-transfusion
        1468, # cnae-9
        1489, # phoneme
        1590, # adult
        41138, # APSFailure
        43551, # Employee Turnover
        43922, # kick-starters
    ]
    
    results = []
    output_file = "tab_arena_results.csv"
    
    inject_zsisab(num_prototypes=512, chunk_size=16384)
    print("Injected ZS-ISAB architecture (num_prototypes=512, chunk_size=16384)")
    
    for d_id in dataset_ids:
        print(f"\n--- Fetching Dataset ID: {d_id} ---")
        try:
            dataset = openml.datasets.get_dataset(d_id, download_data=True)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            
            # Basic preprocessing
            X = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)) if col.dtype.name in ['category', 'object'] else col)
            X = X.fillna(X.median(numeric_only=True)).fillna(0).values
            y = LabelEncoder().fit_transform(y)
            
            if len(np.unique(y)) > 2:
                print(f"Skipping {dataset.name} (Multiclass - not supported in this simplified ROC-AUC loop)")
                continue
                
            print(f"Dataset: {dataset.name} | Shape: {X.shape} | Classes: {len(np.unique(y))}")
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            fold_acc, fold_roc, fold_vram, fold_time = [], [], [], []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                clear_gpu()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                clf = TabPFNClassifier(device=device, N_ensemble_configurations=32)
                
                start_mem = get_vram_usage()
                start_time = time.time()
                
                clf.fit(X_train, y_train, overwrite_warning=True)
                probs = clf.predict_proba(X_test)
                
                exec_time = time.time() - start_time
                peak_mem = max(0.0, get_vram_usage() - start_mem)
                
                preds = probs.argmax(axis=1)
                acc = accuracy_score(y_test, preds)
                roc = roc_auc_score(y_test, probs[:, 1])
                
                fold_acc.append(acc)
                fold_roc.append(roc)
                fold_vram.append(peak_mem)
                fold_time.append(exec_time)
                print(f"  Fold {fold+1}: Acc={acc:.4f}, ROC={roc:.4f}, VRAM={peak_mem:.1f}MB, Time={exec_time:.1f}s")
                
            res_dict = {
                'Dataset_ID': d_id,
                'Dataset_Name': dataset.name,
                'Rows': X.shape[0],
                'Features': X.shape[1],
                'Mean_Accuracy': np.mean(fold_acc),
                'Mean_ROC_AUC': np.mean(fold_roc),
                'Mean_VRAM_MB': np.mean(fold_vram),
                'Mean_Time_s': np.mean(fold_time)
            }
            results.append(res_dict)
            
            pd.DataFrame(results).to_csv(output_file, index=False)
            
        except Exception as e:
            print(f"Error processing dataset {d_id}: {e}")
            
    print(f"\nBenchmark completed. Results saved to {output_file}")
    print(pd.DataFrame(results))

if __name__ == '__main__':
    run_benchmarks()
