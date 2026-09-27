import openml
import time
import torch
import gc
import sys
import os
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from zsisab.wrapper import inject_zsisab, restore_vanilla_tabpfn

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
    
    # OpenML CC18 Datasets (30 standard classification datasets)
    benchmark_suite = openml.study.get_suite('OpenML-CC18')
    dataset_ids = benchmark_suite.data[:30] # Get exactly 30 datasets
    
    results = []
    output_file = "comparison_results_30.csv"
    
    for d_id in dataset_ids:
        print(f"\n{'='*50}")
        print(f"--- Fetching Dataset ID: {d_id} ---")
        try:
            dataset = openml.datasets.get_dataset(d_id, download_data=True)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            
            # Subsample if dataset is too large just to make the 30-dataset loop finish in reasonable time,
            # but user requested scaling, so we'll leave it intact or upsample later if needed.
            
            X = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)) if col.dtype.name in ['category', 'object'] else col)
            X = X.fillna(X.median(numeric_only=True)).fillna(0).values
            y = LabelEncoder().fit_transform(y)
            
            # Skip multiclass to focus on clean binary ROC-AUC
            if len(np.unique(y)) > 2:
                print(f"Skipping {dataset.name} (Multiclass - focusing on binary for ROC-AUC scaling comparison)")
                continue
                
            N_rows = X.shape[0]
            print(f"Dataset: {dataset.name} | Shape: {X.shape}")
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            train_idx, test_idx = next(skf.split(X, y))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            dataset_result = {
                'Dataset_Name': dataset.name,
                'Rows': N_rows,
                'Features': X.shape[1],
            }
            
            # ---------------------------------------------------------
            # 1. VANILLA TABPFN (WITH OOM TRY-CATCH)
            # ---------------------------------------------------------
            print("  -> Testing Vanilla TabPFN...")
            restore_vanilla_tabpfn()
            clear_gpu()
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            
            vanilla_acc, vanilla_roc, vanilla_time, vanilla_vram = np.nan, np.nan, np.nan, np.nan
            try:
                clf = TabPFNClassifier(device=device, N_ensemble_configurations=1)
                start_mem = get_vram_usage()
                start_time = time.time()
                
                clf.fit(X_train, y_train, overwrite_warning=True)
                probs = clf.predict_proba(X_test)
                
                vanilla_time = time.time() - start_time
                vanilla_vram = max(0.0, get_vram_usage() - start_mem)
                
                preds = probs.argmax(axis=1)
                vanilla_acc = accuracy_score(y_test, preds)
                vanilla_roc = roc_auc_score(y_test, probs[:, 1])
                print(f"     [Vanilla] Acc: {vanilla_acc:.4f}, VRAM: {vanilla_vram:.1f}MB, Time: {vanilla_time:.1f}s")
                dataset_result['Vanilla_OOM'] = False
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"     [Vanilla] ❌ CUDA Out Of Memory! Caught safely. Skipping Vanilla for this dataset.")
                    dataset_result['Vanilla_OOM'] = True
                else:
                    print(f"     [Vanilla] ❌ Failed: {str(e)[:100]}")
                    dataset_result['Vanilla_OOM'] = False
            
            # ---------------------------------------------------------
            # 2. ZS-ISAB TABPFN
            # ---------------------------------------------------------
            print("  -> Testing ZS-ISAB (M=512)...")
            inject_zsisab(num_prototypes=512, chunk_size=16384)
            clear_gpu()
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            
            zsisab_acc, zsisab_roc, zsisab_time, zsisab_vram = np.nan, np.nan, np.nan, np.nan
            try:
                clf = TabPFNClassifier(device=device, N_ensemble_configurations=1)
                start_mem = get_vram_usage()
                start_time = time.time()
                
                clf.fit(X_train, y_train, overwrite_warning=True)
                probs = clf.predict_proba(X_test)
                
                zsisab_time = time.time() - start_time
                zsisab_vram = max(0.0, get_vram_usage() - start_mem)
                
                preds = probs.argmax(axis=1)
                zsisab_acc = accuracy_score(y_test, preds)
                zsisab_roc = roc_auc_score(y_test, probs[:, 1])
                print(f"     [ZS-ISAB] Acc: {zsisab_acc:.4f}, VRAM: {zsisab_vram:.1f}MB, Time: {zsisab_time:.1f}s")
                dataset_result['ZS_ISAB_OOM'] = False
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"     [ZS-ISAB] ❌ CUDA Out Of Memory! Caught safely.")
                    dataset_result['ZS_ISAB_OOM'] = True
                else:
                    print(f"     [ZS-ISAB] ❌ Failed: {str(e)[:100]}")
                    dataset_result['ZS_ISAB_OOM'] = False
                    
            dataset_result.update({
                'Vanilla_Acc': vanilla_acc, 'Vanilla_ROC': vanilla_roc, 'Vanilla_VRAM': vanilla_vram, 'Vanilla_Time': vanilla_time,
                'ZS_ISAB_Acc': zsisab_acc, 'ZS_ISAB_ROC': zsisab_roc, 'ZS_ISAB_VRAM': zsisab_vram, 'ZS_ISAB_Time': zsisab_time
            })
            results.append(dataset_result)
            pd.DataFrame(results).to_csv(output_file, index=False)
            
        except Exception as e:
            print(f"Error processing dataset {d_id}: {e}")
            
    # ---------------------------------------------------------
    # 3. GENERATE PLOTS AND AVERAGES
    # ---------------------------------------------------------
    print("\n============================================================")
    print("ALL DATASETS PROCESSED. GENERATING AVERAGES AND PLOTS...")
    print("============================================================")
    
    df = pd.DataFrame(results)
    
    # Filter datasets where BOTH successfully ran to get fair averages
    df_fair = df[(df['Vanilla_OOM'] == False) & (df['ZS_ISAB_OOM'] == False)].dropna()
    
    print("\n--- FAIR AVERAGES (Only datasets where Vanilla didn't OOM) ---")
    print(f"Average Vanilla Accuracy: {df_fair['Vanilla_Acc'].mean():.4f}")
    print(f"Average ZS-ISAB Accuracy: {df_fair['ZS_ISAB_Acc'].mean():.4f}")
    print(f"Average Vanilla VRAM:     {df_fair['Vanilla_VRAM'].mean():.1f} MB")
    print(f"Average ZS-ISAB VRAM:     {df_fair['ZS_ISAB_VRAM'].mean():.1f} MB")
    print(f"Average Vanilla Time:     {df_fair['Vanilla_Time'].mean():.2f} s")
    print(f"Average ZS-ISAB Time:     {df_fair['ZS_ISAB_Time'].mean():.2f} s")
    
    print(f"\nTotal Datasets where Vanilla OOM'd: {df['Vanilla_OOM'].sum()}")
    print(f"Total Datasets where ZS-ISAB OOM'd: {df['ZS_ISAB_OOM'].sum()}")
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Accuracy Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df_fair['Dataset_Name'], df_fair['Vanilla_Acc'], marker='o', label='Vanilla TabPFN')
    plt.plot(df_fair['Dataset_Name'], df_fair['ZS_ISAB_Acc'], marker='s', label='ZS-ISAB (Ours)')
    plt.xticks(rotation=90)
    plt.title('Accuracy Comparison Across Datasets')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    
    # Plot 2: VRAM Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(df_fair)) - 0.2, df_fair['Vanilla_VRAM'], 0.4, label='Vanilla TabPFN')
    plt.bar(np.arange(len(df_fair)) + 0.2, df_fair['ZS_ISAB_VRAM'], 0.4, label='ZS-ISAB (Ours)')
    plt.xticks(np.arange(len(df_fair)), df_fair['Dataset_Name'], rotation=90)
    plt.title('Peak VRAM Comparison')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vram_comparison.png')
    
    # Plot 3: Time Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(df_fair)) - 0.2, df_fair['Vanilla_Time'], 0.4, label='Vanilla TabPFN')
    plt.bar(np.arange(len(df_fair)) + 0.2, df_fair['ZS_ISAB_Time'], 0.4, label='ZS-ISAB (Ours)')
    plt.xticks(np.arange(len(df_fair)), df_fair['Dataset_Name'], rotation=90)
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (Seconds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_comparison.png')
    
    print("\nPlots saved as 'accuracy_comparison.png', 'vram_comparison.png', and 'time_comparison.png'")

if __name__ == '__main__':
    run_benchmarks()
