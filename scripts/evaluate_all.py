from __future__ import annotations

import os
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import gc
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set environment token for TabPFN
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_WDvw1MHEYQRQz8NKJBMqEFoink8X-sagyYRMKWM8Vo4"

# We import the architectures to allow dynamic patching
import tabpfn.architectures.tabpfn_v2 as tabpfn_v2
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
import tabpfn.architectures.tabpfn_v2_6 as tabpfn_v2_6
from zsisab.baselines import AlongColumnAttentionMSA, AlongColumnAttentionLinear, AlongColumnAttentionISAB, AlongColumnAttentionTopKBlock
from zsisab.wrapper import inject_zsisab_into_tabpfn, restore_vanilla_tabpfn, patch_tabpfn_load_state_dict
from tabpfn import TabPFNClassifier
from zsisab.data_generator import generate_scm_dataset

# Helper to clear CUDA memory
def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# Preprocessing helper
def preprocess_dataset(X, y):
    X = pd.DataFrame(X).copy()
    for col in X.columns:
        if X[col].dtype == "object" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median() if X[col].dtype != "object" else X[col].mode()[0])
            
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X.values, y

def evaluate_trees(X_train, X_test, y_train, y_test, model_type="lgb"):
    if model_type == "lgb":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)
    elif model_type == "xgb":
        import xgboost as xgb
        model = xgb.XGBClassifier(random_state=42, verbosity=0, n_estimators=100)
    elif model_type == "cat":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(random_seed=42, verbose=0, iterations=100)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    start_time = time.time()
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    infer_time = time.time() - start_time
    
    preds = np.array(preds).ravel()
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs[:, 1])
    
    return acc, auc, fit_time + infer_time

def evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="vanilla", msa_strategy="similarity"):
    clear_gpu()
    
    if variant == "linear":
        tabpfn_v2.AlongColumnAttention = AlongColumnAttentionLinear
        tabpfn_v2_5.AlongColumnAttention = AlongColumnAttentionLinear
        tabpfn_v2_6.AlongColumnAttention = AlongColumnAttentionLinear
        patch_tabpfn_load_state_dict()
    elif variant == "isab_naive":
        tabpfn_v2.AlongColumnAttention = AlongColumnAttentionISAB
        tabpfn_v2_5.AlongColumnAttention = AlongColumnAttentionISAB
        tabpfn_v2_6.AlongColumnAttention = AlongColumnAttentionISAB
        patch_tabpfn_load_state_dict()
    elif variant == "zs_isab":
        inject_zsisab_into_tabpfn(num_prototypes=128)
    elif variant == "msa":
        class TempMSA(AlongColumnAttentionMSA):
            def __init__(self, *args, **kwargs):
                kwargs["blocking_strategy"] = msa_strategy
                kwargs["topk"] = 4
                kwargs["blk_kv"] = 32
                super().__init__(*args, **kwargs)
                
        tabpfn_v2.AlongColumnAttention = TempMSA
        tabpfn_v2_5.AlongColumnAttention = TempMSA
        tabpfn_v2_6.AlongColumnAttention = TempMSA
        patch_tabpfn_load_state_dict()
    else:
        restore_vanilla_tabpfn()
        
    try:
        from tabpfn.constants import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
            
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        preds = clf.predict(X_test)
        
        elapsed_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs[:, 1])
        
        return acc, auc, elapsed_time, peak_vram
    except Exception as e:
        print(f"Error evaluating variant={variant}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0
    finally:
        restore_vanilla_tabpfn()

def main():
    print("====================================================")
    restore_vanilla_tabpfn()

    # 1. Evaluate on Real Datasets
    print("--- Phase 5: Evaluation on Real OpenML/Toy Datasets ---")
    datasets = {
        "Breast Cancer": lambda: load_breast_cancer(return_X_y=True),
        "credit-g": lambda: fetch_openml(data_id=31, return_X_y=True, as_frame=True),
        "diabetes": lambda: fetch_openml(data_id=37, return_X_y=True, as_frame=True),
    }
    
    results = []
    
    for name, load_fn in datasets.items():
        print(f"\nEvaluating dataset: {name}...")
        try:
            X, y = load_fn()
            X, y = preprocess_dataset(X, y)
            
            if X.shape[0] > 1000:
                X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=42, stratify=y)
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            print(f"Shape: Train={X_train.shape}, Test={X_test.shape}")
            
            # Trees
            for tree_type in ["lgb", "xgb", "cat"]:
                acc, auc, elapsed = evaluate_trees(X_train, X_test, y_train, y_test, tree_type)
                results.append({"Dataset": name, "Model": f"Tuned_{tree_type.upper()}", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": 0.0})
                
            # Vanilla TabPFN
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="vanilla")
            results.append({"Dataset": name, "Model": "Vanilla_TabPFN", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
            
            # Linear Attention TabPFN (Baseline)
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="linear")
            results.append({"Dataset": name, "Model": "Linear_Attention_TabPFN (Baseline)", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
            
            # Naive ISAB TabPFN (Baseline)
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="isab_naive")
            results.append({"Dataset": name, "Model": "ISAB_Naive_TabPFN (Baseline)", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})

            # Zero-Shot ISAB TabPFN
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="zs_isab")
            results.append({"Dataset": name, "Model": "Zero-Shot ISAB (Ours)", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
                
        except Exception as e:
            print(f"Failed to process dataset {name}: {e}")
            
    df_results = pd.DataFrame(results)
    print("\n==================== REAL DATASET RESULTS ====================")
    print(df_results.to_markdown(index=False))
    df_results.to_csv("real_dataset_evaluation_results.csv", index=False)
 
    # 2. Benchmark scaling on large synthetic SCM datasets
    print("\n--- Phase 5: Scaling Benchmark on Synthetic SCM Datasets ---")
    sizes = [1000, 2048, 4096, 8192]
    scaling_results = []
    
    for size in sizes:
        print(f"\nScaling Benchmark: N={size} rows...")
        X, y = generate_scm_dataset(num_samples=size + 100, num_features=10, task_type="classification", num_classes=2, random_state=42)
        X, y = preprocess_dataset(X, y)
        X_train, X_test = X[:size], X[size:size+100]
        y_train, y_test = y[:size], y[size:size+100]
        
        variants = {
            "Vanilla_TabPFN": ("vanilla", None),
            "Linear_Attention_TabPFN": ("linear", None),
            "Zero-Shot ISAB (Ours)": ("zs_isab", None),
            "Partitioned_Attention_TabPFN": ("msa", "random")
        }
        
        for model_name, (variant, strategy) in variants.items():
            print(f"  Benchmarking {model_name}...")
            times = []
            vrams = []
            accs = []
            aucs = []
            # Run 3 times for stats stability
            for run_i in range(3):
                acc, auc, elapsed, vram = evaluate_tabpfn_variant(
                    X_train, X_test, y_train, y_test, 
                    variant=variant, msa_strategy=strategy
                )
                times.append(elapsed)
                vrams.append(vram)
                accs.append(acc)
                aucs.append(auc)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            mean_vram = np.mean(vrams)
            std_vram = np.std(vrams)
            mean_acc = np.mean(accs)
            mean_auc = np.mean(aucs)
            
            scaling_results.append({
                "Rows": size, 
                "Model": model_name, 
                "Accuracy": mean_acc,
                "ROC_AUC": mean_auc,
                "Time Mean (s)": mean_time, 
                "Time Std (s)": std_time, 
                "Peak VRAM Mean (MB)": mean_vram,
                "Peak VRAM Std (MB)": std_vram
            })
        
    df_scaling = pd.DataFrame(scaling_results)
    print("\n==================== SCALING SCENARIO RESULTS ====================")
    print(df_scaling.to_markdown(index=False))
    df_scaling.to_csv("scaling_evaluation_results.csv", index=False)
    generate_plots(df_results, df_scaling)


import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(df_results, df_scaling):
    print("\n--- Generating Plots ---")
    os.makedirs("assets", exist_ok=True)
    
    # 1. Accuracy Grouped Bar Chart
    if not df_results.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_results, x='Dataset', y='Accuracy', hue='Model')
        plt.title('Zero-Shot Accuracy Comparison Across Datasets')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('assets/evaluation_accuracy.png', dpi=300)
        plt.close()
        print("Saved assets/evaluation_accuracy.png")

    # 2. Scaling Performance Line Charts
    if not df_scaling.empty:
        # Time Scaling
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_scaling, x='Rows', y='Time Mean (s)', hue='Model', marker='o')
        plt.title('Execution Time vs Sequence Length (Linear Scaling)')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Number of Rows (N)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('assets/scaling_time.png', dpi=300)
        plt.close()
        print("Saved assets/scaling_time.png")
        
        # VRAM Scaling
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_scaling, x='Rows', y='Peak VRAM Mean (MB)', hue='Model', marker='o')
        plt.title('Peak VRAM vs Sequence Length (Linear Scaling)')
        plt.ylabel('VRAM (MB)')
        plt.xlabel('Number of Rows (N)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('assets/scaling_vram.png', dpi=300)
        plt.close()
        print("Saved assets/scaling_vram.png")

if __name__ == "__main__":
    main()
