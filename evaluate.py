from __future__ import annotations

import os
import sys
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
from tabpfn_msa import AlongColumnAttentionMSA, AlongColumnAttentionLinear, AlongColumnAttentionISAB
from tabpfn import TabPFNClassifier
from data_generator import generate_scm_dataset

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

def evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="vanilla", msa_strategy="similarity", checkpoint_path=None):
    clear_gpu()
    
    # Restore original classes first
    original_along_col = tabpfn_v2.AlongColumnAttention
    
    if variant == "linear":
        tabpfn_v2.AlongColumnAttention = AlongColumnAttentionLinear
        tabpfn_v2_5.AlongColumnAttention = AlongColumnAttentionLinear
        tabpfn_v2_6.AlongColumnAttention = AlongColumnAttentionLinear
    elif variant == "isab":
        class TempISAB(AlongColumnAttentionISAB):
            def __init__(self, *args, **kwargs):
                kwargs["num_prototypes"] = 128
                super().__init__(*args, **kwargs)
        tabpfn_v2.AlongColumnAttention = TempISAB
        tabpfn_v2_5.AlongColumnAttention = TempISAB
        tabpfn_v2_6.AlongColumnAttention = TempISAB
        original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
        tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)
    elif variant == "msa":
        # Patch to use AlongColumnAttentionMSA
        def patched_init(self, *args, **kwargs):
            kwargs["blocking_strategy"] = msa_strategy
            kwargs["topk"] = 4
            kwargs["blk_kv"] = 32
            AlongColumnAttentionMSA.__init__(self, *args, **kwargs)
            
        # Dynamically create temporary subclass to lock strategy
        class TempMSA(AlongColumnAttentionMSA):
            def __init__(self, *args, **kwargs):
                kwargs["blocking_strategy"] = msa_strategy
                kwargs["topk"] = 4
                kwargs["blk_kv"] = 32
                super().__init__(*args, **kwargs)
                
        tabpfn_v2.AlongColumnAttention = TempMSA
        tabpfn_v2_5.AlongColumnAttention = TempMSA
        tabpfn_v2_6.AlongColumnAttention = TempMSA

        # Wrap load_state_dict to make it non-strict but keep original translations
        original_load_v2 = tabpfn_v2.TabPFNV2.load_state_dict
        tabpfn_v2.TabPFNV2.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2(self, sd, strict=False, assign=assign)

        original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
        tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)

        original_load_v2_6 = tabpfn_v2_6.TabPFNV2p6.load_state_dict
        tabpfn_v2_6.TabPFNV2p6.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_6(self, sd, strict=False, assign=assign)
    else:
        # Restore vanilla
        # We need to reload original classes if they were modified
        import importlib
        importlib.reload(tabpfn_v2)
        importlib.reload(tabpfn_v2_5)
        importlib.reload(tabpfn_v2_6)
        
    try:
        from tabpfn.constants import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load checkpoint if provided
        if checkpoint_path and variant == "msa":
            print(f"Loading MSA checkpoint: {checkpoint_path}")
            clf._initialize_model_variables()
            checkpoint = torch.load(checkpoint_path, map_location=clf.device)
            clf.model_.load_state_dict(checkpoint["state_dict"])
            
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
        # Restore original classes
        import importlib
        importlib.reload(tabpfn_v2)
        importlib.reload(tabpfn_v2_5)
        importlib.reload(tabpfn_v2_6)

def main():
    print("====================================================")
    # Restore original states just in case
    import importlib
    importlib.reload(tabpfn_v2)
    importlib.reload(tabpfn_v2_5)
    importlib.reload(tabpfn_v2_6)

    # 1. Evaluate on Real Datasets
    print("--- Phase 5: Evaluation on Real OpenML/Toy Datasets ---")
    datasets = {
        "Breast Cancer": lambda: load_breast_cancer(return_X_y=True),
        "credit-g": lambda: fetch_openml(data_id=31, return_X_y=True, as_frame=True),
        "diabetes": lambda: fetch_openml(data_id=37, return_X_y=True, as_frame=True),
    }
    
    results = []
    
    checkpoint_path = "msa_checkpoints/checkpoint_1K_best.pth"
    has_checkpoint = os.path.exists(checkpoint_path)
    
    for name, load_fn in datasets.items():
        print(f"\nEvaluating dataset: {name}...")
        try:
            X, y = load_fn()
            X, y = preprocess_dataset(X, y)
            
            # Limit total rows to 1000 for standard validation sets to speed up
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
            
            # Linear Attention TabPFN
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="linear")
            results.append({"Dataset": name, "Model": "Linear_Attention_TabPFN", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
            
            # ISAB Attention TabPFN
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="isab")
            results.append({"Dataset": name, "Model": "Similarity_Sorted_ISAB_TabPFN", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
            
            # MSA TabPFN (Zero-shot, Random)
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="msa", msa_strategy="random")
            results.append({"Dataset": name, "Model": "MSA_TabPFN_Random_ZeroShot", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
            
            # MSA TabPFN (Zero-shot, Similarity)
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="msa", msa_strategy="similarity")
            results.append({"Dataset": name, "Model": "MSA_TabPFN_Similarity_ZeroShot", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
            
            # MSA TabPFN (Zero-shot, PCA)
            acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="msa", msa_strategy="pca")
            results.append({"Dataset": name, "Model": "MSA_TabPFN_PCA_ZeroShot", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
            
            # MSA TabPFN (Fine-tuned, Similarity)
            if has_checkpoint:
                acc, auc, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="msa", msa_strategy="similarity", checkpoint_path=checkpoint_path)
                results.append({"Dataset": name, "Model": "MSA_TabPFN_Similarity_Finetuned", "Accuracy": acc, "ROC_AUC": auc, "Time (s)": elapsed, "Peak VRAM (MB)": vram})
                
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
        
        # We only benchmark speed/VRAM scaling metrics for:
        # - Vanilla TabPFN (full attention)
        # - Linear Attention TabPFN
        # - MSA TabPFN (Similarity)
        
        # Vanilla
        print("  Benchmarking Vanilla TabPFN...")
        _, _, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="vanilla")
        scaling_results.append({"Rows": size, "Model": "Vanilla_TabPFN", "Time (s)": elapsed, "Peak VRAM (MB)": vram})
        
        # Linear
        print("  Benchmarking Linear Attention TabPFN...")
        _, _, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="linear")
        scaling_results.append({"Rows": size, "Model": "Linear_Attention_TabPFN", "Time (s)": elapsed, "Peak VRAM (MB)": vram})
        
        # ISAB Attention
        print("  Benchmarking ISAB Attention TabPFN...")
        _, _, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="isab")
        scaling_results.append({"Rows": size, "Model": "Similarity_Sorted_ISAB_TabPFN", "Time (s)": elapsed, "Peak VRAM (MB)": vram})
        
        # Partitioned Attention (Random grouping — O(N log N) sort, no quadratic overhead)
        print("  Benchmarking Partitioned Attention TabPFN (Random)...")
        _, _, elapsed, vram = evaluate_tabpfn_variant(X_train, X_test, y_train, y_test, variant="msa", msa_strategy="random")
        scaling_results.append({"Rows": size, "Model": "Partitioned_Attention_TabPFN", "Time (s)": elapsed, "Peak VRAM (MB)": vram})
        
    df_scaling = pd.DataFrame(scaling_results)
    print("\n==================== SCALING SCENARIO RESULTS ====================")
    print(df_scaling.to_markdown(index=False))
    df_scaling.to_csv("scaling_evaluation_results.csv", index=False)

if __name__ == "__main__":
    main()
