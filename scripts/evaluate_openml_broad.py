import os
import sys
import time
import torch
import warnings
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def get_peak_memory():
    if torch.cuda.is_available():
        # returns peak VRAM in MB
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    # returns RSS RAM in MB
    return psutil.Process().memory_info().rss / (1024 ** 2)

# --- PYTORCH COMPATIBILITY PATCH FOR TABPFN 0.1.11 ---
import typing
import torch.nn.modules.transformer
torch.nn.modules.transformer.Optional = typing.Optional
# ---------------------------------------------------

# --- SCIKIT-LEARN COMPATIBILITY PATCH FOR TABPFN 0.1.11 ---
import sklearn.utils.validation
import sklearn.utils
_original_check_X_y = sklearn.utils.validation.check_X_y
_original_check_array = sklearn.utils.validation.check_array

def _patched_check_X_y(*args, **kwargs):
    kwargs.pop('force_all_finite', None)
    return _original_check_X_y(*args, **kwargs)

def _patched_check_array(*args, **kwargs):
    kwargs.pop('force_all_finite', None)
    return _original_check_array(*args, **kwargs)

sklearn.utils.validation.check_X_y = _patched_check_X_y
sklearn.utils.validation.check_array = _patched_check_array
sklearn.utils.check_X_y = _patched_check_X_y
sklearn.utils.check_array = _patched_check_array
# ----------------------------------------------------------

from tabpfn import TabPFNClassifier

warnings.filterwarnings('ignore')

# 18 Datasets (Original 8 + 10 New standard OpenML datasets)
DATASETS = [
    # Original 8
    ('breast-cancer', 13),
    ('credit-g', 31),
    ('diabetes', 37),
    ('vehicle', 54),
    ('kc2', 1063),
    ('pc1', 1068),
    ('haberman', 43),
    ('blood-transfusion', 1464),
    # 10 New diverse ones
    ('phoneme', 1489),
    ('spambase', 44),
    ('qsar-biodeg', 1494),
    ('churn', 40701),
    ('electricity', 151),
    ('phishing_websites', 4534),
    ('eeg-eye-state', 1471),
    ('wall-robot-navigation', 1497),
    ('bank-marketing', 1461),
    ('nomao', 1486)
]

def load_and_prep(data_id):
    data = fetch_openml(data_id=data_id, as_frame=True)
    X = data.data
    y = data.target
    for col in X.columns:
        if X[col].dtype == 'category' or X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
    X = X.fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(set(y)) > 2:
        y = (y > 0).astype(int)
    
    # Subsample very large datasets for broad evaluation suite to avoid long waits
    if len(X) > 5000:
        X, _, y, _ = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
        
    return train_test_split(X, y, test_size=0.33, random_state=42)

def evaluate_broad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Broad OpenML Evaluation on {device}")
    
    results = []
    
    for name, data_id in DATASETS:
        print(f"\n--- Evaluating Dataset: {name} ---")
        try:
            X_train, X_test, y_train, y_test = load_and_prep(data_id)
            print(f"Dataset size: Train N={len(X_train)}, Features={X_train.shape[1]}")
            
            # 1. Vanilla TabPFN
            clear_gpu()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            from zsisab.wrapper import restore_vanilla_tabpfn
            restore_vanilla_tabpfn()
            
            start_mem = get_peak_memory()
            start_time = time.time()
            
            clf_vanilla = TabPFNClassifier(device=device, N_ensemble_configurations=1)
            clf_vanilla.fit(X_train, y_train, overwrite_warning=True)
            probs_v = clf_vanilla.predict_proba(X_test)
            preds_v = clf_vanilla.predict(X_test)
            
            latency_v = time.time() - start_time
            peak_mem_v = max(0.0, get_peak_memory() - start_mem)
            acc_v = accuracy_score(y_test, preds_v)
            auc_v = roc_auc_score(y_test, probs_v[:, 1])
            results.append({
                'Dataset': name, 'Model': 'Vanilla TabPFN', 
                'Accuracy': acc_v, 'ROC AUC': auc_v, 
                'Latency (s)': latency_v, 'Peak Memory (MB)': peak_mem_v
            })
            
            # 2. Zero-Shot ISAB
            clear_gpu()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            from zsisab.wrapper import inject_zsisab_into_tabpfn
            inject_zsisab_into_tabpfn(num_prototypes=128)
            
            start_mem = get_peak_memory()
            start_time = time.time()
            
            clf_isab = TabPFNClassifier(device=device, N_ensemble_configurations=1)
            clf_isab.fit(X_train, y_train, overwrite_warning=True)
            probs_i = clf_isab.predict_proba(X_test)
            preds_i = clf_isab.predict(X_test)
            
            latency_i = time.time() - start_time
            peak_mem_i = max(0.0, get_peak_memory() - start_mem)
            acc_i = accuracy_score(y_test, preds_i)
            auc_i = roc_auc_score(y_test, probs_i[:, 1])
            results.append({
                'Dataset': name, 'Model': 'Zero-Shot ISAB', 
                'Accuracy': acc_i, 'ROC AUC': auc_i, 
                'Latency (s)': latency_i, 'Peak Memory (MB)': peak_mem_i
            })
            
            print(f"Vanilla: Acc={acc_v:.4f}, AUC={auc_v:.4f}, Time={latency_v:.2f}s, Mem={peak_mem_v:.1f}MB")
            print(f"ZS-ISAB: Acc={acc_i:.4f}, AUC={auc_i:.4f}, Time={latency_i:.2f}s, Mem={peak_mem_i:.1f}MB")
            restore_vanilla_tabpfn()
            
        except Exception as e:
            print(f"Failed on {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv("broad_evaluation_results.csv", index=False)
        print("\nResults saved to broad_evaluation_results.csv")
        
        os.makedirs("assets", exist_ok=True)
        device_label = "RTX 3090 Ti" if torch.cuda.is_available() else "Server CPU"
        
        # Accuracy plot
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model')
        plt.title(f'Zero-Shot Accuracy Comparison Across Broad OpenML Suite ({device_label})')
        plt.ylabel('Zero-Shot Accuracy')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('assets/broad_evaluation_accuracy.png', dpi=300)
        plt.close()
        
        # Memory/Time plot
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df, x='Dataset', y='Latency (s)', hue='Model')
        plt.title(f'Inference Latency Comparison Across Broad OpenML Suite ({device_label})')
        plt.ylabel('Latency (seconds)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('assets/broad_evaluation_latency.png', dpi=300)
        plt.close()
        
        print("Generated plots in assets/")

if __name__ == "__main__":
    evaluate_broad()
