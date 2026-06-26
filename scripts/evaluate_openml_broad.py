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

# 30 standard diverse classification datasets (skipping nomao which has 118 features > 100 max)
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
    # 10 previously added
    ('phoneme', 1489),
    ('spambase', 44),
    ('qsar-biodeg', 1494),
    ('churn', 40701),
    ('electricity', 151),
    ('phishing_websites', 4534),
    ('eeg-eye-state', 1471),
    ('wall-robot-navigation', 1497),
    ('bank-marketing', 1461),
    # 13 additional diverse classification datasets to reach 30 target (excluding nomao)
    ('wine', 187),
    ('page-blocks', 30),
    ('segment', 36),
    ('steel-plates-fault', 1504),
    ('first-order-theorem-proving', 1475),
    ('gas-drift', 1476),
    ('wilt', 1441),
    ('balance-scale', 11),
    ('analcatdata_authorship', 457),
    ('dna', 40670),
    ('car', 40975),
    ('tic-tac-toe', 50),
    ('madelon', 1110)
]

def load_and_prep(data_id):
    data = fetch_openml(data_id=data_id, as_frame=True)
    X = data.data
    y = data.target
    
    # Restrict feature count to TabPFN max limits (100 features) dynamically
    if X.shape[1] > 100:
        # Select first 100 columns
        X = X.iloc[:, :100]
        
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
    print(f"Running Broad 30-Dataset Evaluation on {device}...")
    
    results = []
    
    for name, data_id in DATASETS:
        print(f"\n--- Evaluating Dataset: {name} ---")
        try:
            X_train, X_test, y_train, y_test = load_and_prep(data_id)
            print(f"Size: Train N={len(X_train)}, Features={X_train.shape[1]}")
            
            # 1. Vanilla TabPFN Baseline
            clear_gpu()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            from nsatabpfn.wrapper import restore_vanilla_tabpfn
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
            print(f"Vanilla: Acc={acc_v:.4f}, AUC={auc_v:.4f}, Time={latency_v:.2f}s, Mem={peak_mem_v:.1f}MB")
            
            # 2. NSA-TabPFN at M = [64, 128, 256]
            from nsatabpfn.wrapper import inject_nsatabpfn
            for M in [64, 128, 256]:
                clear_gpu()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                inject_nsatabpfn(num_prototypes=M)
                
                start_mem = get_peak_memory()
                start_time = time.time()
                
                clf_nsa = TabPFNClassifier(device=device, N_ensemble_configurations=1)
                clf_nsa.fit(X_train, y_train, overwrite_warning=True)
                probs_i = clf_nsa.predict_proba(X_test)
                preds_i = clf_nsa.predict(X_test)
                
                latency_i = time.time() - start_time
                peak_mem_i = max(0.0, get_peak_memory() - start_mem)
                acc_i = accuracy_score(y_test, preds_i)
                auc_i = roc_auc_score(y_test, probs_i[:, 1])
                results.append({
                    'Dataset': name, 'Model': f'NSA-TabPFN (M={M})', 
                    'Accuracy': acc_i, 'ROC AUC': auc_i, 
                    'Latency (s)': latency_i, 'Peak Memory (MB)': peak_mem_i
                })
                print(f"NSA-TabPFN (M={M}): Acc={acc_i:.4f}, AUC={auc_i:.4f}, Time={latency_i:.2f}s, Mem={peak_mem_i:.1f}MB")
                
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
        
        # Display side-by-side results in console as a clean pivoted table
        pivoted = df.pivot(index='Dataset', columns='Model', values=['Accuracy', 'ROC AUC', 'Latency (s)', 'Peak Memory (MB)'])
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n=== CONSOLIDATED 30-DATASET EVALUATION RESULTS ===")
        print(pivoted)
        
        os.makedirs("assets", exist_ok=True)
        device_label = "RTX 3050 Laptop" if torch.cuda.is_available() else "CPU"
        
        # Accuracy plot
        plt.figure(figsize=(16, 7))
        sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model')
        plt.title(f'NSA-TabPFN vs Vanilla Accuracy Comparison Across 30 Datasets ({device_label})')
        plt.ylabel('Zero-Shot Accuracy')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('assets/broad_evaluation_accuracy.png', dpi=300)
        plt.close()
        
        # Latency plot
        plt.figure(figsize=(16, 7))
        sns.barplot(data=df, x='Dataset', y='Latency (s)', hue='Model')
        plt.title(f'NSA-TabPFN vs Vanilla Latency Comparison Across 30 Datasets ({device_label})')
        plt.ylabel('Latency (seconds)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('assets/broad_evaluation_latency.png', dpi=300)
        plt.close()
        
        print("Generated plots in assets/")

if __name__ == "__main__":
    evaluate_broad()
