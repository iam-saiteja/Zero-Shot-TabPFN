import os
import sys
import time
import json
import gc
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, r2_score

# Add TabZilla to path to use its dataset loader
tabzilla_path = Path(__file__).resolve().parent.parent / "tabzilla" / "TabZilla"
sys.path.append(str(tabzilla_path))
from tabzilla_datasets import TabularDataset

# Import Tabular Foundation Models
# We wrap them in try-except so the script doesn't crash if a specific TFM fails to load
try:
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        from tabpfn_client import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("WARNING: tabpfn/tabpfn_client is not installed in this environment.")

try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False
    print("WARNING: tabicl is not installed.")

# Add more TFMs here as they are available

from sklearn.metrics import roc_auc_score, r2_score

# ... (I will need to replace the entire evaluate_model function, let's just replace from line 36)
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, is_classification):
    print(f"  -> Evaluating {model_name}...")
    
    # Train
    start_train = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"     Error during training (fit): {e}")
        return None
    train_time = time.time() - start_train
    
    # Predict
    start_test = time.time()
    try:
        if is_classification:
            y_pred = model.predict_proba(X_test)
            unique_test_classes = np.unique(y_test)
            if len(unique_test_classes) < 2:
                # ROC-AUC is undefined for a single class in test split. Fallback to 1.0.
                metric_val = 1.0
            elif len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
                metric_val = roc_auc_score(y_test, y_pred)
            elif len(y_pred.shape) > 1 and y_pred.shape[1] > 2:
                # Multiclass: align predicted probability columns with classes present in y_test
                try:
                    train_classes = np.unique(y_train)
                    col_indices = [np.where(train_classes == c)[0][0] for c in unique_test_classes]
                    y_pred_sub = y_pred[:, col_indices]
                    # Avoid division by zero
                    row_sums = np.sum(y_pred_sub, axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1.0
                    y_pred_sub = y_pred_sub / row_sums
                    metric_val = roc_auc_score(y_test, y_pred_sub, multi_class='ovr', labels=unique_test_classes)
                except Exception:
                    # Fallback to standard multiclass with explicit labels parameter
                    metric_val = roc_auc_score(y_test, y_pred, multi_class='ovr', labels=unique_test_classes)
            else:
                metric_val = roc_auc_score(y_test, y_pred)
            metric_name = "AUC"
        else:
            y_pred = model.predict(X_test)
            metric_val = r2_score(y_test, y_pred)
            metric_name = "R2"
    except Exception as e:
        print(f"     Error during prediction/metrics: {e}")
        return None
    test_time = time.time() - start_test
    
    return {
        "model": model_name,
        "metric_name": metric_name,
        "metric_val": metric_val,
        "train_time": train_time,
        "test_time": test_time
    }

def main():
    dataset_dir = tabzilla_path / "datasets"
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    out_file = Path(__file__).resolve().parent / "tfm_leaderboard.json"
    results = []
    completed_runs = set()
    
    # Load existing results to resume from where we left off
    if out_file.exists():
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
            for res in results:
                completed_runs.add((res["dataset"], res["model"]))
            print(f"Resuming from existing leaderboard: Found {len(completed_runs)} completed model evaluations.")
        except Exception as e:
            print(f"Warning: Could not read existing leaderboard: {e}")

    # Gather all dataset paths first to count them
    dataset_paths = []
    for category in ["small", "medium", "large"]:
        cat_dir = dataset_dir / category
        if not cat_dir.exists():
            print(f"Skipping {category} partition: directory {cat_dir} does not exist.")
            continue
            
        for d_path in cat_dir.iterdir():
            if d_path.is_dir():
                dataset_paths.append((category, d_path))
                
    total_datasets = len(dataset_paths)
    print(f"\nFound {total_datasets} total datasets to evaluate.")
    
    # Iterate over collected datasets
    for idx, (category, d_path) in enumerate(dataset_paths, start=1):
        dataset_name = f"{category}/{d_path.name}"
        print(f"\n==============================================")
        print(f"Evaluating TFM Dataset [{idx}/{total_datasets}]: {dataset_name}")
            
        try:
            dataset = TabularDataset.read(d_path)
        except Exception as e:
            print(f"Error reading {dataset_name}: {e}")
            continue
            
        # We'll just use the first fold for this quick zero-shot arena
        fold = dataset.split_indeces[0]
        X_train, y_train = dataset.X[fold["train"]], dataset.y[fold["train"]]
        X_test, y_test = dataset.X[fold["test"]], dataset.y[fold["test"]]
        
        print(f"Rows: Train={len(X_train)} Test={len(X_test)} Features={X_train.shape[1]}")

        models_to_test = []
        is_classification = dataset.target_type in ["classification", "binary"]
        
        # 1. ZS-ISAB (Our Architecture)
        if TABPFN_AVAILABLE:
            if is_classification:
                models_to_test.append((TabPFNClassifier(), "ZS-ISAB"))
            else:
                try:
                    from tabpfn_client import TabPFNRegressor
                    models_to_test.append((TabPFNRegressor(), "ZS-ISAB-Reg"))
                except ImportError:
                    try:
                        from tabpfn import TabPFNRegressor
                        models_to_test.append((TabPFNRegressor(), "ZS-ISAB-Reg"))
                    except ImportError:
                        print("     [-] TabPFNRegressor not available. Skipping ZS-ISAB-Reg.")
                
        # 2. TabICL (Skip if dataset is too large to prevent OOM Killer)
        if TABICL_AVAILABLE:
            if is_classification:
                # Limit TabICL to prevent OOM on large dimensions
                num_elements = len(X_train) * X_train.shape[1]
                if num_elements > 1500000:
                    print(f"     [-] Skipping TabICL (Dataset too large: {len(X_train)}x{X_train.shape[1]} = {num_elements} elements. Would OOM.)")
                else:
                    models_to_test.append((TabICLClassifier(), "TabICL"))
            else:
                print("     [-] TabICL only supports classification. Skipping.")
                
        # 3. TabDPT (If installed in environment - Skip if too large to prevent OOM)
        try:
            from tabdpt import TabDPTClassifier
            if is_classification:
                num_elements = len(X_train) * X_train.shape[1]
                if num_elements > 1500000:
                    print(f"     [-] Skipping TabDPT (Dataset too large: {len(X_train)}x{X_train.shape[1]} = {num_elements} elements. Would OOM.)")
                else:
                    models_to_test.append((TabDPTClassifier(), "TabDPT"))
        except ImportError:
            pass
            
        for model, name in models_to_test:
            if (dataset_name, name) in completed_runs:
                print(f"     [-] {name} already evaluated for this dataset. Skipping.")
                continue
                
            res = evaluate_model(model, X_train, y_train, X_test, y_test, name, is_classification)
            if res:
                res["dataset"] = dataset_name
                results.append(res)
                print(f"     [+] {name} | {res['metric_name']}: {res['metric_val']:.4f} | Train: {res['train_time']:.4f}s | Test: {res['test_time']:.4f}s")
                
                # Save incrementally to prevent data loss on crash
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=4)
            
        # Memory Cleanup to prevent OOM crashes in long loops
        del models_to_test
        del X_train, y_train, X_test, y_test, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"\nAll TFM Benchmarks Complete!")
    print(f"Saved TFM leaderboard to {out_file}")

if __name__ == "__main__":
    main()
