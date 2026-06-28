import os
import sys
import time
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

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
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Predict
    start_test = time.time()
    try:
        if is_classification:
            y_pred = model.predict_proba(X_test)
            if len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            metric_val = roc_auc_score(y_test, y_pred, multi_class='ovr')
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

    # Iterate over datasets recursively to handle size partitions
    for category in ["small", "medium", "large"]:
        cat_dir = dataset_dir / category
        if not cat_dir.exists():
            print(f"Skipping {category} partition: directory {cat_dir} does not exist. (Did you run organize_datasets.py?)")
            continue
            
        for d_path in cat_dir.iterdir():
            if not d_path.is_dir():
                continue
                
            dataset_name = f"{category}/{d_path.name}"
        print(f"\n==============================================")
        print(f"Evaluating TFM Dataset: {dataset_name}")
        
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
        
        if TABPFN_AVAILABLE:
            if is_classification:
                models_to_test.append((TabPFNClassifier(), "TabPFN-v3-Local"))
            else:
                try:
                    from tabpfn_client import TabPFNRegressor
                    models_to_test.append((TabPFNRegressor(), "TabPFN-v3-Local-Reg"))
                except ImportError:
                    try:
                        from tabpfn import TabPFNRegressor
                        models_to_test.append((TabPFNRegressor(), "TabPFN-v3-Local-Reg"))
                    except ImportError:
                        print("     [-] TabPFNRegressor not available. Skipping TabPFN.")
                
        if TABICL_AVAILABLE:
            if is_classification:
                models_to_test.append((TabICLClassifier(), "TabICL"))
            else:
                print("     [-] TabICL only supports classification. Skipping.")
            
        for model, name in models_to_test:
            res = evaluate_model(model, X_train, y_train, X_test, y_test, name, is_classification)
            if res:
                res["dataset"] = dataset_name
                results.append(res)
                print(f"     [+] {name} | {res['metric_name']}: {res['metric_val']:.4f} | Train: {res['train_time']:.4f}s | Test: {res['test_time']:.4f}s")
                
                # Save incrementally to prevent data loss on crash
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=4)
                
    print(f"\nAll TFM Benchmarks Complete!")
    print(f"Saved TFM leaderboard to {out_file}")

if __name__ == "__main__":
    main()
