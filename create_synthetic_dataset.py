import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from tabzilla_datasets import TabularDataset

def main():
    print("Generating synthetic dataset...")
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Create 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    split_indeces = []
    
    for train_index, test_index in kf.split(X):
        split_indeces.append({
            "train": train_index,
            "val": test_index,
            "test": test_index
        })

    dataset = TabularDataset(
        name="synthetic__1",
        X=X,
        y=y,
        cat_idx=[],
        target_type="binary",
        num_classes=2,
        num_features=20,
        num_instances=1000,
        cat_dims=[],
        split_indeces=split_indeces,
        split_source="synthetic"
    )

    out_dir = Path("datasets/synthetic__1")
    dataset.write(out_dir, overwrite=True)
    print(f"Dataset successfully written to {out_dir}")

if __name__ == "__main__":
    main()