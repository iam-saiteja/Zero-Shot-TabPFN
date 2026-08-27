"""
Unit test for ZSISABModel to verify compatibility with AutoGluon / TabArena.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ZSISABModel


def test_zsisab_binary():
    print("Testing ZSISABModel on Binary Classification toy dataset...")
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(np.random.choice([0, 1], size=50), name="target")

    model = ZSISABModel(problem_type="binary")
    model.fit(X=X, y=y)
    
    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    assert len(preds) == 50, f"Expected 50 predictions, got {len(preds)}"
    assert probs.shape == (50, 2), f"Expected shape (50, 2), got {probs.shape}"
    print(" [OK] Binary Classification passed successfully!")


def test_zsisab_multiclass():
    print("Testing ZSISABModel on Multiclass Classification toy dataset...")
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(np.random.choice([0, 1, 2], size=60), name="target")

    model = ZSISABModel(problem_type="multiclass")
    model.fit(X=X, y=y)
    
    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    assert len(preds) == 60, f"Expected 60 predictions, got {len(preds)}"
    assert probs.shape == (60, 3), f"Expected shape (60, 3), got {probs.shape}"
    print(" [OK] Multiclass Classification passed successfully!")


if __name__ == "__main__":
    test_zsisab_binary()
    test_zsisab_multiclass()
    print("\n All TabArena ZS-ISAB Unit Tests Passed 100%!")
