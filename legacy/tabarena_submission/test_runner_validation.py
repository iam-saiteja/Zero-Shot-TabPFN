"""
Rigorous validation test for ZS-ISAB injection, fitting, and probability prediction.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add current workspace to path
sys.path.insert(0, r"c:\Users\iamsa\Documents\ISAB-r")

import typing
import torch
import torch.nn.modules.transformer
torch.nn.modules.transformer.Optional = typing.Optional

from zsisab.wrapper import inject_zsisab
from tabpfn import TabPFNClassifier


def run_full_validation():
    print("=" * 60)
    print("RUNNING RIGOROUS END-TO-END VALIDATION FOR ZS-ISAB")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    # 1. Test Injection
    print("\n[1/3] Testing inject_zsisab...")
    inject_zsisab(num_prototypes=64, chunk_size=512)
    print(" -> Injected successfully!")

    # 2. Test Binary Classification
    print("\n[2/3] Testing Binary Classification (X: 100x4, Y: binary)...")
    np.random.seed(42)
    X_train = np.random.randn(100, 4).astype(np.float32)
    y_train = np.random.choice([0, 1], size=100)
    X_test = np.random.randn(20, 4).astype(np.float32)

    model = TabPFNClassifier(device=device, N_ensemble_configurations=2)
    model.fit(X_train, y_train, overwrite_warning=True)
    
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    print(f" -> Probs shape: {probs.shape} (Expected (20, 2))")
    print(f" -> Preds shape: {preds.shape} (Expected (20,))")
    assert probs.shape == (20, 2), f"Bad shape {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4), "Probabilities must sum to 1"
    print(" -> Binary classification test PASSED perfectly!")

    # 3. Test Multiclass Classification
    print("\n[3/3] Testing Multiclass Classification (X: 120x6, Y: 3 classes)...")
    X_train_mc = np.random.randn(120, 6).astype(np.float32)
    y_train_mc = np.random.choice([0, 1, 2], size=120)
    X_test_mc = np.random.randn(30, 6).astype(np.float32)

    model_mc = TabPFNClassifier(device=device, N_ensemble_configurations=2)
    model_mc.fit(X_train_mc, y_train_mc, overwrite_warning=True)

    probs_mc = model_mc.predict_proba(X_test_mc)
    preds_mc = model_mc.predict(X_test_mc)

    print(f" -> Multiclass Probs shape: {probs_mc.shape} (Expected (30, 3))")
    assert probs_mc.shape == (30, 3), f"Bad shape {probs_mc.shape}"
    assert np.allclose(probs_mc.sum(axis=1), 1.0, atol=1e-4), "Probabilities must sum to 1"
    print(" -> Multiclass classification test PASSED perfectly!")

    print("\n" + "=" * 60)
    print("ALL VALIDATION CHECKS PASSED WITH 100% SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_validation()
