from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

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
from nsatabpfn.wrapper import inject_nsatabpfn, restore_vanilla_tabpfn

def main() -> None:
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using Device:", device)

    print("\nLoading Breast Cancer Dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    print("\nApplying NSA-TabPFN Injection (M=64)...")
    inject_nsatabpfn(num_prototypes=64, verbose=True)

    print("Initializing TabPFNClassifier on device:", device)
    clf = TabPFNClassifier(device=device, N_ensemble_configurations=1)

    print("Fitting model on train size:", len(X_train))
    clf.fit(X_train, y_train, overwrite_warning=True)

    print("Predicting probabilities under NSA-TabPFN...")
    prediction_probabilities = clf.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, prediction_probabilities[:, 1])
    print(f"ROC AUC: {roc_auc:.4f}")

    print("Predicting labels under NSA-TabPFN...")
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nRestoring Vanilla TabPFN...")
    restore_vanilla_tabpfn(verbose=True)

    print("\nVerification check:")
    if accuracy > 0.85:
        print("SUCCESS: NSA-TabPFN runs cleanly and accuracy is high!")
    else:
        print("WARNING: Accuracy is lower than expected.")

if __name__ == "__main__":
    main()
