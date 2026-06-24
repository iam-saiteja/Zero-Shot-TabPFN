from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

# Set token
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_WDvw1MHEYQRQz8NKJBMqEFoink8X-sagyYRMKWM8Vo4"

# 1. Monkey patch AlongColumnAttention in tabpfn architectures before import
import tabpfn.architectures.tabpfn_v2 as tabpfn_v2
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
import tabpfn.architectures.tabpfn_v2_6 as tabpfn_v2_6
from tabpfn_msa import AlongColumnAttentionMSA

# Override AlongColumnAttention in all active versions
tabpfn_v2.AlongColumnAttention = AlongColumnAttentionMSA
tabpfn_v2_5.AlongColumnAttention = AlongColumnAttentionMSA
tabpfn_v2_6.AlongColumnAttention = AlongColumnAttentionMSA

def load_state_dict_non_strict(self, state_dict, strict=True):
    import torch
    return torch.nn.Module.load_state_dict(self, state_dict, strict=False)

tabpfn_v2.TabPFNV2.load_state_dict = load_state_dict_non_strict
tabpfn_v2_5.TabPFNV2p5.load_state_dict = load_state_dict_non_strict
tabpfn_v2_6.TabPFNV2p6.load_state_dict = load_state_dict_non_strict

# 2. Now import TabPFNClassifier
from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_verify(blocking_strategy: str = "random"):
    print(f"\n--- Verifying TabPFN with MSA (blocking strategy: {blocking_strategy}) ---")
    
    # Dynamically patch __init__ of AlongColumnAttentionMSA to use the specified strategy
    original_init = AlongColumnAttentionMSA.__init__
    
    def patched_init(self, *args, **kwargs):
        # Force MSA settings
        kwargs["blocking_strategy"] = blocking_strategy
        kwargs["topk"] = 4
        kwargs["blk_kv"] = 32
        original_init(self, *args, **kwargs)
        
    AlongColumnAttentionMSA.__init__ = patched_init
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    
    try:
        clf = TabPFNClassifier(device="auto")
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"SUCCESS: Accuracy is {accuracy:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original init
        AlongColumnAttentionMSA.__init__ = original_init

if __name__ == "__main__":
    for strategy in ["random", "similarity", "pca"]:
        run_verify(strategy)
