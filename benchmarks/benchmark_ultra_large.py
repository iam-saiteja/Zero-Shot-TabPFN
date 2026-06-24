import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys

sys.path.append("c:/Users/itachi/Documents/MSA")
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_WDvw1MHEYQRQz8NKJBMqEFoink8X-sagyYRMKWM8Vo4"

import tabpfn.architectures.tabpfn_v2 as tabpfn_v2
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
import tabpfn.architectures.tabpfn_v2_6 as tabpfn_v2_6
from tabpfn_msa import AlongColumnAttentionTwoPass
from tabpfn import TabPFNClassifier
from data_generator import generate_scm_dataset
from evaluate import preprocess_dataset, clear_gpu

def evaluate_variant(X_train, X_test, y_train, y_test, variant="vanilla"):
    clear_gpu()
    if variant == "isab":
        class TempISAB(AlongColumnAttentionTwoPass):
            def __init__(self, *args, **kwargs):
                kwargs["num_prototypes"] = 128
                super().__init__(*args, **kwargs)
        tabpfn_v2.AlongColumnAttention = TempISAB
        tabpfn_v2_5.AlongColumnAttention = TempISAB
        tabpfn_v2_6.AlongColumnAttention = TempISAB
        original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
        tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)
    else:
        import importlib
        importlib.reload(tabpfn_v2)
        importlib.reload(tabpfn_v2_5)
        importlib.reload(tabpfn_v2_6)
        
    try:
        from tabpfn.constants import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, device="cuda")
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        elapsed = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        return elapsed, peak_vram, "Success"
    except Exception as e:
        return 0.0, 0.0, str(e)
    finally:
        import importlib
        importlib.reload(tabpfn_v2)
        importlib.reload(tabpfn_v2_5)
        importlib.reload(tabpfn_v2_6)

sizes = [8192, 16384, 32768]
print("=========================================")
print("ULTRA-LARGE SCALING BENCHMARK (N=8K..32K)")
print("=========================================")
for size in sizes:
    X, y = generate_scm_dataset(num_samples=size+100, num_features=10, task_type="classification", num_classes=2, random_state=42)
    X, y = preprocess_dataset(X, y)
    X_train, X_test = X[:size], X[size:]
    y_train, y_test = y[:size], y[size:]
    
    t_v, vr_v, s_v = evaluate_variant(X_train, X_test, y_train, y_test, variant="vanilla")
    t_i, vr_i, s_i = evaluate_variant(X_train, X_test, y_train, y_test, variant="isab")
    print(f"N={size}: Vanilla={t_v:.2f}s ({vr_v:.1f}MB) | ISAB-R={t_i:.2f}s ({vr_i:.1f}MB)")
