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
from sklearn.metrics import accuracy_score, roc_auc_score

def run_isab_benchmark(size):
    clear_gpu()
    class TempISAB(AlongColumnAttentionTwoPass):
        def __init__(self, *args, **kwargs):
            kwargs["num_prototypes"] = 128
            super().__init__(*args, **kwargs)
    tabpfn_v2.AlongColumnAttention = TempISAB
    tabpfn_v2_5.AlongColumnAttention = TempISAB
    tabpfn_v2_6.AlongColumnAttention = TempISAB
    original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
    tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)
        
    try:
        X, y = generate_scm_dataset(num_samples=size+100, num_features=10, task_type="classification", num_classes=2, random_state=42)
        X, y = preprocess_dataset(X, y)
        X_train, X_test = X[:size], X[size:]
        y_train, y_test = y[:size], y[size:]
        
        from tabpfn.constants import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, device="cuda", ignore_pretraining_limits=True)
        start_time = time.time()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        preds = clf.predict(X_test)
        elapsed = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs[:, 1])
        return elapsed, peak_vram, acc, auc, "Success"
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0, str(e)
    finally:
        import importlib
        importlib.reload(tabpfn_v2)
        importlib.reload(tabpfn_v2_5)
        importlib.reload(tabpfn_v2_6)

sizes = [65536, 131072, 262144, 524288, 1048576]
print("=========================================")
print("MILLION-ROW SCALING BENCHMARK (ISAB-R)")
print("=========================================")
for size in sizes:
    t, vram, acc, auc, status = run_isab_benchmark(size)
    if status == "Success":
        print(f"N={size}: Acc={acc:.4f} AUC={auc:.4f} | Time={t:.1f}s VRAM={vram:.1f}MB")
    else:
        print(f"N={size} failed: {status}")
        break
