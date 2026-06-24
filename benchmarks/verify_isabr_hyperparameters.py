import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import pandas as pd

sys.path.append("c:/Users/itachi/Documents/MSA")
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_WDvw1MHEYQRQz8NKJBMqEFoink8X-sagyYRMKWM8Vo4"

import tabpfn.architectures.tabpfn_v2 as tabpfn_v2
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
import tabpfn.architectures.tabpfn_v2_6 as tabpfn_v2_6
from tabpfn.architectures.kv_cache import KVCacheEntry
from tabpfn.architectures.shared.scaled_dot_product_attention import scaled_dot_product_attention

CURRENT_M = 32
USE_LOGIT_SCALING = False
USE_NORM_ALIGNMENT = False

class DebugTwoPass(tabpfn_v2_5.AlongColumnAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    @staticmethod
    def _chunk_means(train_rows: torch.Tensor, M: int) -> torch.Tensor:
        Bc, N, E = train_rows.shape
        device = train_rows.device
        perm = torch.randperm(N, device=device)
        chunk_size = max(1, N // M)
        num_elements = M * chunk_size
        selected_perm = perm[:num_elements]
        gathered = train_rows[:, selected_perm]
        gathered = gathered.view(Bc, M, chunk_size, E)
        return gathered.mean(dim=2)

    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        Bc, R, E = x_BcRE.shape
        H, D = self.num_heads, self.head_dim
        M = CURRENT_M

        q_BcRHD = self.q_projection(x_BcRE).view(Bc, R, H, D)

        if cached_kv is not None:
            k_Bc1 = cached_kv.key
            v_Bc1 = cached_kv.value
            assert k_Bc1 is not None
            assert v_Bc1 is not None
            if k_Bc1.dtype != q_BcRHD.dtype:
                k_Bc1 = k_Bc1.to(q_BcRHD.dtype)
                v_Bc1 = v_Bc1.to(q_BcRHD.dtype)
            output_BcSHD = scaled_dot_product_attention(q_BcRHD, k_Bc1, v_Bc1)
            return self.out_projection(output_BcSHD.reshape(Bc, R, H * D)), None

        N = R if single_eval_pos is None else single_eval_pos
        train_rows = x_BcRE[:, :N]

        if N <= M:
            k_refined = self.k_projection(train_rows).view(Bc, N, H, D)
            v_refined = self.v_projection(train_rows).view(Bc, N, H, D)
        else:
            proto_init = self._chunk_means(train_rows, M)
            
            if USE_NORM_ALIGNMENT:
                train_mean = train_rows.mean(dim=1, keepdim=True)
                train_std = train_rows.std(dim=1, keepdim=True).clamp(min=1e-6)
                proto_mean = proto_init.mean(dim=1, keepdim=True)
                proto_std = proto_init.std(dim=1, keepdim=True).clamp(min=1e-6)
                proto_init = (proto_init - proto_mean) / proto_std * train_std + train_mean
                
            q_p = self.q_projection(proto_init).view(Bc, M, H, D).transpose(1, 2)
            k_r = self.k_projection(train_rows).view(Bc, N, H, D).transpose(1, 2)
            v_r = self.v_projection(train_rows).view(Bc, N, H, D).transpose(1, 2)

            attn_weights = F.softmax(torch.matmul(q_p, k_r.transpose(-2, -1)) / math.sqrt(D), dim=-1)
            k_refined = torch.matmul(attn_weights, k_r).transpose(1, 2).contiguous()
            v_refined = torch.matmul(attn_weights, v_r).transpose(1, 2).contiguous()

        if single_eval_pos == R:
            if USE_LOGIT_SCALING and N > M:
                scale_factor = math.sqrt(math.log(N) / math.log(M))
                q_BcRHD = q_BcRHD * scale_factor
            output_BcSHD = scaled_dot_product_attention(q_BcRHD, k_refined, v_refined)
        else:
            if USE_LOGIT_SCALING and N > M:
                scale_factor = math.sqrt(math.log(N) / math.log(M))
                q_train = q_BcRHD[:, :N] * scale_factor
                q_test = q_BcRHD[:, N:] * scale_factor
            else:
                q_train = q_BcRHD[:, :N]
                q_test = q_BcRHD[:, N:]
                
            out_train_BcNHD = scaled_dot_product_attention(
                q_train, k_refined, v_refined
            )
            out_test_BcMHD = scaled_dot_product_attention(
                q_test, k_refined[:, :, :1], v_refined[:, :, :1]
            )
            output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)

        kv_entry = None
        if return_kv:
            kv_entry = KVCacheEntry(
                key=k_refined[:, :, :1].contiguous().detach(),
                value=v_refined[:, :, :1].contiguous().detach(),
            )

        output_BcSF = output_BcSHD.reshape(Bc, R, H * D)
        return self.out_projection(output_BcSF), kv_entry

tabpfn_v2.AlongColumnAttention = DebugTwoPass
tabpfn_v2_5.AlongColumnAttention = DebugTwoPass
tabpfn_v2_6.AlongColumnAttention = DebugTwoPass

original_load_v2 = tabpfn_v2.TabPFNV2.load_state_dict
tabpfn_v2.TabPFNV2.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2(self, sd, strict=False, assign=assign)
original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)
original_load_v2_6 = tabpfn_v2_6.TabPFNV2p6.load_state_dict
tabpfn_v2_6.TabPFNV2p6.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_6(self, sd, strict=False, assign=assign)

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X_bc, y_bc = load_breast_cancer(return_X_y=True)
X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(
    X_bc, y_bc, test_size=0.33, random_state=42
)

configs = [
    {"M": 32, "scale": False, "norm": False},
    {"M": 32, "scale": True, "norm": False},
    {"M": 32, "scale": True, "norm": True},
    {"M": 64, "scale": False, "norm": False},
    {"M": 64, "scale": True, "norm": False},
    {"M": 64, "scale": True, "norm": True},
    {"M": 128, "scale": False, "norm": False},
]

results = []
for cfg in configs:
    CURRENT_M = cfg["M"]
    USE_LOGIT_SCALING = cfg["scale"]
    USE_NORM_ALIGNMENT = cfg["norm"]
    
    print(f"Running config: M={CURRENT_M}, LogitScaling={USE_LOGIT_SCALING}, NormAlignment={USE_NORM_ALIGNMENT}")
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, device="auto")
    clf.fit(X_bc_train, y_bc_train)
    probs = clf.predict_proba(X_bc_test)
    preds = clf.predict(X_bc_test)
    acc = accuracy_score(y_bc_test, preds)
    auc = roc_auc_score(y_bc_test, probs[:, 1])
    results.append({"M": CURRENT_M, "LogitScaling": USE_LOGIT_SCALING, "NormAlignment": USE_NORM_ALIGNMENT, "Accuracy": acc, "ROC_AUC": auc})

CURRENT_M = 500
print("Running Vanilla...")
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, device="auto")
clf.fit(X_bc_train, y_bc_train)
probs = clf.predict_proba(X_bc_test)
preds = clf.predict(X_bc_test)
acc = accuracy_score(y_bc_test, preds)
auc = roc_auc_score(y_bc_test, probs[:, 1])
results.append({"M": "Vanilla", "LogitScaling": False, "NormAlignment": False, "Accuracy": acc, "ROC_AUC": auc})

print("\nRESULTS TABLE:")
print(pd.DataFrame(results).to_markdown(index=False))
