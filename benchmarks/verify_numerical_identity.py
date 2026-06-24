import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

sys.path.append("c:/Users/itachi/Documents/MSA")
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
from tabpfn.architectures.kv_cache import KVCacheEntry

device = "cpu"
original_layer = tabpfn_v2_5.AlongColumnAttention(embedding_size=128, num_heads=8, head_dim=16, device=device)

torch.manual_seed(42)
Bc, R, E = 2, 10, 128
single_eval_pos = 6
x = torch.randn(Bc, R, E, device=device)

with torch.no_grad():
    out_orig, kv_orig = original_layer(x, single_eval_pos=single_eval_pos, return_kv=True)

def run_debug_twopass(layer, x_BcRE, single_eval_pos, M=32, cached_kv=None, return_kv=False):
    Bc, R, E = x_BcRE.shape
    H, D = layer.num_heads, layer.head_dim
    q_BcRHD = layer.q_projection(x_BcRE).view(Bc, R, H, D)
    
    if cached_kv is not None:
        k_Bc1 = cached_kv.key
        v_Bc1 = cached_kv.value
        from tabpfn.architectures.shared.scaled_dot_product_attention import scaled_dot_product_attention
        output_BcSHD = scaled_dot_product_attention(q_BcRHD, k_Bc1, v_Bc1)
        return layer.out_projection(output_BcSHD.reshape(Bc, R, H * D)), None

    N = R if single_eval_pos is None else single_eval_pos
    train_rows = x_BcRE[:, :N]
    
    if N <= M:
        k_refined = layer.k_projection(train_rows).view(Bc, N, H, D)
        v_refined = layer.v_projection(train_rows).view(Bc, N, H, D)
    else:
        perm = torch.arange(N)
        chunk_size = max(1, N // M)
        gathered = train_rows[:, perm[:M*chunk_size]].view(Bc, M, chunk_size, E)
        proto_init = gathered.mean(dim=2)
        
        q_p = layer.q_projection(proto_init).view(Bc, M, H, D).transpose(1, 2)
        k_r = layer.k_projection(train_rows).view(Bc, N, H, D).transpose(1, 2)
        v_r = layer.v_projection(train_rows).view(Bc, N, H, D).transpose(1, 2)
        attn_weights = F.softmax(torch.matmul(q_p, k_r.transpose(-2, -1)) / math.sqrt(D), dim=-1)
        k_refined = torch.matmul(attn_weights, k_r).transpose(1, 2).contiguous()
        v_refined = torch.matmul(attn_weights, v_r).transpose(1, 2).contiguous()

    from tabpfn.architectures.shared.scaled_dot_product_attention import scaled_dot_product_attention
    if single_eval_pos == R:
        output_BcSHD = scaled_dot_product_attention(q_BcRHD, k_refined, v_refined)
    else:
        out_train_BcNHD = scaled_dot_product_attention(q_BcRHD[:, :N], k_refined, v_refined)
        out_test_BcMHD = scaled_dot_product_attention(q_BcRHD[:, N:], k_refined[:, :, :1], v_refined[:, :, :1])
        output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)

    kv_entry = None
    if return_kv:
        kv_entry = KVCacheEntry(
            key=k_refined[:, :, :1].contiguous().detach(),
            value=v_refined[:, :, :1].contiguous().detach(),
        )
    return layer.out_projection(output_BcSHD.reshape(Bc, R, H * D)), kv_entry

with torch.no_grad():
    out_dbg, kv_dbg = run_debug_twopass(original_layer, x, single_eval_pos=single_eval_pos, M=32, return_kv=True)

print("--- Fallback Numerical Match Check ---")
print("Max diff in output:", torch.max(torch.abs(out_orig - out_dbg)).item())
print("Max diff in cached key:", torch.max(torch.abs(kv_orig.key - kv_dbg.key)).item())
print("Max diff in cached value:", torch.max(torch.abs(kv_orig.value - kv_dbg.value)).item())

with torch.no_grad():
    out_orig_cached, _ = original_layer(x[:, single_eval_pos:], single_eval_pos=0, cached_kv=kv_orig)
    out_dbg_cached, _ = run_debug_twopass(original_layer, x[:, single_eval_pos:], single_eval_pos=0, M=32, cached_kv=kv_dbg)

print("\n--- Cache Path Numerical Match Check ---")
print("Max diff in cached output:", torch.max(torch.abs(out_orig_cached - out_dbg_cached)).item())
