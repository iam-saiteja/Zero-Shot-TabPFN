import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

# 1. DYNAMIC SYSTEM PATH INCLUSION
# Dynamically locate the project root relative to the directory containing this script.
# This prevents crashes when the repository is moved or executed in different environments.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
from tabpfn.architectures.kv_cache import KVCacheEntry

# 2. DEVICE SELECTION
# Dynamically choose CPU or CUDA based on availability.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running numerical identity verification on device: {device}")

# Initialize original TabPFN along-column attention layer
original_layer = tabpfn_v2_5.AlongColumnAttention(embedding_size=128, num_heads=8, head_dim=16, device=device)

# Seed for reproducible random values
torch.manual_seed(42)
Bc, R, E = 2, 10, 128
single_eval_pos = 6
x = torch.randn(Bc, R, E, device=device)

# Compute original full self-attention output and key-value cache
with torch.no_grad():
    out_orig, kv_orig = original_layer(x, single_eval_pos=single_eval_pos, return_kv=True)

# 3. REPLICATED DEBUG ISAB-R (TwoPass) IMPLEMENTATION
# This function replicates AlongColumnAttentionTwoPass to verify numerical identity.
def run_debug_twopass(layer, x_BcRE, single_eval_pos, M=32, cached_kv=None, return_kv=False):
    Bc, R, E = x_BcRE.shape
    H, D = layer.num_heads, layer.head_dim
    q_BcRHD = layer.q_projection(x_BcRE).view(Bc, R, H, D)
    
    # Cache lookup path: if keys/values are cached, perform direct attention
    if cached_kv is not None:
        k_Bc1 = cached_kv.key
        v_Bc1 = cached_kv.value
        from tabpfn.architectures.shared.scaled_dot_product_attention import scaled_dot_product_attention
        output_BcSHD = scaled_dot_product_attention(q_BcRHD, k_Bc1, v_Bc1)
        return layer.out_projection(output_BcSHD.reshape(Bc, R, H * D)), None

    N = R if single_eval_pos is None else single_eval_pos
    train_rows = x_BcRE[:, :N]
    
    # 4. FALLBACK PATH (N <= M)
    # If the number of training rows is less than or equal to M, ISAB-R must fall back
    # to standard full self-attention, generating mathematically identical keys/values.
    if N <= M:
        k_refined = layer.k_projection(train_rows).view(Bc, N, H, D)
        v_refined = layer.v_projection(train_rows).view(Bc, N, H, D)
    else:
        # Vectorized chunk means for prototype initialization
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
        # Train queries attend to all heads of key/value
        out_train_BcNHD = scaled_dot_product_attention(q_BcRHD[:, :N], k_refined, v_refined)
        
        # 5. MQA PATH ALIGNMENT FOR TEST QUERIES
        # Test queries only attend to the first head's KV cache (slice [:, :, :1]) of training rows.
        # This matches the Multi-Query Attention configuration built into TabPFN.
        out_test_BcMHD = scaled_dot_product_attention(q_BcRHD[:, N:], k_refined[:, :, :1], v_refined[:, :, :1])
        output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)

    # 6. CACHED KV CONSTRUCTION
    # For prediction/KV caching, the cache contains only the first head of keys and values.
    kv_entry = None
    if return_kv:
        kv_entry = KVCacheEntry(
            key=k_refined[:, :, :1].contiguous().detach(),
            value=v_refined[:, :, :1].contiguous().detach(),
        )
    return layer.out_projection(output_BcSHD.reshape(Bc, R, H * D)), kv_entry

# Run debug implementation of TwoPass
with torch.no_grad():
    out_dbg, kv_dbg = run_debug_twopass(original_layer, x, single_eval_pos=single_eval_pos, M=32, return_kv=True)

# 7. VERIFICATION CHECKS
# Since N (6) <= M (32), the fallback path is active. We verify that outputs and caches
# match the original vanilla implementation exactly (difference of 0.0 or within float precision).
print("--- Fallback Numerical Match Check ---")
print("Max diff in output:", torch.max(torch.abs(out_orig - out_dbg)).item())
print("Max diff in cached key:", torch.max(torch.abs(kv_orig.key - kv_dbg.key)).item())
print("Max diff in cached value:", torch.max(torch.abs(kv_orig.value - kv_dbg.value)).item())

# Verify that downstream inference using the generated KV Cache is also identical
with torch.no_grad():
    out_orig_cached, _ = original_layer(x[:, single_eval_pos:], single_eval_pos=0, cached_kv=kv_orig)
    out_dbg_cached, _ = run_debug_twopass(original_layer, x[:, single_eval_pos:], single_eval_pos=0, M=32, cached_kv=kv_dbg)

print("\n--- Cache Path Numerical Match Check ---")
print("Max diff in cached output:", torch.max(torch.abs(out_orig_cached - out_dbg_cached)).item())

