"""Where does exact attention (tf_icl.forward, no patch) actually break on this
GPU? Uses the REAL pretrained checkpoint's config (embed_dim=128, 12 ICL blocks,
8 heads) - not the tiny synthetic test model sparse_kernel_test.py used - since
memory scales with embed_dim/heads/depth, and we want the real deployment answer.
Random data (testing the attention mechanism's memory/speed ceiling, not accuracy).
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(__file__))
import torch
from tabicl import TabICLClassifier

clf = TabICLClassifier(allow_auto_download=True)
# Trigger checkpoint load without a real fit (tiny dummy fit just to populate clf.model_)
import numpy as np
clf.fit(np.random.randn(20, 4).astype("float32"), np.random.randint(0, 2, 20))
model = clf.model_
tf_icl = model.icl_predictor.tf_icl
dev = next(model.parameters()).device
icl_dim = tf_icl.blocks[0].norm1.normalized_shape[0]  # embed_dim * row_num_cls, not model.embed_dim directly
print(f"real checkpoint: {len(tf_icl.blocks)} ICL blocks, embed_dim={model.embed_dim}, icl_dim={icl_dim}, "
      f"heads={tf_icl.blocks[0].attn.num_heads}")

for N in (50_000, 100_000, 200_000, 400_000, 700_000, 1_000_000, 1_500_000):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    try:
        X = torch.randn(1, N + 64, icl_dim, device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = tf_icl(X, train_size=N)
        torch.cuda.synchronize()
        sec = time.time() - t0
        mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"N={N:>9,}  OK   {sec:>7.2f}s   peak {mb:>8.0f} MB")
        del X, out
    except torch.cuda.OutOfMemoryError as e:
        print(f"N={N:>9,}  OOM  ({str(e).splitlines()[0]})")
        torch.cuda.empty_cache()
        break
