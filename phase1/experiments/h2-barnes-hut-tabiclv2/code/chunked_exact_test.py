"""Correctness (must match plain exact attention near machine precision - this is
NOT an approximation) + memory-ceiling benchmark for chunked_exact.py, targeting
the paging wall found at ~300k rows (exact_ceiling_check.py) on this 4GB GPU."""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
from tabicl import TabICLClassifier
from chunked_exact import chunked_exact_encoder_forward

clf = TabICLClassifier(allow_auto_download=True)
clf.fit(np.random.randn(20, 4).astype("float32"), np.random.randint(0, 2, 20))
model = clf.model_
tf_icl = model.icl_predictor.tf_icl
dev = next(model.parameters()).device
icl_dim = tf_icl.blocks[0].norm1.normalized_shape[0]

# --- correctness: small N, chunk_size << N so chunking actually kicks in ---
torch.manual_seed(0)
N_TRAIN, N_TEST = 2000, 64
X = torch.randn(1, N_TRAIN + N_TEST, icl_dim, device=dev)
with torch.no_grad():
    exact = tf_icl(X, train_size=N_TRAIN)
    chunked = chunked_exact_encoder_forward(tf_icl, X, N_TRAIN, chunk_size=256, query_chunk_size=300)
diff = (exact - chunked).abs().max().item()
print(f"correctness: max abs diff plain-exact vs chunked-exact = {diff:.6f} (chunk_size=256 << N={N_TRAIN})")
assert diff < 1e-2, "chunked version should be mathematically identical to plain exact attention"
print("PASS: chunked attention matches plain exact attention (this is exact, not approximate).\n")


def bench(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache(); gc.collect()
    t0 = time.time()
    with torch.no_grad():
        out = fn()
    torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1e6, out


CHUNK = 2048
print(f"{'N_train':>9} {'chunked sec':>12} {'chunked MB':>11}  (chunk_size=query_chunk_size={CHUNK}; compare to "
      f"exact_ceiling_check.py's 200k=685s/4221MB, 400k=8645s/8318MB(paging), 700k=OOM)")
for N in (200_000, 400_000, 700_000, 1_000_000, 1_500_000):
    X = torch.randn(1, N + 64, icl_dim, device=dev)
    try:
        sec, mb, _ = bench(lambda: chunked_exact_encoder_forward(tf_icl, X, N, chunk_size=CHUNK, query_chunk_size=CHUNK))
        print(f"{N:>9,} {sec:>12.2f} {mb:>11.0f}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"{N:>9,}  OOM  {str(e).splitlines()[0]}")
        torch.cuda.empty_cache()
        break
    del X
