"""Correctness (vs the dense masked version already validated) + real speed/memory
benchmark for sparse_kernel.py, on the actual hardware this whole project has been
running on (checked: NVIDIA RTX 3050 Laptop GPU, 4GB)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from tabicl._model.tabicl import TabICL
from patch_nearfield import _patched_forward as dense_masked_forward
from sparse_kernel import sparse_bh_encoder_forward

torch.manual_seed(0)
dev = "cuda"
model = TabICL(embed_dim=32, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=2, row_num_cls=1,
               col_nhead=4, row_nhead=4, icl_nhead=4, max_classes=3, zero_init=False)
model.eval().to(dev)
tf_icl = model.icl_predictor.tf_icl

# --- Correctness: same clusters both paths (fix the seed right before each call) ---
B, N_TRAIN, N_TEST, d = 1, 512, 32, 32
X = torch.randn(B, N_TRAIN + N_TEST, d, device=dev)

torch.manual_seed(1)
dense = dense_masked_forward(tf_icl, X, N_TRAIN, M=64, iters=3, t=4)
torch.manual_seed(1)
sparse = sparse_bh_encoder_forward(tf_icl, X, N_TRAIN, M=64, iters=3, t=4, cap=N_TRAIN)  # generous cap: no truncation, isolates correctness from the cap/accuracy tradeoff
diff = (dense - sparse).abs().max().item()
print(f"correctness: max abs diff dense-vs-sparse = {diff:.6f} (same RNG seed -> same clusters/top-t)")
assert diff < 1e-3, "sparse kernel does not match the validated dense version"
print("PASS: sparse kernel matches the already-validated dense masked implementation.\n")


def bench(fn, *a, **kw):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.time()
    with torch.no_grad():
        out = fn(*a, **kw)
    torch.cuda.synchronize()
    sec = time.time() - t0
    mb = torch.cuda.max_memory_allocated() / 1e6
    return sec, mb, out


print(f"{'N_train':>8} {'exact sec':>10} {'exact MB':>9} | {'dense-BH sec':>12} {'dense-BH MB':>11} | "
      f"{'sparse-BH sec':>13} {'sparse-BH MB':>12}")
for N_TRAIN in (1000, 4000, 16000, 40000):
    X = torch.randn(1, N_TRAIN + 64, d, device=dev)
    try:
        sec_e, mb_e, _ = bench(lambda: tf_icl(X, train_size=N_TRAIN))
    except torch.cuda.OutOfMemoryError:
        sec_e, mb_e = float("nan"), float("nan")
        torch.cuda.empty_cache()
    try:
        sec_d, mb_d, _ = bench(lambda: dense_masked_forward(tf_icl, X, N_TRAIN, M=64, iters=3, t=4))
    except torch.cuda.OutOfMemoryError:
        sec_d, mb_d = float("nan"), float("nan")
        torch.cuda.empty_cache()
    try:
        sec_s, mb_s, _ = bench(lambda: sparse_bh_encoder_forward(tf_icl, X, N_TRAIN, M=64, iters=3, t=4))
    except torch.cuda.OutOfMemoryError:
        sec_s, mb_s = float("nan"), float("nan")
        torch.cuda.empty_cache()
    print(f"{N_TRAIN:>8} {sec_e:>10.3f} {mb_e:>9.0f} | {sec_d:>12.3f} {mb_d:>11.0f} | {sec_s:>13.3f} {mb_s:>12.0f}")
