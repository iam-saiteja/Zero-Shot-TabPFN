"""Correctness (vs the already-validated dense masked version) + real speed/memory
benchmark for flex_attention_kernel.py, at the scale where the hand-rolled sparse
kernel and chunked-exact attention both failed."""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(__file__))
import torch
from tabicl._model.tabicl import TabICL
from patch_nearfield import _patched_forward as dense_masked_forward
from flex_attention_kernel import flex_bh_encoder_forward

torch.manual_seed(0)
dev = "cuda"
model = TabICL(embed_dim=64, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1, row_num_cls=1,
               col_nhead=4, row_nhead=4, icl_nhead=4, max_classes=3, zero_init=False, icl_ssmax=False).eval().to(dev)
tf_icl = model.icl_predictor.tf_icl

B, N_TRAIN, N_TEST, d = 1, 2048, 128, 64  # head_dim=16 (4 heads) - FlexAttention/Triton needs >=16
X = torch.randn(B, N_TRAIN + N_TEST, d, device=dev)
torch.manual_seed(1)
dense = dense_masked_forward(tf_icl, X, N_TRAIN, M=64, iters=3, t=4)
torch.manual_seed(1)
flexed = flex_bh_encoder_forward(tf_icl, X, N_TRAIN, M=64, iters=3, t=4)
diff = (dense - flexed).abs().max().item()
print(f"correctness vs dense-masked (already validated): max abs diff = {diff:.6f}")
assert diff < 1e-1, "flex kernel does not match the validated version"
print("PASS\n")


def bench(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache(); gc.collect()
    t0 = time.time()
    with torch.no_grad():
        out = fn()
    torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1e6, out


print(f"{'N_train':>9} | {'exact sec':>10} {'exact MB':>9} | {'flex-BH sec':>12} {'flex-BH MB':>11}  "
      f"(compare: exact_ceiling_check.py 200k=685s/4221MB, 400k=8645s/8318MB(paging), 700k=OOM)")
for N in (50_000, 200_000, 400_000, 700_000):
    X = torch.randn(1, N + 128, d, device=dev)
    try:
        sec_e, mb_e, _ = bench(lambda: tf_icl(X, train_size=N))
    except torch.cuda.OutOfMemoryError:
        sec_e, mb_e = float("nan"), float("nan")
        torch.cuda.empty_cache()
    try:
        sec_f, mb_f, _ = bench(lambda: flex_bh_encoder_forward(tf_icl, X, N, M=64, iters=3, t=4))
    except torch.cuda.OutOfMemoryError as e:
        print(f"{N:>9,} | {sec_e:>10.2f} {mb_e:>9.0f} | OOM  {str(e).splitlines()[0][:60]}")
        torch.cuda.empty_cache()
        del X
        continue
    print(f"{N:>9,} | {sec_e:>10.2f} {mb_e:>9.0f} | {sec_f:>12.2f} {mb_f:>11.0f}")
    del X
