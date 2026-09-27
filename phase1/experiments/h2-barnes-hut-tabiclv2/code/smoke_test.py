"""H2 smoke test: verify the tf_icl patch (patch.py) per protocol.md's sanity checks.
Untrained random-init TabICL (default config) - this checks the PATCH MECHANISM
(shapes, determinism, fallback, scoping), not real accuracy. Run:
phase1/.venv-tabicl/Scripts/python.exe phase1/experiments/h2-barnes-hut-tabiclv2/code/smoke_test.py
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(__file__))
import torch
from tabicl._model.tabicl import TabICL
from patch import patch_tf_icl, unpatch_tf_icl

torch.manual_seed(0)
B, N_TRAIN, N_TEST, H = 1, 300, 20, 8
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = TabICL(embed_dim=32, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=2,
               col_nhead=4, row_nhead=4, icl_nhead=4, max_classes=3, zero_init=False)
model.eval().to(dev)

X = torch.randn(B, N_TRAIN + N_TEST, H, device=dev)
y_train = torch.randint(0, 3, (B, N_TRAIN), device=dev)

with torch.no_grad():
    vanilla = model._train_forward(X, y_train)
    vanilla2 = model._train_forward(X, y_train)
assert torch.equal(vanilla, vanilla2), "sanity (1) failed: vanilla is not deterministic"
print("(1) vanilla deterministic: OK")

col_fwd_before = model.col_embedder.tf_col.forward

patch_tf_icl(model, M=N_TRAIN + 10, iters=3)  # M >= train_size -> must take the fallback branch
with torch.no_grad():
    fallback = model._train_forward(X, y_train)
assert torch.equal(fallback, vanilla), "sanity (2) failed: M>=train_size fallback should equal vanilla exactly"
print("(2) M>=train_size fallback == vanilla: OK")

patch_tf_icl(model, M=32, iters=3)  # real anchor path
with torch.no_grad():
    patched = model._train_forward(X, y_train)
assert patched.shape == vanilla.shape, f"shape mismatch: {patched.shape} vs {vanilla.shape}"
assert torch.isfinite(patched).all(), "sanity (4) failed: non-finite values"
print(f"(4) shape {patched.shape} matches, all finite: OK")

assert model.col_embedder.tf_col.forward == col_fwd_before, "sanity (3) failed: tf_col was touched"
print("(3) tf_col untouched (identity check): OK")

fid = (patched - vanilla).abs().mean().item()
rel = fid / vanilla.abs().mean().clamp(min=1e-8).item()
print(f"patched-vs-vanilla mean abs diff: {fid:.4f} (relative: {rel:.4f}) at M=32,train_size={N_TRAIN}")
print("gate check (protocol.md): expect same order of magnitude as H1's 0.02-0.06, not >0.2 -> "
      + ("PASS" if fid < 0.2 else "FAIL, investigate"))

unpatch_tf_icl(model)
with torch.no_grad():
    restored = model._train_forward(X, y_train)
assert torch.equal(restored, vanilla), "unpatch did not restore vanilla behavior"
print("unpatch restores vanilla exactly: OK")

print("\nAll H2 smoke-test checks passed (patch mechanism verified on random-init weights;"
      " NOT a trained-model accuracy result).")
