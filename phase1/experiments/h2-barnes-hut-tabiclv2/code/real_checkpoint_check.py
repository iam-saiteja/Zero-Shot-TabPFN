"""Step 1 of analysis.md's next-steps: rerun the tf_icl patch against TabICLv2's
REAL pretrained checkpoint (downloaded via HF Hub), on a real small OpenML
dataset, instead of random-init weights and random tensors. Still uses the
anchor-only patch (near-field not yet built) and still bypasses the sklearn
cache path (calls _train_forward directly) - so this is real weights + real
data, but still not the harness/real-TabArena-protocol number.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from tabicl import TabICLClassifier
from patch import patch_tf_icl, unpatch_tf_icl

print("Loading TabICLv2 pretrained checkpoint (downloads on first use)...")
clf = TabICLClassifier(allow_auto_download=True)

d = fetch_openml(data_id=1489, as_frame=True, parser="auto")  # phoneme, same dataset H1 used
X = d.data.select_dtypes("number").fillna(-1).values.astype(np.float32)
y = LabelEncoder().fit_transform(d.target)
rng = np.random.RandomState(0)
idx = rng.permutation(len(X))[:800]
X, y = X[idx], y[idx]
X_train, y_train, X_test = X[:600], y[:600], X[600:]

clf.fit(X_train, y_train)
model = clf.model_
dev = next(model.parameters()).device
print(f"Loaded on {dev}, tf_icl blocks: {len(model.icl_predictor.tf_icl.blocks)}, embed_dim: {model.embed_dim}")

Xt = torch.tensor(np.concatenate([X_train, X_test]), dtype=torch.float32, device=dev).unsqueeze(0)
yt = torch.tensor(y_train, dtype=torch.long, device=dev).unsqueeze(0)

with torch.no_grad():
    vanilla = model._train_forward(Xt, yt)

results = {}
for M in (16, 32, 64, 128):
    patch_tf_icl(model, M=M, iters=3)
    with torch.no_grad():
        patched = model._train_forward(Xt, yt)
    unpatch_tf_icl(model)
    assert torch.isfinite(patched).all()
    fid = (patched.softmax(-1) - vanilla.softmax(-1)).abs().mean().item()
    van_pred = vanilla.argmax(-1).cpu().numpy()
    pat_pred = patched.argmax(-1).cpu().numpy()
    agree = (van_pred == pat_pred).mean()
    results[M] = (fid, agree)
    print(f"M={M:4d}  train_size=600  softmax-fidelity(mean abs diff)={fid:.4f}  "
          f"pred-agreement-with-exact={agree:.3f}")

print("\nNot yet: near-field expansion, real held-out accuracy vs labels, harness/TabArena protocol.")
