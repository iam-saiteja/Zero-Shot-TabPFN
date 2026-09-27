"""Compare anchor-only vs near-field (Barnes-Hut) patch on the real TabICLv2 checkpoint."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from tabicl import TabICLClassifier
from patch import patch_tf_icl, unpatch_tf_icl
from patch_nearfield import patch_tf_icl_bh, unpatch_tf_icl_bh

clf = TabICLClassifier(allow_auto_download=True)
d = fetch_openml(data_id=1489, as_frame=True, parser="auto")
X = d.data.select_dtypes("number").fillna(-1).values.astype(np.float32)
y = LabelEncoder().fit_transform(d.target)
rng = np.random.RandomState(0)
idx = rng.permutation(len(X))[:800]
X, y = X[idx], y[idx]
X_train, y_train, X_test = X[:600], y[:600], X[600:]
clf.fit(X_train, y_train)
model = clf.model_
dev = next(model.parameters()).device

Xt = torch.tensor(np.concatenate([X_train, X_test]), dtype=torch.float32, device=dev).unsqueeze(0)
yt = torch.tensor(y_train, dtype=torch.long, device=dev).unsqueeze(0)

with torch.no_grad():
    vanilla = model._train_forward(Xt, yt)
van_pred = vanilla.argmax(-1).cpu().numpy()


def score(name, patched):
    assert torch.isfinite(patched).all(), f"{name}: non-finite output"
    fid = (patched.softmax(-1) - vanilla.softmax(-1)).abs().mean().item()
    agree = (patched.argmax(-1).cpu().numpy() == van_pred).mean()
    print(f"{name:28s} fidelity={fid:.4f}  pred-agreement-with-exact={agree:.3f}")


for M in (32, 64):
    patch_tf_icl(model, M=M, iters=3)
    with torch.no_grad():
        p = model._train_forward(Xt, yt)
    unpatch_tf_icl(model)
    score(f"anchor-only M={M}", p)

    for t in (2, 4, 8):
        patch_tf_icl_bh(model, M=M, iters=3, t=t)
        with torch.no_grad():
            p = model._train_forward(Xt, yt)
        unpatch_tf_icl_bh(model)
        score(f"barnes-hut M={M},t={t}", p)
