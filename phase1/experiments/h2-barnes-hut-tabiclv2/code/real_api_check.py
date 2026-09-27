"""Smoke test for patch_kvcache.py through the REAL public API (TabICLClassifier
.fit()/.predict_proba()), with the classifier's own real preprocessing - not the
_train_forward bypass used in earlier H2 scripts. Patch must be applied BEFORE
.fit() (see patch_kvcache.py's docstring for why)."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier
from patch_kvcache import patch_tf_icl_kvcache, unpatch_tf_icl_kvcache

d = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
X, y = d.data, LabelEncoder().fit_transform(d.target.astype(str))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print("Vanilla (real preprocessing, real predict_proba path)...")
clf_vanilla = TabICLClassifier(allow_auto_download=True)
clf_vanilla.fit(Xtr, ytr)
p_vanilla = clf_vanilla.predict_proba(Xte)
err_vanilla = 1 - roc_auc_score(yte, p_vanilla[:, 1])
print(f"  vanilla error: {err_vanilla:.4f}")

print("Barnes-Hut (patched BEFORE fit, per docstring)...")
patch_tf_icl_kvcache(M=64, iters=3, t=4)
clf_bh = TabICLClassifier(allow_auto_download=True)
clf_bh.fit(Xtr, ytr)
p_bh = clf_bh.predict_proba(Xte)
unpatch_tf_icl_kvcache()
err_bh = 1 - roc_auc_score(yte, p_bh[:, 1])
print(f"  barnes-hut error: {err_bh:.4f}")

assert np.isfinite(p_bh).all(), "non-finite probabilities"
assert p_bh.shape == p_vanilla.shape
fid = np.abs(p_bh - p_vanilla).mean()
agree = (p_bh.argmax(1) == p_vanilla.argmax(1)).mean()
print(f"fidelity(mean abs diff)={fid:.4f}  pred-agreement={agree:.3f}")

# sanity: unpatched classifier built AFTER unpatch must behave like a fresh vanilla one
clf_check = TabICLClassifier(allow_auto_download=True)
clf_check.fit(Xtr, ytr)
p_check = clf_check.predict_proba(Xte)
assert np.allclose(p_check, p_vanilla, atol=1e-4), "unpatch did not fully restore vanilla behavior"
print("unpatch restores vanilla exactly (checked with a fresh classifier): OK")
