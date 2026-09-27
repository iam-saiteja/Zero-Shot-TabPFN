"""Evaluate on the REAL TabArena-tiny dataset names (harness/tabarena_tiny_results_reference),
so results are genuinely comparable to the real cached official baselines sitting there -
not another arbitrary dataset list. Classification-only (13 of 18 datasets); TabICLClassifier
doesn't cover regression here. Single train/test split per dataset (the reference itself is
single-fold, "tiny" - not the full 30-split protocol), via _train_forward directly (same
caveat as sweep.py: bypasses TabICLClassifier's normal preprocessing/caching inference path).
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from tabicl import TabICLClassifier
from patch import patch_tf_icl, unpatch_tf_icl
from patch_nearfield import patch_tf_icl_bh, unpatch_tf_icl_bh

REF = os.path.join(os.path.dirname(__file__), "..", "..", "..", "harness",
                    "tabarena_tiny_results_reference", "eval", "results_per_split.csv")
ref = pd.read_csv(REF)
problem_type = ref.groupby("dataset").problem_type.first()
CLASSIFICATION = sorted(problem_type[problem_type.isin(["binary", "multiclass"])].index)
print(f"{len(CLASSIFICATION)} classification datasets: {CLASSIFICATION}")

clf = TabICLClassifier(allow_auto_download=True)
rows = []
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "tabarena_tiny_new_rows.csv")

for name in CLASSIFICATION:
    try:
        d = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
        X = d.data.copy()
        for c in X.columns:
            if str(X[c].dtype) in ("category", "object"):
                X[c] = X[c].astype("category").cat.codes
        X = X.astype(np.float32).fillna(-1).values
        y = LabelEncoder().fit_transform(d.target.astype(str))
        n_classes = len(set(y))
        ptype = "binary" if n_classes == 2 else "multiclass"

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                               stratify=y if min(np.bincount(y)) >= 2 else None)
        if len(Xtr) > 2000:
            Xtr, ytr = Xtr[:2000], ytr[:2000]
        train_size = len(Xtr)

        t0 = time.time()
        clf.fit(Xtr, ytr)
        fit_s = time.time() - t0
        model = clf.model_
        dev = next(model.parameters()).device
        Xt = torch.tensor(np.concatenate([Xtr, Xte]), dtype=torch.float32, device=dev).unsqueeze(0)
        yt = torch.tensor(ytr, dtype=torch.long, device=dev).unsqueeze(0)

        def evaluate(logits):
            probs = logits[0].softmax(-1).cpu().numpy()
            if ptype == "binary":
                err = 1 - roc_auc_score(yte, probs[:, 1])
            else:
                labels = np.arange(probs.shape[1])
                err = log_loss(yte, probs, labels=labels)
            return err

        t0 = time.time()
        with torch.no_grad():
            vanilla = model._train_forward(Xt, yt)
        infer_s = time.time() - t0
        rows.append(dict(dataset=name, fold=0, method="TabICLv2 (ours, vanilla)",
                          metric_error=evaluate(vanilla), time_train_s=fit_s, time_infer_s=infer_s,
                          metric="roc_auc" if ptype == "binary" else "log_loss", problem_type=ptype))

        for tag, patch_fn, unpatch_fn, kw in [
            ("anchor-only", patch_tf_icl, unpatch_tf_icl, dict(M=64, iters=3)),
            ("barnes-hut", patch_tf_icl_bh, unpatch_tf_icl_bh, dict(M=64, iters=3, t=4)),
        ]:
            patch_fn(model, **kw)
            t0 = time.time()
            with torch.no_grad():
                out = model._train_forward(Xt, yt)
            infer_s = time.time() - t0
            unpatch_fn(model)
            rows.append(dict(dataset=name, fold=0, method=f"TabICLv2+{tag} (ours)",
                              metric_error=evaluate(out), time_train_s=fit_s, time_infer_s=infer_s,
                              metric="roc_auc" if ptype == "binary" else "log_loss", problem_type=ptype))

        print(name, ptype, f"n_train={train_size}",
              {r["method"]: round(r["metric_error"], 4) for r in rows[-3:]}, flush=True)
    except Exception as e:
        print(f"SKIP {name}: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

print("done", len(rows), "rows ->", out_path)
