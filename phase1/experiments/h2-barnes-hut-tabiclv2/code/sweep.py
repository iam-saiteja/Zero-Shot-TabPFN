"""H2 confirmatory sweep: same 10 OpenML datasets + 2 seeds as H1's pilot.py,
now that the near-field bug is fixed. anchor-only vs barnes-hut (M=64,t=4)
vs vanilla, on the real TabICLv2 checkpoint. Still bypasses the sklearn
cache path (_train_forward direct call)."""
import sys, os, csv, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier
from patch import patch_tf_icl, unpatch_tf_icl
from patch_nearfield import patch_tf_icl_bh, unpatch_tf_icl_bh

DATASETS = {"adult": 1590, "bank-marketing": 1461, "electricity": 151, "MagicTelescope": 1120,
            "pendigits": 32, "phoneme": 1489, "spambase": 44, "satimage": 182, "jm1": 1053, "kr-vs-kp": 3}
N_TRAIN, N_TEST = 600, 200


def load(name, seed):
    d = fetch_openml(data_id=DATASETS[name], as_frame=True, parser="auto")
    X = d.data.copy()
    for c in X.columns:
        if str(X[c].dtype) in ("category", "object"):
            X[c] = X[c].astype("category").cat.codes
    X = X.astype(np.float32).fillna(-1).values
    y = LabelEncoder().fit_transform(d.target)
    rng = np.random.RandomState(seed)
    p = rng.permutation(len(X))
    nt = min(N_TRAIN, len(X) - N_TEST)
    return X[p[:nt]], y[p[:nt]], X[p[nt:nt + N_TEST]], y[p[nt:nt + N_TEST]]


if __name__ == "__main__":
    clf = TabICLClassifier(allow_auto_download=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "sweep_results.csv")
    rows = []
    for name in DATASETS:
        for seed in (0, 1):
            try:
                Xtr, ytr, Xte, yte = load(name, seed)
                clf.fit(Xtr, ytr)
                model = clf.model_
                dev = next(model.parameters()).device
                Xt = torch.tensor(np.concatenate([Xtr, Xte]), dtype=torch.float32, device=dev).unsqueeze(0)
                yt = torch.tensor(ytr, dtype=torch.long, device=dev).unsqueeze(0)
                train_size = len(Xtr)

                with torch.no_grad():
                    vanilla = model._train_forward(Xt, yt)
                van_pred = vanilla.argmax(-1)[0].cpu().numpy()
                van_acc = accuracy_score(yte, van_pred)

                patch_tf_icl(model, M=64, iters=3)
                with torch.no_grad():
                    p_anchor = model._train_forward(Xt, yt)
                unpatch_tf_icl(model)

                patch_tf_icl_bh(model, M=64, iters=3, t=4)
                with torch.no_grad():
                    p_bh = model._train_forward(Xt, yt)
                unpatch_tf_icl_bh(model)

                for tag, p in (("anchor", p_anchor), ("barnes_hut", p_bh)):
                    fid = (p.softmax(-1) - vanilla.softmax(-1)).abs().mean().item()
                    agree = (p.argmax(-1)[0].cpu().numpy() == van_pred).mean()
                    acc = accuracy_score(yte, p.argmax(-1)[0].cpu().numpy())
                    rows.append(dict(dataset=name, seed=seed, train_size=train_size, variant=tag,
                                      fidelity=round(fid, 4), agreement=round(float(agree), 4),
                                      acc=round(acc, 4), van_acc=round(van_acc, 4),
                                      acc_delta_pp=round(100 * (acc - van_acc), 2)))
                print(name, seed, "van_acc", round(van_acc, 3),
                      {r["variant"]: (r["fidelity"], r["agreement"]) for r in rows[-2:]}, flush=True)
            except Exception as e:
                print(f"SKIP {name} seed={seed}: {e}", flush=True)
            import pandas as pd
            pd.DataFrame(rows).to_csv(out_path, index=False)
    print("done", len(rows), "rows ->", out_path)
