"""The actually-correct 'build it properly' fix: predict_proba's DEFAULT path
(kv_cache=False, the common case) calls self.model_(...) -> TabICL.forward ->
icl_predictor.forward -> tf_icl.forward - exactly Encoder.forward, which
patch_nearfield.py/patch.py already patch and validated (H1's TabPFN v1 sweep,
H2's 16/16 and 9/11 real-dataset results). patch_kvcache.py (forward_with_cache)
is real but only matters for the opt-in kv_cache=True feature, not this.

So: apply patch_tf_icl_bh AFTER a normal .fit() (real preprocessing), call the
REAL .predict_proba() (real ensemble-of-feature-shuffles averaging) - not the
_train_forward bypass used in sweep.py/tabarena_tiny_eval.py. Runs the same
real TabArena-tiny dataset list as before, to see if this closes the earlier
vanilla-vs-official-cache mismatch.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
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

out_path = os.path.join(os.path.dirname(__file__), "..", "results", "real_api_rows.csv")
rows = []
for name in CLASSIFICATION:
    try:
        d = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
        X, y_raw = d.data, d.target.astype(str)
        y = LabelEncoder().fit_transform(y_raw)
        ptype = "binary" if len(set(y)) == 2 else "multiclass"
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                               stratify=y if min(np.bincount(y)) >= 2 else None)

        clf = TabICLClassifier(allow_auto_download=True)
        clf.fit(Xtr, ytr)
        p_van = clf.predict_proba(Xte)

        def err(p):
            if ptype == "binary":
                return 1 - roc_auc_score(yte, p[:, 1])
            return log_loss(yte, p, labels=np.arange(p.shape[1]))

        van_err = err(p_van)
        model = clf.model_
        patch_tf_icl_bh(model, M=64, iters=3, t=4)
        p_bh = clf.predict_proba(Xte)
        unpatch_tf_icl_bh(model)
        bh_err = err(p_bh)

        rows.append(dict(dataset=name, ptype=ptype, n_train=len(Xtr),
                          our_vanilla=round(van_err, 4), our_bh=round(bh_err, 4)))
        print(name, ptype, f"n={len(Xtr)}", "vanilla", round(van_err, 4), "bh", round(bh_err, 4), flush=True)
    except Exception as e:
        print(f"SKIP {name}: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

df = pd.DataFrame(rows)
off = ref[(ref.method == "TABICLV2 (default)") & (ref.dataset.isin(df.dataset))][["dataset", "metric_error"]]
off.columns = ["dataset", "official_cached"]
cmp = df.merge(off, on="dataset")
cmp["gap"] = (cmp.our_vanilla - cmp.official_cached).abs()
print("\n=== our vanilla vs official cached TABICLV2(default), via REAL predict_proba ===")
print(cmp[["dataset", "official_cached", "our_vanilla", "gap"]].round(4).to_string())
print(f"\nmean abs gap: {cmp.gap.mean():.4f} (was much larger with the _train_forward bypass)")
print(f"bh beats vanilla error on: {(cmp.our_bh < cmp.our_vanilla).sum()}/{len(cmp)}")
