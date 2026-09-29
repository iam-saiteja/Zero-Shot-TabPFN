"""Single-dataset worker for real_tabarena_run.py. Run as a fresh subprocess per
dataset (not imported) - this is the actual fix for the CUDA context poisoning
found twice now: once from an extreme-feature-count dataset (Bioresponse), once
from an ordinary one (taiwanese_bankruptcy_prediction) after dozens of datasets
had already run in the same process - cumulative CUDA memory fragmentation, not
a property of any single "bad" dataset. A fresh process per dataset means a
fresh CUDA context every time: no accumulation, and a crash on one dataset
can't touch any other.

Usage: python tabarena_worker.py <dataset_name> <task_id> <problem_type> <out_csv>
Appends its rows directly to out_csv (one dataset's worth) and exits 0 on
success or partial success (per-split failures inside are caught and skipped,
same as before) - only a hard crash (e.g. this dataset is ALSO pathological)
exits non-zero, visible to the orchestrator without touching anything else.
"""
import sys, os, csv, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import openml
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from tabicl import TabICLClassifier
from patch import patch_tf_icl, unpatch_tf_icl
from patch_nearfield import patch_tf_icl_bh, unpatch_tf_icl_bh

FIELDS = ["dataset", "repeat", "fold", "method", "metric_error", "time_train_s", "time_infer_s",
          "problem_type", "n_train"]


def append_rows(new_rows):
    """Append-only, never re-reads/rewrites the whole file - avoids the duplicate-row
    bug a naive read-concat-overwrite-per-split loop would cause."""
    if not new_rows:
        return
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)

name, task_id, ptype, out_path, meta_csv = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
meta = pd.read_csv(meta_csv)
splits = list(zip(meta[meta.dataset_name == name].repeat, meta[meta.dataset_name == name].fold))

done = set()
if os.path.exists(out_path):
    prev = pd.read_csv(out_path)
    done = set(zip(prev.dataset, prev.repeat, prev.fold, prev.method))


def err(ptype, yte, probs):
    if ptype == "binary":
        return 1 - roc_auc_score(yte, probs[:, 1])
    return log_loss(yte, probs, labels=np.arange(probs.shape[1]))


task = openml.tasks.get_task(task_id, download_splits=True, download_data=True)
X_full, y_full = task.get_X_and_y(dataset_format="dataframe")
clf = TabICLClassifier(allow_auto_download=True)

total_new = 0
for repeat, fold in splits:
    methods = ["vanilla", "anchor", "barnes_hut"]
    if all((name, repeat, fold, f"TabICLv2+{m} (ours)" if m != "vanilla" else "TabICLv2 (ours, vanilla)") in done
           for m in methods):
        continue
    rows = []
    try:
        train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat)
        Xtr, Xte = X_full.iloc[train_idx], X_full.iloc[test_idx]
        ytr, yte = y_full.iloc[train_idx], y_full.iloc[test_idx]
        le = LabelEncoder().fit(pd.concat([ytr, yte]))
        ytr_e, yte_e = le.transform(ytr), le.transform(yte)

        fit_t0 = time.time()
        clf.fit(Xtr, ytr_e)
        fit_s = time.time() - fit_t0

        t0 = time.time()
        p_van = clf.predict_proba(Xte)
        rows.append(dict(dataset=name, repeat=repeat, fold=fold, method="TabICLv2 (ours, vanilla)",
                          metric_error=err(ptype, yte_e, p_van), time_train_s=fit_s,
                          time_infer_s=time.time() - t0, problem_type=ptype, n_train=len(Xtr)))

        model = clf.model_
        for tag, patch_fn, unpatch_fn, kw in [("anchor", patch_tf_icl, unpatch_tf_icl, dict(M=64, iters=3)),
                                               ("barnes_hut", patch_tf_icl_bh, unpatch_tf_icl_bh,
                                                dict(M=64, iters=3, t=4))]:
            t0 = time.time()
            patch_fn(model, **kw)
            p = clf.predict_proba(Xte)
            unpatch_fn(model)
            rows.append(dict(dataset=name, repeat=repeat, fold=fold, method=f"TabICLv2+{tag} (ours)",
                              metric_error=err(ptype, yte_e, p), time_train_s=fit_s,
                              time_infer_s=time.time() - t0, problem_type=ptype, n_train=len(Xtr)))
        print(f"{name} r{repeat}f{fold} n={len(Xtr)} fit={fit_s:.1f}s "
              f"errs={[round(r['metric_error'], 4) for r in rows[-3:]]}", flush=True)
    except Exception as e:
        print(f"SKIP {name} r{repeat}f{fold}: {type(e).__name__}: {e}", flush=True)
    append_rows(rows)
    total_new += len(rows)

print(f"worker done: {name}, {total_new} new rows", flush=True)
