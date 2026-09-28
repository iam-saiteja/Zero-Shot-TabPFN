"""The real TabArena v0.1 protocol run: official 51-task suite, official OpenML
splits (repeat x fold, per each task's real protocol - 30 splits for smaller
tasks, 9 for larger ones), real preprocessing, real predict_proba() path.
Classification only (38 of 51 tasks) - TabICLClassifier doesn't do regression.

Ordered smallest-to-largest by train row count so usable partial results
accumulate quickly; the largest tasks (up to 100k rows) will be slow given
this hardware's own measured ceiling (100k rows took 156s for just the ICL
attention forward pass - real fit+predict will be more). Resumable: skips
(dataset, repeat, fold) triples already in the output CSV.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import openml
from sklearn.metrics import roc_auc_score, log_loss
from tabicl import TabICLClassifier
from patch import patch_tf_icl, unpatch_tf_icl
from patch_nearfield import patch_tf_icl_bh, unpatch_tf_icl_bh

META_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "..", "harness", "tabarena", "src", "tabarena",
                        "benchmark", "task", "metadata", "sources", "data", "TabArena-v0.1_tasks_metadata.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "real_tabarena_results.csv")

meta = pd.read_csv(META_CSV)
meta = meta[meta.problem_type.isin(["binary", "multiclass"])]
tasks = (meta.groupby("dataset_name")
         .agg(task_id=("task_id_str", "first"), problem_type=("problem_type", "first"),
              n_train=("num_instances_train", "first"))
         .sort_values("n_train"))
splits_by_task = meta.groupby("dataset_name")[["repeat", "fold"]].apply(lambda d: list(zip(d.repeat, d.fold)))
print(f"{len(tasks)} classification tasks, {sum(len(s) for s in splits_by_task)} total splits", flush=True)

done = set()
if os.path.exists(OUT_PATH):
    prev = pd.read_csv(OUT_PATH)
    done = set(zip(prev.dataset, prev.repeat, prev.fold, prev.method))
    print(f"resuming: {len(prev)} rows already done", flush=True)

rows = [] if not os.path.exists(OUT_PATH) else pd.read_csv(OUT_PATH).to_dict("records")


def err(ptype, yte, probs):
    # yte is LabelEncoder-integer-encoded; classes_ sorted alphabetically, so index 1
    # is always the "second" class regardless of what the original string labels were.
    if ptype == "binary":
        return 1 - roc_auc_score(yte, probs[:, 1])
    labels = np.arange(probs.shape[1])
    return log_loss(yte, probs, labels=labels)


clf = TabICLClassifier(allow_auto_download=True)
for name, row in tasks.iterrows():
    ptype = row.problem_type
    task_id = int(row.task_id)
    try:
        task = openml.tasks.get_task(task_id, download_splits=True, download_data=True)
        X_full, y_full = task.get_X_and_y(dataset_format="dataframe")
    except Exception as e:
        print(f"SKIP {name}: failed to load task {task_id}: {e}", flush=True)
        continue
    for repeat, fold in splits_by_task[name]:
        methods = ["vanilla", "anchor", "barnes_hut"]
        if all((name, repeat, fold, f"TabICLv2+{m} (ours)" if m != "vanilla" else "TabICLv2 (ours, vanilla)") in done
               for m in methods):
            continue
        try:
            train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat)
            Xtr, Xte = X_full.iloc[train_idx], X_full.iloc[test_idx]
            ytr, yte = y_full.iloc[train_idx], y_full.iloc[test_idx]
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder().fit(pd.concat([ytr, yte]))
            ytr_e, yte_e = le.transform(ytr), le.transform(yte)

            fit_t0 = time.time()
            clf.fit(Xtr, ytr_e)
            fit_s = time.time() - fit_t0

            t0 = time.time()
            p_van = clf.predict_proba(Xte)
            infer_s = time.time() - t0
            rows.append(dict(dataset=name, repeat=repeat, fold=fold, method="TabICLv2 (ours, vanilla)",
                              metric_error=err(ptype, yte_e, p_van), time_train_s=fit_s,
                              time_infer_s=infer_s, problem_type=ptype, n_train=len(Xtr)))

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
                  f"errs={[round(r['metric_error'],4) for r in rows[-3:]]}", flush=True)
        except Exception as e:
            print(f"SKIP {name} r{repeat}f{fold}: {type(e).__name__}: {e}", flush=True)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

print("done", len(rows), "rows ->", OUT_PATH, flush=True)
