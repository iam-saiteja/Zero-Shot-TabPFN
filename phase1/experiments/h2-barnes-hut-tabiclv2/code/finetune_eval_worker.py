"""Single-dataset worker to evaluate a fine-tuned TabICLv2 checkpoint
(from finetune.py) on the real, held-out TabArena eval set - the same 30
datasets/500 real official OpenML splits already used for the Barnes-Hut
vanilla baseline (real_tabarena_results.csv, method="TabICLv2 (ours, vanilla)").

Same subprocess-per-dataset isolation as tabarena_worker.py (fresh CUDA
context per dataset - the fix for the two real CUDA-poisoning incidents
found earlier this project). Only evaluates the fine-tuned checkpoint
(no Barnes-Hut/anchor patching here - that's a separate, already-closed
axis) so the Elo delta isolates exactly what fine-tuning changed.

Usage: python finetune_eval_worker.py <dataset> <task_id> <problem_type> <out_csv> <meta_csv> <ckpt_path>
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

FIELDS = ["dataset", "repeat", "fold", "method", "metric_error", "time_train_s", "time_infer_s",
          "problem_type", "n_train"]
METHOD = "TabICLv2 (finetuned)"


def append_rows(new_rows, out_path):
    if not new_rows:
        return
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)


def err(ptype, yte, probs):
    if ptype == "binary":
        return 1 - roc_auc_score(yte, probs[:, 1])
    return log_loss(yte, probs, labels=np.arange(probs.shape[1]))


def main():
    name, task_id, ptype, out_path, meta_csv, ckpt_path = (
        sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    meta = pd.read_csv(meta_csv)
    splits = list(zip(meta[meta.dataset_name == name].repeat, meta[meta.dataset_name == name].fold))

    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = set(zip(prev.dataset, prev.repeat, prev.fold, prev.method))

    task = openml.tasks.get_task(task_id, download_splits=True, download_data=True)
    X_full, y_full = task.get_X_and_y(dataset_format="dataframe")
    clf = TabICLClassifier(model_path=ckpt_path, allow_auto_download=False)

    total_new = 0
    for repeat, fold in splits:
        if (name, repeat, fold, METHOD) in done:
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
            p = clf.predict_proba(Xte)
            rows.append(dict(dataset=name, repeat=repeat, fold=fold, method=METHOD,
                              metric_error=err(ptype, yte_e, p), time_train_s=fit_s,
                              time_infer_s=time.time() - t0, problem_type=ptype, n_train=len(Xtr)))
            print(f"{name} r{repeat}f{fold} n={len(Xtr)} fit={fit_s:.1f}s err={rows[0]['metric_error']:.4f}",
                  flush=True)
        except Exception as e:
            print(f"SKIP {name} r{repeat}f{fold}: {type(e).__name__}: {e}", flush=True)
        append_rows(rows, out_path)
        total_new += len(rows)

    print(f"worker done: {name}, {total_new} new rows", flush=True)


if __name__ == "__main__":
    main()
