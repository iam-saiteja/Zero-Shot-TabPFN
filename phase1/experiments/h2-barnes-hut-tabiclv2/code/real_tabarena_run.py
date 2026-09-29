"""Orchestrator: spawns tabarena_worker.py as a FRESH SUBPROCESS per dataset.

Two real CUDA-context-poisoning incidents in one long-lived process (once from
an extreme-feature-count dataset, once from an ordinary one after dozens of
datasets had already run) made this necessary - the fix isn't blacklisting
datasets one at a time, it's isolating each dataset's CUDA context so nothing
accumulates and no single crash touches any other dataset. See
tabarena_worker.py's docstring and docs/research-log.md for the full story.

Still resumable: the worker itself skips already-done (dataset,repeat,fold,
method) rows, so re-running this orchestrator picks up wherever it left off,
same as before.
"""
import os, subprocess, sys
import pandas as pd

HERE = os.path.dirname(__file__)
META_CSV = os.path.join(HERE, "..", "..", "..", "harness", "tabarena", "src", "tabarena",
                        "benchmark", "task", "metadata", "sources", "data", "TabArena-v0.1_tasks_metadata.csv")
OUT_PATH = os.path.join(HERE, "..", "results", "real_tabarena_results.csv")
WORKER = os.path.join(HERE, "tabarena_worker.py")
PYTHON = sys.executable

meta = pd.read_csv(META_CSV)
meta = meta[meta.problem_type.isin(["binary", "multiclass"])]
tasks = (meta.groupby("dataset_name")
         .agg(task_id=("task_id_str", "first"), problem_type=("problem_type", "first"),
              n_train=("num_instances_train", "first"), n_feat=("num_features", "first"))
         .sort_values("n_train"))

# Bioresponse/hiva_agnostic (1776/1617 features) caused a hard CUDA crash before
# subprocess isolation existed - everything else tops out at 212 features. Kept
# excluded since they were never actually run successfully even once; revisit
# now that isolation exists, if wanted, as a deliberate separate decision.
HIGH_FEATURE_RISK = {"Bioresponse", "hiva_agnostic"}
skipped = [t for t in tasks.index if t in HIGH_FEATURE_RISK]
tasks = tasks[~tasks.index.isin(HIGH_FEATURE_RISK)]
if skipped:
    print(f"EXCLUDED (extreme feature count, never successfully run): {skipped}", flush=True)

print(f"{len(tasks)} classification tasks, subprocess-isolated per dataset", flush=True)

for name, row in tasks.iterrows():
    print(f"=== {name} (n_train={row.n_train:.0f}, n_feat={row.n_feat}) ===", flush=True)
    result = subprocess.run(
        [PYTHON, WORKER, name, str(int(row.task_id)), row.problem_type, OUT_PATH, META_CSV],
        cwd=HERE,
    )
    if result.returncode != 0:
        print(f"*** {name} worker exited {result.returncode} - contained to this dataset, continuing ***", flush=True)

print("orchestrator done", flush=True)
