"""Real Elo comparison, via bencheval (not hand-rolled), of the fine-tuned
TabICLv2 checkpoint against the existing vanilla/anchor/barnes_hut baseline
- same 30 real held-out TabArena datasets, same official splits, so this
isolates exactly what fine-tuning changed. Everything here traces to an
actual run: real_tabarena_results.csv (already-committed baseline) merged
with finetune_eval_results.csv (this run's output).

Column-labeling note: pivot/groupby output must be addressed by the actual
`method` string values, never by assumed column order - a hardcoded-order
bug here already caused one real, fully-backwards result earlier this
project (see docs/research-log.md, the pivot mislabeling entry).
"""
import os
import pandas as pd
from bencheval.evaluator import BenchmarkEvaluator

HERE = os.path.dirname(__file__)
BASELINE = os.path.join(HERE, "..", "results", "real_tabarena_results.csv")
FINETUNED = os.path.join(HERE, "..", "results", "finetune_eval_results.csv")

baseline = pd.read_csv(BASELINE)
finetuned = pd.read_csv(FINETUNED)

combined = pd.concat([baseline, finetuned], ignore_index=True)
combined["metric"] = combined["problem_type"].map({"binary": "roc_auc"}).fillna("log_loss")
combined["task"] = combined["dataset"]

print(f"{combined.dataset.nunique()} datasets, methods: {sorted(combined.method.unique())}, "
      f"{len(combined)} total rows")

ev = BenchmarkEvaluator(method_col="method", task_col="task", error_col="metric_error",
                         columns_to_agg_extra=None)
results_per_task = ev.compute_results_per_task(combined)
elo = ev.compute_elo(results_per_task, BOOTSTRAP_ROUNDS=200)
elo = elo.sort_values("elo", ascending=False)

print("\n=== Real Elo, 30 held-out TabArena datasets, computed via bencheval ===")
print(elo.to_string())

for m in combined.method.unique():
    if m in elo.index:
        print(f"{m}: rank {elo.index.get_loc(m)+1}/{len(elo)}, elo={elo.loc[m, 'elo']:.1f}")

vanilla_elo = elo.loc["TabICLv2 (ours, vanilla)", "elo"]
ft_elo = elo.loc["TabICLv2 (finetuned)", "elo"]
print(f"\nfine-tune delta: {ft_elo - vanilla_elo:+.1f} Elo vs vanilla baseline")

out_path = os.path.join(HERE, "..", "results", "finetune_elo.csv")
elo.to_csv(out_path)
print("saved ->", out_path)
