"""Merge our real-API results (real_api_default_path.py's output) with the
genuine cached official baselines and compute a real Elo number via
bencheval - the actual TabArena evaluation library, not a hand-rolled metric.
Restricted to the 13 classification datasets where we have real results
(single split each, like the reference "tiny" suite itself - fold=0 only).
"""
import os
import pandas as pd
from bencheval.evaluator import BenchmarkEvaluator

REF = os.path.join(os.path.dirname(__file__), "..", "..", "..", "harness",
                    "tabarena_tiny_results_reference", "eval", "results_per_split.csv")
OURS = os.path.join(os.path.dirname(__file__), "..", "results", "real_api_rows.csv")

ref = pd.read_csv(REF)
ours = pd.read_csv(OURS)
datasets = sorted(ours.dataset.unique())
ref = ref[ref.dataset.isin(datasets) & ref.problem_type.isin(["binary", "multiclass"])].copy()

new_rows = []
for _, r in ours.iterrows():
    for method, col in [("TabICLv2 (ours, vanilla)", "our_vanilla"), ("TabICLv2+barnes-hut (ours)", "our_bh")]:
        new_rows.append(dict(dataset=r.dataset, fold=0, method=method, metric_error=r[col],
                              metric="roc_auc" if r.ptype == "binary" else "log_loss", problem_type=r.ptype))
new_df = pd.DataFrame(new_rows)

combined = pd.concat([ref[["dataset", "fold", "method", "metric_error", "metric", "problem_type"]], new_df],
                      ignore_index=True)
combined["task"] = combined["dataset"]  # bencheval's default task_col name

n_methods = combined.method.nunique()
print(f"{len(datasets)} datasets, {n_methods} methods (incl. our 2), {len(combined)} rows")

ev = BenchmarkEvaluator(method_col="method", task_col="task", error_col="metric_error",
                         columns_to_agg_extra=None)
results_per_task = ev.compute_results_per_task(combined)
elo = ev.compute_elo(results_per_task, BOOTSTRAP_ROUNDS=200)
elo = elo.sort_values("elo", ascending=False)

print(f"\n=== Real Elo, {len(datasets)} real TabArena-tiny classification datasets, computed via bencheval ===")
print(elo.to_string())
rank = elo.index.get_loc("TabICLv2 (ours, vanilla)") + 1
rank_bh = elo.index.get_loc("TabICLv2+barnes-hut (ours)") + 1
print(f"\nour vanilla rank: {rank}/{len(elo)}   elo={elo.loc['TabICLv2 (ours, vanilla)', 'elo']}")
print(f"our barnes-hut rank: {rank_bh}/{len(elo)}   elo={elo.loc['TabICLv2+barnes-hut (ours)', 'elo']}")
official_row = "TABICLV2 (default)"
if official_row in elo.index:
    print(f"official cached TabICLv2 rank: {elo.index.get_loc(official_row)+1}/{len(elo)}   elo={elo.loc[official_row,'elo']}")

out_path = os.path.join(os.path.dirname(__file__), "..", "results", "real_elo.csv")
elo.to_csv(out_path)
print("saved ->", out_path)
