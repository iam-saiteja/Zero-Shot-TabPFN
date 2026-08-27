"""
Run TabArena benchmark evaluation for ZS-ISAB on sample or full TabArena datasets.
"""
from __future__ import annotations

import os
import sys

# Add current folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabarena_zsisab import ZSISABModel
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

# Test datasets on TabArena-Lite (or specify full suite)
DATASETS = ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"]

def main():
    print("=" * 60)
    print("Starting TabArena Benchmark Evaluation for ZS-ISAB")
    print("=" * 60)

    # Instantiate experiment runner
    context = TabArenaContext()
    
    # Run evaluation
    print(f"Running ZS-ISAB across datasets: {DATASETS}")
    # Following TabArena execution patterns:
    # ExperimentBundle handles dataset loading, folds, metrics, and caching predictions
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[ZSISABModel],
        datasets=DATASETS,
        context=context,
    )
    results = bundle.run(debug_mode=True)
    print("\nBenchmark Finished! Results summary:")
    print(results)

if __name__ == "__main__":
    main()
