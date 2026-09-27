"""
Official BeyondArena-Lite evaluation script for Zero-Shot ISAB (ZS-ISAB).
Uses BeyondArenaExperimentBundle and BeyondArenaContext matching examples/beyondarena/run_quickstart_beyondarena.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add tabarena package source directory
for candidate in [
    Path("/home/csd23rb8/tabarena/packages/tabarena/src"),
    Path.home() / "tabarena/packages/tabarena/src",
    Path("C:/Users/iamsa/Documents/tabarena/packages/tabarena/src"),
]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabarena.models.zsisab.info import zsisab_info


def main():
    print("=" * 70)
    print("RUNNING OFFICIAL BEYONDARENA-LITE BENCHMARK FOR ZS-ISAB")
    print("=" * 70)

    output_dir = Path(__file__).parent / "beyondarena_lite_results"

    experiments = BeyondArenaExperimentBundle(
        models=[
            (zsisab_info.search_space, 0),
        ],
    ).build_experiments()

    context = BeyondArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=str(output_dir / "cache"),
        subset=["lite", "classification"],
        build_kwargs={"dataset_names": ["credit_g", "bank_marketing", "amazon_employee_access"]},
        new_result_prefix="[New] ",
    )

    print("\nGenerating BeyondArena official leaderboard...")
    leaderboard = context.compare(output_dir=output_dir / "eval")

    print("\n" + "=" * 70)
    print("OFFICIAL BEYONDARENA-LITE LEADERBOARD OUTPUT:")
    print("=" * 70)
    print(leaderboard.to_markdown())


if __name__ == "__main__":
    main()
