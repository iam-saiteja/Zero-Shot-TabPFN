"""
Official TabArena-Lite evaluation script for Zero-Shot ISAB (ZS-ISAB).
Runs TabArena Lite tasks (split 0) with automated leaderboard comparison.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

tabarena_src = Path("C:/Users/iamsa/Documents/tabarena/packages/tabarena/src")
if tabarena_src.exists() and str(tabarena_src) not in sys.path:
    sys.path.insert(0, str(tabarena_src))

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models.zsisab.info import zsisab_info


def main():
    parser = argparse.ArgumentParser(description="Run TabArena-Lite for ZS-ISAB")
    parser.add_argument(
        "--subset",
        default="tiny",
        choices=["tiny", "small", "all", "classification", "regression"],
        help="Dataset size/type subset (tiny=18 datasets <=2k rows, small=36 datasets <=10k rows, all=all 51 datasets)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"RUNNING OFFICIAL TABARENA-LITE BENCHMARK (Subset: {args.subset})")
    print("=" * 70)

    output_dir = Path(__file__).parent / "tabarena_lite_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(zsisab_info.search_space, 0)],
    ).build_experiments()

    context = TabArenaContext()

    subset_filter = ["lite", args.subset] if args.subset != "all" else ["lite"]
    jobs = context.build_jobs(
        experiments,
        task_subset={"subset": subset_filter},
    )
    print(f"Dispatched {len(jobs)} jobs for subset {subset_filter}...")

    context.run_jobs(
        jobs,
        expname=str(output_dir / "cache"),
        new_result_prefix="[New] ",
    )

    print("\nGenerating TabArena-Lite official leaderboard...")
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = context.compare(output_dir=eval_dir)
    website_lb = context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("\n" + "=" * 70)
    print("OFFICIAL TABARENA-LITE LEADERBOARD OUTPUT:")
    print("=" * 70)
    print(website_lb.to_markdown(index=False))

    with open(output_dir / "tabarena_lite_leaderboard.md", "w") as f:
        f.write(website_lb.to_markdown(index=False))
    website_lb.to_csv(output_dir / "tabarena_lite_leaderboard.csv", index=False)
    print(f"\nSaved leaderboard outputs to {output_dir}")


if __name__ == "__main__":
    main()
