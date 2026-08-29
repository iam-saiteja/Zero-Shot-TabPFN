"""
Official TabArena-Lite evaluation script for ZS-TabFM and ZS-ISAB.
Runs TabArena Lite tasks (split 0) with automated leaderboard comparison.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Auto-detect tabarena path if running locally
for p in [Path("C:/Users/iamsa/Documents/tabarena/packages/tabarena/src"), Path.home() / "tabarena/packages/tabarena/src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models._registry import discover_models


def main():
    parser = argparse.ArgumentParser(description="Run TabArena-Lite Benchmark")
    parser.add_argument(
        "--model",
        default="zstabfm",
        choices=["zstabfm", "zsisab"],
        help="Model to evaluate (zstabfm = ZS-TabFM foundation model, zsisab = ZS-ISAB)",
    )
    parser.add_argument(
        "--subset",
        default="tiny",
        choices=["tiny", "small", "all", "classification", "regression"],
        help="Dataset size/type subset (tiny=18 datasets <=2k rows, small=36 datasets <=10k rows, all=all 51 datasets)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"RUNNING OFFICIAL TABARENA-LITE BENCHMARK: {args.model.upper()} (Subset: {args.subset})")
    print("=" * 70)

    output_dir = Path(__file__).parent / f"tabarena_lite_{args.model}_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dict = discover_models()
    model_key = "ZS-TabFM" if args.model == "zstabfm" else "ZS-ISAB"
    
    if model_key not in models_dict:
        raise KeyError(f"Model {model_key} not found in TabArena registry! Available: {list(models_dict.keys())}")
    
    model_info = models_dict[model_key]

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(model_info.search_space, 0)],
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
    print(f"OFFICIAL TABARENA-LITE LEADERBOARD OUTPUT ({args.model.upper()}):")
    print("=" * 70)
    try:
        print(website_lb.to_markdown(index=False))
    except Exception:
        print(website_lb[["method", "elo", "rank", "winrate", "normalized-error"]].to_string(index=False))

    with open(output_dir / "tabarena_lite_leaderboard.md", "w", encoding="utf-8") as f:
        f.write(website_lb.to_markdown(index=False))
    website_lb.to_csv(output_dir / "tabarena_lite_leaderboard.csv", index=False)
    print(f"\nSaved leaderboard outputs to {output_dir}")


if __name__ == "__main__":
    main()
