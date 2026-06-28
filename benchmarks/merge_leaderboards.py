import sys
import json
from pathlib import Path
import pandas as pd

def main():
    benchmarks_dir = Path(__file__).resolve().parent
    tabzilla_results_dir = benchmarks_dir.parent / "tabzilla" / "TabZilla" / "results"
    
    tfm_file = benchmarks_dir / "tfm_leaderboard.json"
    
    if not tfm_file.exists():
        print(f"Error: Could not find TFM results at {tfm_file}")
        return
        
    with open(tfm_file, "r") as f:
        tfm_results = json.load(f)
        
    df_tfm = pd.DataFrame(tfm_results)
    
    # We will aggregate the TabZilla JSONs 
    # For now, just a placeholder script to merge DataFrames
    print("Merge Script Initialized. Ready to combine ZS-ISAB and TFM results once the full sweeps are complete!")
    
    # Later: parse all tabzilla results, extract max AUC per model per dataset, concat with df_tfm, and sort by AUC.

if __name__ == "__main__":
    main()
