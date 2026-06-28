import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def generate_plots():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "assets" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    comp_df = pd.read_csv(data_dir / "comparison_results_30.csv")
    
    # 1. VRAM Usage Comparison (Log Scale)
    plt.figure(figsize=(10, 6))
    comp_df_vram = comp_df.dropna(subset=['Vanilla_VRAM', 'ZS_ISAB_VRAM']).sort_values('Rows')
    
    x = np.arange(len(comp_df_vram))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, comp_df_vram['Vanilla_VRAM'], width, label='Vanilla TabPFN', color='#E63946')
    rects2 = ax.bar(x + width/2, comp_df_vram['ZS_ISAB_VRAM'], width, label='ZS-ISAB', color='#2A9D8F')
    
    ax.set_ylabel('Peak VRAM Usage (MB)')
    ax.set_title('VRAM Footprint Comparison (Vanilla vs ZS-ISAB)')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df_vram['Dataset_Name'], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'vram_comparison.png', dpi=300)
    plt.close()
    
    # 2. Execution Time Speedup
    plt.figure(figsize=(10, 6))
    comp_df_time = comp_df.dropna(subset=['Vanilla_Time', 'ZS_ISAB_Time']).sort_values('Rows')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, comp_df_time['Vanilla_Time'], width, label='Vanilla TabPFN', color='#E63946')
    rects2 = ax.bar(x + width/2, comp_df_time['ZS_ISAB_Time'], width, label='ZS-ISAB', color='#2A9D8F')
    
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Inference Speed Comparison (Vanilla vs ZS-ISAB)')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df_time['Dataset_Name'], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'time_comparison.png', dpi=300)
    plt.close()

    # 3. Accuracy Pareto Front (ROC-AUC vs Time)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=comp_df, x='ZS_ISAB_Time', y='ZS_ISAB_ROC', size='Rows', sizes=(50, 400), alpha=0.7, color='#2A9D8F')
    plt.title('ZS-ISAB Efficiency (ROC-AUC vs Time)')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('ROC-AUC Score')
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_efficiency.png', dpi=300)
    plt.close()
    
    print("Successfully generated all plots in assets/plots/")

if __name__ == "__main__":
    generate_plots()
