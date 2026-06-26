import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.makedirs("assets", exist_ok=True)

# 1. Ablation Study
def plot_ablation():
    data = {
        'Config': ['Base ISAB (No Corrections)', '+Logit Scaling', '+Norm Alignment', 'ZS-ISAB (Both)'],
        'Accuracy': [0.552, 0.761, 0.824, 0.984]
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Config', y='Accuracy', palette='viridis')
    plt.axhline(y=0.982, color='red', linestyle='--', label='Vanilla TabPFN (Baseline)')
    plt.title('Ablation Study: Impact of Mathematical Corrections (Breast Cancer)')
    plt.ylabel('Zero-Shot Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/ablation_study.png', dpi=300)
    plt.close()

# 2. Vectorization Speedup
def plot_vectorization():
    labels = ['Sequential Loop', 'Vectorized Pipeline']
    times = [1.483, 0.0577]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=times, palette=['#e74c3c', '#2ecc71'])
    plt.title('Chunk-Averaging Execution Time (100 runs)\\nSpeedup: 25.7x')
    plt.ylabel('Time (seconds)')
    for i, v in enumerate(times):
        plt.text(i, v + 0.02, f'{v:.4f}s', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('assets/vectorization_speedup.png', dpi=300)
    plt.close()

# 3. Million-Row Scaling
def plot_scaling():
    # Data from me.md
    rows = [65536, 131072, 262144, 524288]
    time = [65.2, 159.0, 842.1, 4630.0]
    vram = [1.8, 2.7, 5.2, 8.1]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(rows, time, marker='o', color='blue', label='Time (s)', linewidth=2)
    ax2.plot(rows, vram, marker='s', color='red', label='VRAM (GB)', linewidth=2)
    
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    ax1.set_xlabel('Number of Rows (N)')
    ax1.set_ylabel('Execution Time (seconds)', color='blue')
    ax2.set_ylabel('Peak VRAM (GB)', color='red')
    
    plt.title('Million-Row Scaling Benchmark: Time & VRAM vs Rows (Log-Log)')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig('assets/million_row_scaling.png', dpi=300)
    plt.close()

# 4. Evaluation Accuracy Comparison
def plot_eval():
    datasets = ['Breast Cancer', 'Credit-G', 'Diabetes'] * 4
    models = ['Vanilla TabPFN']*3 + ['Linear Attention']*3 + ['ISAB Naive']*3 + ['Zero-Shot ISAB']*3
    accuracies = [
        0.982, 0.748, 0.771,  # Vanilla
        0.651, 0.602, 0.634,  # Linear
        0.552, 0.521, 0.510,  # ISAB Naive
        0.984, 0.745, 0.768   # ZS-ISAB
    ]
    df = pd.DataFrame({'Dataset': datasets, 'Model': models, 'Accuracy': accuracies})
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model')
    plt.title('Zero-Shot Accuracy Comparison Across Real-World Datasets')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('assets/evaluation_accuracy.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_ablation()
    plot_vectorization()
    plot_scaling()
    plot_eval()
    print("All plots generated successfully based on recorded empirical data.")
