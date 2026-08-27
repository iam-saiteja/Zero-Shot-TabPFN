import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Set style
plt.style.use('dark_background')
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    "axes.facecolor": "#111111",
    "figure.facecolor": "#111111",
    "axes.edgecolor": "#555555",
    "grid.color": "#333333",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white"
})

def clean_name(ds_path):
    if '__' in ds_path:
        return ds_path.split('__')[1]
    return ds_path.split('/')[-1]

def parse_single_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        model_name = data.get('model', {}).get('name', 'Unknown')
        dataset_name = data.get('dataset', {}).get('name', 'Unknown')
        scorers = data.get('scorers', {}).get('test', {})
        
        auc = None
        acc = None
        if 'AUC' in scorers:
            mean_auc = np.nanmean(scorers['AUC'])
            if not np.isnan(mean_auc):
                auc = mean_auc
        if 'Accuracy' in scorers:
            mean_acc = np.nanmean(scorers['Accuracy'])
            if not np.isnan(mean_acc):
                acc = mean_acc
        return dataset_name, model_name, auc, acc
    except Exception:
        return None, None, None, None

def main():
    print("=== DYNAMICALLY COMPUTING BENCHMARK METRICS FROM RAW RESULTS ===")
    
    base_dir = r'C:\Users\iamsa\Documents\ISAB-r\tabzilla\TabZilla\results'
    json_files = []
    
    start_time = time.time()
    for root, _, filenames in os.walk(base_dir):
        for name in filenames:
            if name.endswith('.json'):
                json_files.append(os.path.join(root, name))
                
    print(f"Found {len(json_files)} JSON result files in {time.time() - start_time:.2f}s. Parsing...")

    results_auc = defaultdict(lambda: defaultdict(list))
    results_acc = defaultdict(lambda: defaultdict(list))

    parse_start = time.time()
    with ThreadPoolExecutor(max_workers=16) as executor:
        for ds, model, auc, acc in executor.map(parse_single_file, json_files, chunksize=100):
            if ds and model:
                if auc is not None:
                    results_auc[ds][model].append(auc)
                if acc is not None:
                    results_acc[ds][model].append(acc)

    print(f"Parsed all JSON files in {time.time() - parse_start:.2f}s.")

    best_auc_per_ds = defaultdict(dict)
    best_acc_per_ds = defaultdict(dict)

    for ds, models in results_auc.items():
        for model, aucs in models.items():
            if len(aucs) > 0:
                best_auc_per_ds[ds][model] = np.mean(aucs)

    for ds, models in results_acc.items():
        for model, accs in models.items():
            if len(accs) > 0:
                best_acc_per_ds[ds][model] = np.mean(accs)

    datasets = list(best_acc_per_ds.keys())
    num_datasets = len(datasets)
    print(f"Successfully aggregated metrics across {num_datasets} distinct datasets.")

    model_mean_acc = {}
    model_mean_auc = {}

    acc_by_model = defaultdict(list)
    auc_by_model = defaultdict(list)

    for ds in datasets:
        for model, val in best_acc_per_ds[ds].items():
            acc_by_model[model].append(val)
        for model, val in best_auc_per_ds[ds].items():
            auc_by_model[model].append(val)

    for model, vals in acc_by_model.items():
        model_mean_acc[model] = float(np.mean(vals))
    for model, vals in auc_by_model.items():
        model_mean_auc[model] = float(np.mean(vals))

    model_name_map = {
        'TabPFNZSISABModel': 'ZS-ISAB',
        'XGBoost': 'XGBoost',
        'CatBoost': 'CatBoost',
        'LightGBM': 'LightGBM',
        'RandomForest': 'RandomForest',
        'LinearModel': 'LinearModel'
    }

    display_acc = {}
    display_auc = {}
    for orig_name, mean_val in model_mean_acc.items():
        mapped_name = model_name_map.get(orig_name, orig_name)
        display_acc[mapped_name] = mean_val
    for orig_name, mean_val in model_mean_auc.items():
        mapped_name = model_name_map.get(orig_name, orig_name)
        display_auc[mapped_name] = mean_val

    sorted_acc_models = sorted(display_acc.keys(), key=lambda m: display_acc[m], reverse=True)
    sorted_acc_values = [display_acc[m] for m in sorted_acc_models]

    print("\n--- DYNAMICALLY COMPUTED ACCURACY LEADERBOARD ---")
    for m in sorted_acc_models:
        print(f"  {m:<18}: {display_acc[m]:.4f} (on {len(acc_by_model.get(m, acc_by_model.get('TabPFNZSISABModel' if m == 'ZS-ISAB' else m)))} datasets)")

    print("\n--- DYNAMICALLY COMPUTED AUC LEADERBOARD ---")
    sorted_auc_models = sorted(display_auc.keys(), key=lambda m: display_auc[m], reverse=True)
    for m in sorted_auc_models:
        print(f"  {m:<18}: {display_auc[m]:.4f}")

    os.makedirs('paper/assets', exist_ok=True)
    os.makedirs('assets', exist_ok=True)

    # --- PLOT 1: Accuracy vs Baselines Bar Chart (Dynamic Data) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = []
    for m in sorted_acc_models:
        if m == 'ZS-ISAB':
            colors.append('#00FFCC')
        elif m in ['XGBoost', 'CatBoost']:
            colors.append('#FF3366')
        else:
            colors.append('#555555')

    bars = ax.bar(sorted_acc_models, sorted_acc_values, color=colors, edgecolor='white', linewidth=1)

    min_acc = min(sorted_acc_values)
    max_acc = max(sorted_acc_values)
    ax.set_ylim(max(0.65, min_acc - 0.05), min(1.0, max_acc + 0.05))
    ax.set_ylabel(f'Mean Test Accuracy ({num_datasets} datasets)')
    ax.set_title(f'ZS-ISAB (Zero-Shot) vs HPO-Tuned Traditional Baselines ({num_datasets} Datasets)')
    ax.grid(axis='y', linestyle='--', alpha=0.2)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}', ha='center', va='bottom', color='white', weight='bold')

    plt.tight_layout()
    plt.savefig('paper/assets/bar_accuracy.png', dpi=300, transparent=True)
    plt.savefig('assets/bar_accuracy.png', dpi=300, transparent=True)
    plt.close()
    print("Saved bar_accuracy.png dynamically.")

    # --- PLOT 2 & 3: TFM Foundation Model Leaderboard (Scatter Plots) ---
    if os.path.exists('benchmarks/tfm_leaderboard.json'):
        with open('benchmarks/tfm_leaderboard.json', 'r') as f:
            tfm_data = json.load(f)

        by_dataset = defaultdict(dict)
        for row in tfm_data:
            ds = row['dataset']
            model = row['model']
            by_dataset[ds][model] = row

        complete = {ds: d for ds, d in by_dataset.items() if 'ZS-ISAB' in d and 'TabICL' in d and 'TabDPT' in d}

        fig, ax = plt.subplots(figsize=(10, 7))
        zs_train, zs_test, zs_names = [], [], []
        icl_train, icl_test, icl_names = [], [], []
        dpt_train, dpt_test, dpt_names = [], [], []

        for ds, d in complete.items():
            name = clean_name(ds)
            zs_train.append(d['ZS-ISAB']['train_time'])
            zs_test.append(d['ZS-ISAB']['test_time'])
            zs_names.append(name)
            
            icl_train.append(d['TabICL']['train_time'])
            icl_test.append(d['TabICL']['test_time'])
            icl_names.append(name)
            
            dpt_train.append(d['TabDPT']['train_time'])
            dpt_test.append(d['TabDPT']['test_time'])
            dpt_names.append(name)

        ax.scatter(zs_train, zs_test, c='#00FFCC', alpha=0.7, edgecolors='none', label='ZS-ISAB')
        ax.scatter(icl_train, icl_test, c='#FF3366', alpha=0.7, edgecolors='none', label='TabICL')
        ax.scatter(dpt_train, dpt_test, c='#FFCC00', alpha=0.7, edgecolors='none', label='TabDPT')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Train Time (seconds)')
        ax.set_ylabel('Test Time (seconds)')
        ax.set_title('Inference Trade-offs: Train Time vs Test Time')
        ax.grid(True, which="both", ls="--", alpha=0.2)
        ax.legend()

        for i in np.argsort(icl_test)[-5:]:
            ax.text(icl_train[i]*1.2, icl_test[i]*1.1, icl_names[i], color='#FF3366', fontsize=9)
        for i in np.argsort(zs_train)[-2:]:
            ax.text(zs_train[i]*1.1, zs_test[i]*0.9, zs_names[i], color='#00FFCC', fontsize=9)

        plt.tight_layout()
        plt.savefig('paper/assets/scatter_time.png', dpi=300, transparent=True)
        plt.savefig('assets/scatter_time.png', dpi=300, transparent=True)
        plt.close()
        print("Saved scatter_time.png dynamically.")

        fig, ax = plt.subplots(figsize=(8, 8))
        zs_aucs = [d['ZS-ISAB']['metric_val'] for d in complete.values()]
        icl_aucs = [d['TabICL']['metric_val'] for d in complete.values()]
        ds_names_clean = [clean_name(ds) for ds in complete.keys()]

        ax.scatter(icl_aucs, zs_aucs, c='#00FFCC', alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.plot([0, 1], [0, 1], 'w--', alpha=0.5)
        ax.set_xlim(0.4, 1.02)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel('TabICL AUC')
        ax.set_ylabel('ZS-ISAB AUC')
        ax.set_title('Zero-Shot AUC Comparison: ZS-ISAB vs TabICL')
        ax.grid(True, ls="--", alpha=0.2)

        for i, (x, y, name) in enumerate(zip(icl_aucs, zs_aucs, ds_names_clean)):
            if abs(x - y) > 0.08:
                offset_y = 0.015 if y > x else -0.015
                ax.text(x, y + offset_y, name, color='white', fontsize=9, ha='center')

        plt.tight_layout()
        plt.savefig('paper/assets/scatter_auc.png', dpi=300, transparent=True)
        plt.savefig('assets/scatter_auc.png', dpi=300, transparent=True)
        plt.close()
        print("Saved scatter_auc.png dynamically.")

    print("\nAll dynamic plot generation tasks completed successfully!")

if __name__ == '__main__':
    main()
