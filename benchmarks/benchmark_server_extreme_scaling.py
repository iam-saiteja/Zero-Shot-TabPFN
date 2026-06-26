import time
import torch
import psutil
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd

# Append workspace path to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- COMPATIBILITY PATCHES (Applied when running in trial mode) ---
def apply_patches():
    import typing
    import torch.nn.modules.transformer
    torch.nn.modules.transformer.Optional = typing.Optional

    import sklearn.utils.validation
    import sklearn.utils
    original_check_X_y = sklearn.utils.validation.check_X_y
    original_check_array = sklearn.utils.validation.check_array

    def patched_check_X_y(*args, **kwargs):
        kwargs.pop('force_all_finite', None)
        return original_check_X_y(*args, **kwargs)

    def patched_check_array(*args, **kwargs):
        kwargs.pop('force_all_finite', None)
        return original_check_array(*args, **kwargs)

    sklearn.utils.validation.check_X_y = patched_check_X_y
    sklearn.utils.validation.check_array = patched_check_array
    sklearn.utils.check_X_y = patched_check_X_y
    sklearn.utils.check_array = patched_check_array

# --- SUBPROCESS INDIVIDUAL TRIAL EXECUTION ---
def run_single_trial(model_name, N):
    apply_patches()
    from tabpfn import TabPFNClassifier
    from nsatabpfn.wrapper import restore_vanilla_tabpfn, inject_nsatabpfn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_name == 'Vanilla TabPFN':
        restore_vanilla_tabpfn()
    else:
        inject_nsatabpfn(num_prototypes=128)

    def get_vram_usage():
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return psutil.Process().memory_info().rss / (1024 ** 2)

    def generate_synthetic_data(n_samples, n_features=20):
        X = torch.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).long()
        return X.numpy(), y.numpy()

    # Reset peak memory stats before allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

    X_train, y_train = generate_synthetic_data(N)
    X_test, y_test = generate_synthetic_data(100)

    clf = TabPFNClassifier(device=device, N_ensemble_configurations=1)
    
    start_mem = get_vram_usage()
    start_time = time.time()
    
    clf.fit(X_train, y_train, overwrite_warning=True)
    probs = clf.predict_proba(X_test)
    
    exec_time = time.time() - start_time
    peak_mem = max(0.0, get_vram_usage() - start_mem)
    
    preds = probs.argmax(axis=1)
    acc = (preds == y_test).mean()

    # Print results in a standard parseable format for the parent process
    print(f"TRIAL_RESULT: time={exec_time:.4f}, mem={peak_mem:.4f}, acc={acc:.4f}")

# --- PARENT COORDINATOR ---
def run_coordinator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Multi-Process Extreme Scaling Benchmark on ({device})")
    
    results = []
    
    # We benchmark NSA-TabPFN first so that we collect its full million-row curve
    # even if Vanilla TabPFN causes a driver hang or crash later.
    for model_name in ['NSA-TabPFN', 'Vanilla TabPFN']:
        print(f"\n=== Benchmarking Model: {model_name} ===")
        
        N = 1024
        while True:
            print(f"Spawning trial for N = {N:,} rows...")
            
            # Call self as a subprocess for isolation
            cmd = [
                sys.executable, __file__,
                "--run-trial",
                "--model", model_name,
                "--n", str(N)
            ]
            
            try:
                # Set a generous timeout of 10 minutes per trial
                res = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=600
                )
                
                # Check if the subprocess crashed (non-zero exit code)
                if res.returncode != 0:
                    err_msg = res.stderr.strip() if res.stderr else f"Exit code {res.returncode}"
                    print(f"-> Model {model_name} CRASHED/OOM on N = {N:,}: {err_msg}")
                    results.append({
                        'Model': model_name, 'N': N, 
                        'Latency (s)': np.nan, 'Peak Memory (MB)': np.nan,
                        'Accuracy': np.nan, 'Status': f'Crashed/OOM ({err_msg})'
                    })
                    break
                
                # Parse stdout for TRIAL_RESULT
                stdout = res.stdout
                result_line = [line for line in stdout.split('\n') if "TRIAL_RESULT:" in line]
                
                if result_line:
                    # Extract metric values
                    parts = result_line[0].replace("TRIAL_RESULT:", "").strip().split(",")
                    metrics = {}
                    for p in parts:
                        k, v = p.split("=")
                        metrics[k.strip()] = float(v.strip())
                        
                    print(f"-> Success: {metrics['time']:.2f}s, Peak VRAM: {metrics['mem']:.2f} MB, Acc: {metrics['acc']:.4f}")
                    results.append({
                        'Model': model_name, 'N': N, 
                        'Latency (s)': metrics['time'], 'Peak Memory (MB)': metrics['mem'],
                        'Accuracy': metrics['acc'], 'Status': 'Success'
                    })
                    
                    # Double the size
                    N *= 2
                else:
                    # Subprocess finished with 0 but did not print result (unhandled error or silent exit)
                    print(f"-> Model {model_name} finished silently without results on N = {N:,}")
                    results.append({
                        'Model': model_name, 'N': N, 
                        'Latency (s)': np.nan, 'Peak Memory (MB)': np.nan,
                        'Accuracy': np.nan, 'Status': 'Silent Exit'
                    })
                    break
                    
            except subprocess.TimeoutExpired:
                print(f"-> Model {model_name} TIMED OUT on N = {N:,} (exceeded 10 minutes).")
                results.append({
                    'Model': model_name, 'N': N, 
                    'Latency (s)': np.nan, 'Peak Memory (MB)': np.nan,
                    'Accuracy': np.nan, 'Status': 'Timeout'
                })
                break
            except Exception as e:
                print(f"-> Unexpected error coordinating N = {N:,}: {e}")
                break
                
    # Save raw data
    if results:
        df = pd.DataFrame(results)
        df.to_csv("server_extreme_scaling_results.csv", index=False)
        print("\nResults saved to server_extreme_scaling_results.csv")
    
        # Plotting
        os.makedirs("assets", exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        # Time comparison
        plt.figure(figsize=(10, 6))
        df_plot_time = df.dropna(subset=['Latency (s)'])
        sns.lineplot(data=df_plot_time, x='N', y='Latency (s)', hue='Model', marker='o', linewidth=2.5)
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.xlabel('Sequence Length N (Rows)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Dynamic Inference Latency Scaling Limit')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig('assets/server_extreme_scaling_time.png', dpi=300)
        plt.close()
        
        # Memory comparison
        plt.figure(figsize=(10, 6))
        df_plot_mem = df.dropna(subset=['Peak Memory (MB)'])
        sns.lineplot(data=df_plot_mem, x='N', y='Peak Memory (MB)', hue='Model', marker='s', linewidth=2.5, linestyle='--')
        plt.xscale('log', base=2)
        plt.xlabel('Sequence Length N (Rows)')
        plt.ylabel('Peak VRAM Allocation (MB)')
        plt.title('Dynamic Peak VRAM Allocation Scaling Limit')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig('assets/server_extreme_scaling_memory.png', dpi=300)
        plt.close()
        
        print("Generated scaling charts in assets/")
    else:
        print("\nNo successful runs to plot.")

# --- MAIN DISPATCHER ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-trial", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()

    if args.run_trial:
        run_single_trial(args.model, args.n)
    else:
        run_coordinator()

if __name__ == "__main__":
    main()
