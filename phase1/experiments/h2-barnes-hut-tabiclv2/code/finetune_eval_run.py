"""Orchestrator: evaluate a fine-tuned TabICLv2 checkpoint on the exact same
30-dataset real held-out TabArena set used for the Barnes-Hut baseline.
Subprocess-per-dataset, same isolation reasoning as real_tabarena_run.py.

Usage: python finetune_eval_run.py <ckpt_path> [out_csv]
"""
import os, subprocess, sys
import pandas as pd

HERE = os.path.dirname(__file__)
META_CSV = os.path.join(HERE, "..", "..", "..", "harness", "tabarena", "src", "tabarena",
                        "benchmark", "task", "metadata", "sources", "data", "TabArena-v0.1_tasks_metadata.csv")
WORKER = os.path.join(HERE, "finetune_eval_worker.py")
PYTHON = sys.executable

EVAL_DATASETS = ['Amazon_employee_access', 'Bank_Customer_Churn', 'E-CommereShippingData', 'Fitness_Club',
    'HR_Analytics_Job_Change_of_Data_Scientists', 'Is-this-a-good-customer', 'MIC', 'Marketing_Campaign',
    'NATICUSdroid', 'anneal', 'bank-marketing', 'blood-transfusion-service-center', 'churn',
    'coil2000_insurance_policies', 'credit-g', 'credit_card_clients_default', 'diabetes',
    'hazelnut-spread-contaminant-detection', 'heloc', 'in_vehicle_coupon_recommendation', 'jm1',
    'maternal_health_risk', 'online_shoppers_intention', 'polish_companies_bankruptcy', 'qsar-biodeg',
    'seismic-bumps', 'splice', 'students_dropout_and_academic_success', 'taiwanese_bankruptcy_prediction',
    'website_phishing']

if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results", "finetune_eval_results.csv")

    meta = pd.read_csv(META_CSV)
    tasks = (meta.groupby("dataset_name")
             .agg(task_id=("task_id_str", "first"), problem_type=("problem_type", "first"),
                  n_train=("num_instances_train", "first"))
             .loc[EVAL_DATASETS].sort_values("n_train"))

    print(f"{len(tasks)} held-out eval datasets, checkpoint={ckpt_path}", flush=True)
    for name, row in tasks.iterrows():
        print(f"=== {name} (n_train={row.n_train:.0f}) ===", flush=True)
        result = subprocess.run(
            [PYTHON, WORKER, name, str(int(row.task_id)), row.problem_type, out_path, META_CSV, ckpt_path],
            cwd=HERE,
        )
        if result.returncode != 0:
            print(f"*** {name} worker exited {result.returncode} - contained to this dataset, continuing ***",
                  flush=True)
    print("finetune eval done", flush=True)
