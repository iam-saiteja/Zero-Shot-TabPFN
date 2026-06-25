from __future__ import annotations

import os
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import csv
import torch
import numpy as np
from pathlib import Path

# Set token before imports
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_WDvw1MHEYQRQz8NKJBMqEFoink8X-sagyYRMKWM8Vo4"

# 1. Monkey patch AlongColumnAttention in tabpfn architectures using our wrapper
from zsisab.wrapper import inject_zsisab_into_tabpfn
inject_zsisab_into_tabpfn(num_prototypes=128)

from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.data_util import ClassifierBatch
from zsisab.data_generator import generate_scm_dataset
import tabpfn.finetuning.train_util as train_util
from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs
from tabpfn.architectures import ARCHITECTURES
from tabpfn.model_loading import _resolve_architecture_name

def patched_clone_model_for_evaluation(original_model, eval_init_args, model_class):
    import copy
    new_architecture_config = copy.deepcopy(original_model.configs_[0])
    new_inference_config = copy.deepcopy(original_model.inference_config_)
    
    # Safely duplicate model inner weight dict instead of deepcopying module object
    model_state = original_model.models_[0].state_dict()
    new_model_state_dict = copy.deepcopy(model_state)
    
    architecture_name = _resolve_architecture_name(new_architecture_config)
    architecture = ARCHITECTURES[architecture_name]
    eval_model_inner = architecture.get_architecture(new_architecture_config)
    eval_model_inner.load_state_dict(new_model_state_dict)
    
    if isinstance(original_model, FinetunedTabPFNClassifier) or model_class.__name__ == "TabPFNClassifier":
        model_spec_obj = ClassifierModelSpecs(
            model=eval_model_inner,
            architecture_config=new_architecture_config,
            inference_config=new_inference_config,
        )
    else:
        new_bar_dist = copy.deepcopy(original_model.znorm_space_bardist_)
        model_spec_obj = RegressorModelSpecs(
            model=eval_model_inner,
            architecture_config=new_architecture_config,
            inference_config=new_inference_config,
            norm_criterion=new_bar_dist,
        )
        
    eval_model = model_class(model_path=model_spec_obj, **eval_init_args)
    return eval_model

train_util.clone_model_for_evaluation = patched_clone_model_for_evaluation
import tabpfn.finetuning.finetuned_classifier as finetuned_classifier
import tabpfn.finetuning.finetuned_regressor as finetuned_regressor
finetuned_classifier.clone_model_for_evaluation = patched_clone_model_for_evaluation
finetuned_regressor.clone_model_for_evaluation = patched_clone_model_for_evaluation

class CustomCSVLogger:
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.step_file = open(csv_filepath.replace(".csv", "_step.csv"), "w", newline="")
        self.epoch_file = open(csv_filepath.replace(".csv", "_epoch.csv"), "w", newline="")
        self.step_writer = None
        self.epoch_writer = None

    def setup(self, config: dict[str, any]) -> None:
        print("Logger initialized.")

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        if self.step_writer is None:
            self.step_writer = csv.DictWriter(self.step_file, fieldnames=list(metrics.keys()) + ["step"])
            self.step_writer.writeheader()
        metrics["step"] = step
        self.step_writer.writerow(metrics)
        self.step_file.flush()

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        if self.epoch_writer is None:
            self.epoch_writer = csv.DictWriter(self.epoch_file, fieldnames=list(metrics.keys()) + ["step"])
            self.epoch_writer.writeheader()
        metrics["step"] = step
        self.epoch_writer.writerow(metrics)
        self.epoch_file.flush()

    def finish(self) -> None:
        self.step_file.close()
        self.epoch_file.close()
        print("Logger finished and CSV files saved.")

class FinetunedTabPFNClassifierZSISAB(FinetunedTabPFNClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_with_loss(self, batch: ClassifierBatch) -> torch.Tensor:
        self.finetuned_estimator_.model_.train()
        ce_loss = super()._forward_with_loss(batch)
        print(f"   [Step] CE Loss: {ce_loss.item():.4f}")
        return ce_loss

def main():
    print("---------------------------------------------")
    print("Zero-Shot ISAB TabPFN Adaptation: Finetuning Run")
    print("---------------------------------------------")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    print("Generating synthetic classification dataset using SCM...")
    X, y = generate_scm_dataset(
        num_samples=1500,
        num_features=10,
        task_type="classification",
        num_classes=2,
        random_state=42
    )
    print(f"Dataset generated. X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_val = X[:1200], X[1200:]
    y_train, y_val = y[:1200], y[1200:]

    csv_logger = CustomCSVLogger("zsisab_pretraining_metrics.csv")

    print("Initializing FinetunedTabPFNClassifierZSISAB...")
    clf = FinetunedTabPFNClassifierZSISAB(
        device=device,
        epochs=5,
        learning_rate=1e-5,
        validation_split_ratio=0.0,
        n_finetune_ctx_plus_query_samples=1000,
        finetune_ctx_query_split_ratio=0.2,
        n_inference_subsample_samples=1000,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        use_activation_checkpointing=True,
        early_stopping=False,
        experiment_logger=csv_logger,
        extra_classifier_kwargs={"n_estimators": 1}
    )

    try:
        print("Starting training fit loop...")
        os.makedirs("zsisab_checkpoints", exist_ok=True)
        clf.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=Path(os.path.abspath("zsisab_checkpoints"))
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        csv_logger.finish()

if __name__ == "__main__":
    main()
