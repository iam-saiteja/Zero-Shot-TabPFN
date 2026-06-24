from __future__ import annotations

import os
import sys
import time
import csv
import torch
import numpy as np

# Set token before imports
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_WDvw1MHEYQRQz8NKJBMqEFoink8X-sagyYRMKWM8Vo4"

# 1. Monkey patch AlongColumnAttention in tabpfn architectures before importing TabPFN
import tabpfn.architectures.tabpfn_v2 as tabpfn_v2
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
import tabpfn.architectures.tabpfn_v2_6 as tabpfn_v2_6
from tabpfn_msa import AlongColumnAttentionMSA

tabpfn_v2.AlongColumnAttention = AlongColumnAttentionMSA
tabpfn_v2_5.AlongColumnAttention = AlongColumnAttentionMSA
tabpfn_v2_6.AlongColumnAttention = AlongColumnAttentionMSA

# Wrap the original load_state_dict to make it non-strict but preserve key translations
original_load_v2 = tabpfn_v2.TabPFNV2.load_state_dict
tabpfn_v2.TabPFNV2.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2(self, sd, strict=False, assign=assign)

original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)

original_load_v2_6 = tabpfn_v2_6.TabPFNV2p6.load_state_dict
tabpfn_v2_6.TabPFNV2p6.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_6(self, sd, strict=False, assign=assign)

from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.data_util import ClassifierBatch
from msa_pytorch import MiniMaxSparseAttentionPyTorch
from data_generator import generate_scm_dataset
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

class FinetunedTabPFNClassifierWithKL(FinetunedTabPFNClassifier):
    def __init__(
        self,
        *,
        device: str = "cuda",
        epochs: int = 30,
        time_limit: int | None = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        validation_split_ratio: float = 0.1,
        n_finetune_ctx_plus_query_samples: int = 10_000,
        finetune_ctx_query_split_ratio: float = 0.2,
        n_inference_subsample_samples: int = 50_000,
        random_state: int = 0,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        min_delta: float = 1e-4,
        grad_clip_value: float | None = 1.0,
        use_lr_scheduler: bool = True,
        lr_warmup_only: bool = False,
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_final_inference: int = 8,
        use_activation_checkpointing: bool = True,
        save_checkpoint_interval: int | None = 10,
        use_fixed_preprocessing_seed: bool = True,
        experiment_logger: any = None,
        extra_classifier_kwargs: dict[str, any] | None = None,
        eval_metric: any = None,
        kl_weight: float = 1.0,
    ):
        super().__init__(
            device=device,
            epochs=epochs,
            time_limit=time_limit,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_split_ratio=validation_split_ratio,
            n_finetune_ctx_plus_query_samples=n_finetune_ctx_plus_query_samples,
            finetune_ctx_query_split_ratio=finetune_ctx_query_split_ratio,
            n_inference_subsample_samples=n_inference_subsample_samples,
            random_state=random_state,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            grad_clip_value=grad_clip_value,
            use_lr_scheduler=use_lr_scheduler,
            lr_warmup_only=lr_warmup_only,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_final_inference=n_estimators_final_inference,
            use_activation_checkpointing=use_activation_checkpointing,
            save_checkpoint_interval=save_checkpoint_interval,
            use_fixed_preprocessing_seed=use_fixed_preprocessing_seed,
            experiment_logger=experiment_logger,
            extra_classifier_kwargs=extra_classifier_kwargs,
            eval_metric=eval_metric,
        )
        self.kl_weight = kl_weight
        self.last_ce_loss = 0.0
        self.last_kl_loss = 0.0

    def _forward_with_loss(self, batch: ClassifierBatch) -> torch.Tensor:
        # Explicitly set the model to training mode so that self.training is True in custom layers
        self.finetuned_estimator_.model_.train()

        # 1. Compute standard task cross-entropy loss
        ce_loss = super()._forward_with_loss(batch)
        self.last_ce_loss = ce_loss.item()

        # 2. Gather KL losses from all MSA modules
        kl_losses = []
        found_msa = 0
        training_states = []
        for name, module in self.finetuned_estimator_.model_.named_modules():
            if isinstance(module, MiniMaxSparseAttentionPyTorch):
                found_msa += 1
                training_states.append(module.training)
                if module.current_kl_loss is not None:
                    kl_losses.append(module.current_kl_loss)
        
        print(f"   [Debug] Found {found_msa} MiniMaxSparseAttentionPyTorch modules. Training states: {training_states}")
        
        if kl_losses:
            kl_loss = torch.stack(kl_losses).mean()
            self.last_kl_loss = kl_loss.item()
            total_loss = ce_loss + self.kl_weight * kl_loss
        else:
            self.last_kl_loss = 0.0
            total_loss = ce_loss

        # Collect index contrasts
        contrasts = []
        for module in self.finetuned_estimator_.model_.modules():
            if isinstance(module, MiniMaxSparseAttentionPyTorch) and hasattr(module, 'indexer_log') and module.indexer_log:
                contrasts.append(module.indexer_log[-1])
        
        avg_contrast = sum(contrasts) / len(contrasts) if contrasts else 0.0
        
        print(f"   [Step] CE Loss: {self.last_ce_loss:.4f} | KL Loss: {self.last_kl_loss:.4f} | Total Loss: {total_loss.item():.4f} | Index Contrast: {avg_contrast:.4f}")
        return total_loss

def main():
    print("---------------------------------------------")
    print("MSA-TabPFN Adaptation: Pretraining/Finetuning Run")
    print("---------------------------------------------")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Generate synthetic SCM classification dataset
    print("Generating synthetic classification dataset using SCM...")
    X, y = generate_scm_dataset(
        num_samples=1500,
        num_features=10,
        task_type="classification",
        num_classes=2,
        random_state=42
    )
    print(f"Dataset generated. X shape: {X.shape}, y shape: {y.shape}")

    # Split into train and validation sets
    X_train, X_val = X[:1200], X[1200:]
    y_train, y_val = y[:1200], y[1200:]

    # Configure the CSV logger
    csv_logger = CustomCSVLogger("msa_pretraining_metrics.csv")

    # Set up the classifier with KL loss
    # Using small settings to respect local RTX 3050 VRAM constraints
    print("Initializing FinetunedTabPFNClassifierWithKL...")
    clf = FinetunedTabPFNClassifierWithKL(
        device=device,
        epochs=5,
        learning_rate=1e-5,
        kl_weight=1.0,
        validation_split_ratio=0.0,  # We pass validation data explicitly
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

    # We patch AlongColumnAttentionMSA init parameters
    original_init = AlongColumnAttentionMSA.__init__
    
    def patched_init(self, *args, **kwargs):
        kwargs["blocking_strategy"] = "similarity"
        kwargs["topk"] = 4
        kwargs["blk_kv"] = 32
        original_init(self, *args, **kwargs)
        
    AlongColumnAttentionMSA.__init__ = patched_init

    try:
        print("Starting training fit loop...")
        from pathlib import Path
        os.makedirs("msa_checkpoints", exist_ok=True)
        clf.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=Path(os.path.abspath("msa_checkpoints"))
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original init
        AlongColumnAttentionMSA.__init__ = original_init
        csv_logger.finish()

if __name__ == "__main__":
    main()
