"""
Zero-Shot ISAB (ZS-ISAB) model wrapper for AutoGluon / TabArena.
"""
from __future__ import annotations

import typing
import torch.nn.modules.transformer
torch.nn.modules.transformer.Optional = typing.Optional

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator
from zsisab.wrapper import inject_zsisab


class ZSISABModel(AbstractModel):
    """
    AutoGluon model wrapper for Zero-Shot ISAB (ZS-ISAB).

    Paper: "Zero-Shot ISAB: Linear-Complexity Inducing Point Attention
            for Frozen Tabular Transformers"
    Code:  https://github.com/iam-saiteja/Zero-Shot-TabPFN
    """

    ag_key = "ZSISAB"
    ag_name = "ZS-ISAB"
    ag_priority = 105

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self.model = None

    @classmethod
    def _get_default_ag_args_ensemble(cls) -> dict:
        return {"fold_fitting_strategy": "sequential_local"}

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator is not None and self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: float = 0,
        time_limit: float | None = None,
        **kwargs,
    ) -> None:
        params = self._get_model_params()
        num_prototypes = params.get("num_prototypes", 512)
        chunk_size = params.get("chunk_size", 16384)
        n_ensemble = params.get("n_ensemble", 32)
        device = "cuda" if (num_gpus is not None and num_gpus > 0) else "cpu"

        # Inject ZS-ISAB online cross-attention into TabPFN
        inject_zsisab(num_prototypes=num_prototypes, chunk_size=chunk_size)

        self.model = TabPFNClassifier(device=device, N_ensemble_configurations=n_ensemble)
        
        X_processed = self._preprocess(X, is_train=True)
        self.model.fit(X_processed, y.to_numpy(), overwrite_warning=True)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_processed = self._preprocess(X, is_train=False)
        return self.model.predict_proba(X_processed)

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_processed = self._preprocess(X, is_train=False)
        return self.model.predict(X_processed)

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        # ZS-ISAB operates with a strictly bounded streaming footprint O(chunk_size * E)
        return 4 * 1024 ** 3  # 4 GB estimate
