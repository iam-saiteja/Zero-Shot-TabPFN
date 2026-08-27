"""
ZS-ISAB AutoGluon AbstractModel wrapper for TabArena submission.

This file wraps TabPFNZSISABModel in the AutoGluon AbstractModel interface
so it can be benchmarked inside the TabArena framework and submitted to
the live leaderboard at https://tabarena.ai/

Usage inside TabArena:
    from tabarena_zsisab import ZSISABModel
    model = ZSISABModel()

Or register it and run via the TabArena experiment runner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator


class ZSISABModel(AbstractModel):
    """
    AutoGluon wrapper for Zero-Shot ISAB (ZS-ISAB).

    ZS-ISAB is a drop-in attention wrapper that extends pre-trained TabPFN
    to linear O(NM) memory complexity without retraining the frozen model weights.
    It handles datasets with up to 1.25M rows on a single consumer GPU (RTX 3090 Ti).

    Paper: "Zero-Shot ISAB: Linear-Complexity Inducing Point Attention
            for Frozen Tabular Transformers" (submitted to TMLR 2026)
    Code:  https://github.com/iam-saiteja/Zero-Shot-TabPFN

    Hyperparameters
    ---------------
    n_prototypes : int, default 512
        Number of anchor/inducing points M selected from the training set.
    chunk_size : int, default 16384
        Block size B for streaming data through the online softmax accumulator.
    seed : int, default 42
        Seed for the seeded permutation used in anchor selection.
    """

    ag_key = "ZSISAB"
    ag_name = "ZS-ISAB"
    ag_priority = 105  # Just below TabPFN (110) so it appears nearby on leaderboard

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._model = None

    # ------------------------------------------------------------------
    # Hyperparameter search space (for TabArena's HPO sweeps)
    # ------------------------------------------------------------------
    @classmethod
    def _get_default_ag_args_ensemble(cls) -> dict:
        return {"fold_fitting_strategy": "sequential_local"}

    def _get_default_hyperparameters(self) -> dict:
        return {
            "n_prototypes": 512,
            "chunk_size": 16384,
            "seed": 42,
        }

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator is not None and self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).astype(np.float32)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ) -> None:
        from tabpfn_isab import TabPFNZSISABModel  # import from your repo

        params = self._get_model_params()
        n_prototypes = params.get("n_prototypes", 512)
        chunk_size = params.get("chunk_size", 16384)
        seed = params.get("seed", 42)

        X_processed = self._preprocess(X, is_train=True)

        self._model = TabPFNZSISABModel(
            n_prototypes=n_prototypes,
            chunk_size=chunk_size,
            seed=seed,
        )
        self._model.fit(X_processed, y.to_numpy())

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_processed = self._preprocess(X, is_train=False)
        proba = self._model.predict_proba(X_processed)
        return proba

    # ------------------------------------------------------------------
    # Memory estimate (helps TabArena scheduler avoid OOM)
    # ------------------------------------------------------------------
    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        # ZS-ISAB peak VRAM is O(chunk_size * embedding_dim) -- very flat
        # Conservative estimate: 4 GB for the TabPFN checkpoint + 2 GB buffer
        return 6 * 1024 ** 3  # 6 GB in bytes
