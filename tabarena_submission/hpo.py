"""
Hyperparameter configuration and search spaces for ZS-ISAB on TabArena.
"""
from __future__ import annotations

from autogluon.common.space import Categorical, Int


def get_default_hyperparameters() -> dict:
    """Default zero-shot hyperparameter configuration."""
    return {
        "n_prototypes": 512,
        "chunk_size": 16384,
        "seed": 42,
    }


def get_hyperparameter_search_space() -> dict:
    """Hyperparameter search space for tuned TabArena runs."""
    return {
        "n_prototypes": Categorical(128, 256, 512, 1024),
        "chunk_size": Categorical(8192, 16384, 32768),
        "seed": Int(lower=0, upper=1000),
    }
