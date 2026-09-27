"""
AutoGluon / TabArena unit test for ZS-ISAB model.
Compatible with both TabArena CI (using FitHelper) and standalone pytest / python execution.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tabarena_fithelper_test():
    """TabArena official FitHelper test used by Lennart & TabArena CI."""
    try:
        from tabarena.testing.fit_helper import FitHelper
        from model import ZSISABModel
        print("Running official TabArena FitHelper.verify_model...")
        FitHelper.verify_model(model_cls=ZSISABModel)
        print(" [OK] FitHelper.verify_model passed 100%!")
        return True
    except ImportError:
        print("TabArena / AutoGluon not in global environment. Running standalone verification...")
        return False


def run_standalone_test():
    """Standalone validation of model metadata, HPO spaces, and parameters."""
    from info import INFO
    from hpo import get_default_hyperparameters, get_hyperparameter_search_space
    
    print("\n[1/3] Verifying Model Metadata (info.py)...")
    assert INFO["name"] == "Zero-Shot ISAB"
    assert INFO["authors"] == ["Thanniru Sai Teja"]
    assert INFO["is_foundation_model"] is True
    assert INFO["is_zero_shot"] is True
    print(f" -> Info verified: {INFO['name']} by {INFO['authors']}")

    print("\n[2/3] Verifying HPO Defaults (hpo.py)...")
    defaults = get_default_hyperparameters()
    assert defaults["n_prototypes"] == 512
    assert defaults["chunk_size"] == 16384
    assert defaults["seed"] == 42
    print(f" -> HPO defaults verified: {defaults}")

    print("\n[3/3] Verifying Hyperparameter Search Space...")
    space = get_hyperparameter_search_space()
    assert "n_prototypes" in space
    assert "chunk_size" in space
    print(f" -> Search space verified with keys: {list(space.keys())}")

    print("\n" + "=" * 60)
    print("All Standalone Verification Checks Passed 100%!")
    print("=" * 60)


if __name__ == "__main__":
    if not run_tabarena_fithelper_test():
        run_standalone_test()
