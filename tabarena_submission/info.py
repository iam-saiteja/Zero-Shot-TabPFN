"""
Metadata for Zero-Shot ISAB (ZS-ISAB) model in TabArena.
"""
from __future__ import annotations

INFO = {
    "name": "Zero-Shot ISAB",
    "short_name": "ZS-ISAB",
    "paper": "Zero-Shot ISAB: Linear-Complexity Inducing Point Attention for Frozen Tabular Transformers",
    "paper_url": "https://github.com/iam-saiteja/Zero-Shot-TabPFN",
    "code_url": "https://github.com/iam-saiteja/Zero-Shot-TabPFN",
    "authors": ["Thanniru Sai Teja"],
    "supported_problem_types": ["binary", "multiclass", "regression"],
    "is_foundation_model": True,
    "is_zero_shot": True,
    "context_limit": 1257500,
}
