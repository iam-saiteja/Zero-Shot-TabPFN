from __future__ import annotations
import importlib

import tabpfn.layer as tabpfn_layer
from zsisab.engine import get_zsisab_encoder_forward

# Store the original forward function
if not hasattr(tabpfn_layer.TransformerEncoderLayer, "_original_forward"):
    tabpfn_layer.TransformerEncoderLayer._original_forward = tabpfn_layer.TransformerEncoderLayer.forward

def inject_zsisab_into_tabpfn(num_prototypes: int = 128, use_logit_scaling: bool = True, use_norm_alignment: bool = True, verbose: bool = False):
    """
    Monkey-patches TabPFN's TransformerEncoderLayer to intercept zero-shot evaluation
    and compress the context to O(N x M) using Zero-Shot ISAB.
    """
    original_fn = tabpfn_layer.TransformerEncoderLayer._original_forward
    patched_fn = get_zsisab_encoder_forward(
        original_fn, 
        num_prototypes=num_prototypes,
        use_logit_scaling=use_logit_scaling,
        use_norm_alignment=use_norm_alignment
    )
    
    tabpfn_layer.TransformerEncoderLayer.forward = patched_fn
    if verbose:
        print(f"✅ Zero-Shot ISAB successfully injected into TabPFN TransformerEncoderLayer with M={num_prototypes}.")

def restore_vanilla_tabpfn(verbose: bool = False):
    """
    Restores the original TabPFN TransformerEncoderLayer.
    """
    tabpfn_layer.TransformerEncoderLayer.forward = tabpfn_layer.TransformerEncoderLayer._original_forward
    if verbose:
        print("🔄 Restored vanilla TabPFN.")
