from __future__ import annotations
import importlib

import tabpfn.layer as tabpfn_layer
from nsatabpfn.engine import get_nsa_encoder_forward

# Store the original forward function
if not hasattr(tabpfn_layer.TransformerEncoderLayer, "_original_forward"):
    tabpfn_layer.TransformerEncoderLayer._original_forward = tabpfn_layer.TransformerEncoderLayer.forward

def inject_nsatabpfn(num_prototypes: int = 128, verbose: bool = False):
    """
    Monkey-patches TabPFN's TransformerEncoderLayer to intercept zero-shot evaluation
    and compress the context to O(N x M) using Zero-Shot Nystrom (NSA-TabPFN).
    """
    original_fn = tabpfn_layer.TransformerEncoderLayer._original_forward
    patched_fn = get_nsa_encoder_forward(
        original_fn, 
        num_prototypes=num_prototypes
    )
    
    tabpfn_layer.TransformerEncoderLayer.forward = patched_fn
    if verbose:
        print(f"[OK] Zero-Shot Nyström successfully injected into TabPFN TransformerEncoderLayer with M={num_prototypes}.")

def restore_vanilla_tabpfn(verbose: bool = False):
    """
    Restores the original TabPFN TransformerEncoderLayer.
    """
    tabpfn_layer.TransformerEncoderLayer.forward = tabpfn_layer.TransformerEncoderLayer._original_forward
    if verbose:
        print("[Restore] Restored vanilla TabPFN.")
