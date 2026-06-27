from __future__ import annotations
import importlib

import tabpfn.layer as tabpfn_layer
from zsisab.engine import get_zsisab_encoder_forward

# Store original forward fn safely
if not hasattr(tabpfn_layer.TransformerEncoderLayer, "_original_forward"):
    tabpfn_layer.TransformerEncoderLayer._original_forward = tabpfn_layer.TransformerEncoderLayer.forward

def inject_zsisab(num_prototypes: int = 128, chunk_size: int = 16384, verbose: bool = False):
    """
    Monkey-patches TabPFN's TransformerEncoderLayer to use Chunked ZS-ISAB.
    This guarantees peak VRAM is strictly bounded by the chunk_size and 
    allows evaluation on millions of rows natively inside the GPU.
    """
    original_fn = tabpfn_layer.TransformerEncoderLayer._original_forward
    patched_fn = get_zsisab_encoder_forward(
        original_fn, 
        num_prototypes=num_prototypes,
        chunk_size=chunk_size,
        verbose=verbose
    )
    
    tabpfn_layer.TransformerEncoderLayer.forward = patched_fn
    if verbose:
        print(f"✅ Chunked ZS-ISAB Injected (M={num_prototypes}, chunk_size={chunk_size})")

def restore_vanilla_tabpfn(verbose: bool = False):
    """
    Restores the original O(N^2) TabPFN attention.
    """
    tabpfn_layer.TransformerEncoderLayer.forward = tabpfn_layer.TransformerEncoderLayer._original_forward
    if verbose:
        print("🔄 Restored vanilla TabPFN.")
