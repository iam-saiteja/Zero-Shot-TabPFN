from __future__ import annotations
import importlib

import tabpfn.layer as tabpfn_layer
from zsisab.engine import get_zsisab_encoder_forward

# Store original forward fn safely
if not hasattr(tabpfn_layer.TransformerEncoderLayer, "_original_forward") or getattr(tabpfn_layer.TransformerEncoderLayer, "_original_forward") is None:
    curr_fwd = tabpfn_layer.TransformerEncoderLayer.forward
    if not getattr(curr_fwd, "_is_zsisab", False):
        tabpfn_layer.TransformerEncoderLayer._original_forward = curr_fwd

def inject_zsisab(num_prototypes: int = 128, chunk_size: int = 16384, verbose: bool = False, refine_iters: int = 1):
    """
    Monkey-patches TabPFN's TransformerEncoderLayer to use Chunked ZS-ISAB.
    This guarantees peak VRAM is strictly bounded by the chunk_size and
    allows evaluation on millions of rows natively inside the GPU.

    refine_iters: number of chunked k-means refinement passes turning the
    randomly-sampled anchor rows into a data-dependent coreset summary.
    0 reproduces the old pure-random-anchor behavior.
    """
    original_fn = getattr(tabpfn_layer.TransformerEncoderLayer, "_original_forward", None)
    if original_fn is None:
        original_fn = tabpfn_layer.TransformerEncoderLayer.forward
        tabpfn_layer.TransformerEncoderLayer._original_forward = original_fn

    patched_fn = get_zsisab_encoder_forward(
        original_fn,
        num_prototypes=num_prototypes,
        chunk_size=chunk_size,
        verbose=verbose,
        refine_iters=refine_iters
    )
    patched_fn._is_zsisab = True

    tabpfn_layer.TransformerEncoderLayer.forward = patched_fn
    if verbose:
        print(f"✅ Chunked ZS-ISAB Injected (M={num_prototypes}, chunk_size={chunk_size}, refine_iters={refine_iters})")

def restore_vanilla_tabpfn(verbose: bool = False):
    """
    Restores the original O(N^2) TabPFN attention.
    """
    tabpfn_layer.TransformerEncoderLayer.forward = tabpfn_layer.TransformerEncoderLayer._original_forward
    if verbose:
        print("🔄 Restored vanilla TabPFN.")
