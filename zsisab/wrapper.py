from __future__ import annotations

import tabpfn.architectures.tabpfn_v2 as tabpfn_v2
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
import tabpfn.architectures.tabpfn_v2_6 as tabpfn_v2_6
from zsisab.engine import AlongColumnAttentionZS_ISAB


def patch_tabpfn_load_state_dict():
    """
    Patches TabPFN architectures to use strict=False when loading state dicts.
    Required for baseline models that have extra or missing parameters in their attention layers.
    """
    if not hasattr(tabpfn_v2.TabPFNV2, "_original_load_state_dict"):
        tabpfn_v2.TabPFNV2._original_load_state_dict = tabpfn_v2.TabPFNV2.load_state_dict
        def patched_load_v2(self, sd, strict=True, **kwargs):
            return self._original_load_state_dict(sd, strict=False, **kwargs)
        tabpfn_v2.TabPFNV2.load_state_dict = patched_load_v2

    if not hasattr(tabpfn_v2_5.TabPFNV2p5, "_original_load_state_dict"):
        tabpfn_v2_5.TabPFNV2p5._original_load_state_dict = tabpfn_v2_5.TabPFNV2p5.load_state_dict
        def patched_load_v2_5(self, sd, strict=True, **kwargs):
            return self._original_load_state_dict(sd, strict=False, **kwargs)
        tabpfn_v2_5.TabPFNV2p5.load_state_dict = patched_load_v2_5

    if not hasattr(tabpfn_v2_6.TabPFNV2p6, "_original_load_state_dict"):
        tabpfn_v2_6.TabPFNV2p6._original_load_state_dict = tabpfn_v2_6.TabPFNV2p6.load_state_dict
        def patched_load_v2_6(self, sd, strict=True, **kwargs):
            return self._original_load_state_dict(sd, strict=False, **kwargs)
        tabpfn_v2_6.TabPFNV2p6.load_state_dict = patched_load_v2_6


def inject_zsisab_into_tabpfn(num_prototypes: int = 128):
    """
    Monkey-patches TabPFN's core architectures to use Zero-Shot ISAB.
    This replaces O(N^2) row attention with O(N x M) zero-shot preserving attention.
    
    Args:
        num_prototypes (int): The number of prototypes (M) to compress the sequence into.
    """
    # Create a dynamic subclass to lock in the number of prototypes
    class BoundZS_ISAB(AlongColumnAttentionZS_ISAB):
        def __init__(self, *args, **kwargs):
            kwargs["num_prototypes"] = num_prototypes
            super().__init__(*args, **kwargs)

    # Patch the attention mechanisms
    tabpfn_v2.AlongColumnAttention = BoundZS_ISAB
    tabpfn_v2_5.AlongColumnAttention = BoundZS_ISAB
    tabpfn_v2_6.AlongColumnAttention = BoundZS_ISAB

    # We must patch load_state_dict to use strict=False because ZS-ISAB inherits 
    # vanilla TabPFN projections but does not require parameters for its prototypes.
    patch_tabpfn_load_state_dict()

    print(f"✅ Zero-Shot ISAB successfully injected into TabPFN architectures with M={num_prototypes}.")


def restore_vanilla_tabpfn():
    """
    Restores the original TabPFN architectures by reloading the modules.
    """
    import importlib
    importlib.reload(tabpfn_v2)
    importlib.reload(tabpfn_v2_5)
    importlib.reload(tabpfn_v2_6)
    print("🔄 Restored vanilla TabPFN.")
