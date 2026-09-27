"""Instance-level patch: replace tf_icl's exact train_size attention with k-means anchors.

Scope note (honest, not silently downgraded): this ports the coreset-anchor
half of H1 (validated: median fidelity error ~0.07 vs random's ~0.09-0.11 on
real TabPFN v1 data), not yet the near-field exact-expansion half ("Barnes-Hut"
proper). MultiheadAttentionBlock's k/v-substitution API takes a single k/v
tensor shared by all queries; near-field needs a *different* k/v set per
query (its own nearby exact rows + everyone else's monopoles), which needs
attn_mask-based masking, not just k/v substitution. Left as the next step so
this doesn't ship a rushed, unverified version of the harder half.

Patches only `model.icl_predictor.tf_icl` (the dataset-wise ICL Encoder
instance) - tf_col (column embedding, already ISAB) and tf_row (row
interaction) are untouched by construction, since we never touch their
instances or the shared MultiheadAttentionBlock class.
"""
from __future__ import annotations
import types
import torch
import torch.nn.functional as F
from tabicl._model.encoders import Encoder

# Inlined from phase1/zsisab/engine.py::_refine_inducing_points (not imported: that
# package's __init__ requires tabpfn, which isn't installed in this venv - same
# pure-torch logic, already validated on real data in H1).
def _refine_inducing_points(train_rows: torch.Tensor, init_idx: torch.Tensor, M: int, chunk_size: int, iters: int) -> torch.Tensor:
    C = train_rows[:, init_idx, :].clone()
    N = train_rows.shape[1]
    if iters <= 0 or N <= M:
        return C
    for _ in range(iters):
        sum_acc = torch.zeros_like(C)
        count_acc = torch.zeros(C.shape[0], M, device=C.device, dtype=C.dtype)
        for i in range(0, N, chunk_size):
            chunk = train_rows[:, i:i + chunk_size, :]
            c_norm = (C ** 2).sum(-1)
            x_norm = (chunk ** 2).sum(-1)
            dist = x_norm.unsqueeze(-1) - 2 * torch.matmul(chunk, C.transpose(-2, -1)) + c_norm.unsqueeze(1)
            assign = dist.argmin(dim=-1)
            onehot = F.one_hot(assign, num_classes=M).to(chunk.dtype)
            sum_acc += torch.einsum('bcm,bce->bme', onehot, chunk)
            count_acc += onehot.sum(dim=1)
        has_points = count_acc > 0
        new_C = sum_acc / count_acc.clamp(min=1).unsqueeze(-1)
        C = torch.where(has_points.unsqueeze(-1), new_C, C)
    return C


def _patched_forward(self: Encoder, src: torch.Tensor, train_size=None, *, M: int, iters: int):
    if train_size is None or train_size <= M:
        out = src
        for block in self.blocks:
            out = block(q=out, train_size=train_size, rope=self.rope)
        return out

    out = src
    for block in self.blocks:
        train = out[..., :train_size, :]
        seed = torch.randperm(train_size, device=train.device)[:M]
        anchors = _refine_inducing_points(train, seed, M, chunk_size=4096, iters=iters)
        out = block(q=out, k=anchors, v=anchors, train_size=None, rope=self.rope)
    return out


def patch_tf_icl(model, M: int = 64, iters: int = 3):
    """model: a tabicl._model.tabicl.TabICL instance."""
    tf_icl = model.icl_predictor.tf_icl
    tf_icl.forward = types.MethodType(
        lambda self, src, train_size=None: _patched_forward(self, src, train_size, M=M, iters=iters), tf_icl
    )


def unpatch_tf_icl(model):
    tf_icl = model.icl_predictor.tf_icl
    del tf_icl.forward  # restores the class's bound method
