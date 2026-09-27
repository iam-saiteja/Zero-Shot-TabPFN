"""Near-field exact-expansion half of Barnes-Hut for tf_icl (anchor-only,
patch.py, was shown insufficient on the real checkpoint - pred-agreement
80-94% vs exact, worse than H1's TabPFN v1 numbers). Correctness-only: builds
an explicit attn_mask over the concatenated [real train rows ++ monopoles],
so it does NOT yet save compute (still O(train_size) scores per query,
just masked) - a real sparse/gather kernel is the follow-up once this proves
the fidelity improvement is real, matching the same honest sequencing H1
used (mask-based correctness proof before a fast kernel).
"""
from __future__ import annotations
import types
import torch
import torch.nn.functional as F
from tabicl._model.encoders import Encoder


def _cluster(train: torch.Tensor, M: int, iters: int):
    """k-means on train (B,N,d). Returns centroids (B,M,d) and assignment (B,N)."""
    B, N, d = train.shape
    seed = torch.randperm(N, device=train.device)[:M]
    C = train[:, seed, :].clone()
    assign = None
    for _ in range(max(iters, 1)):
        dist = torch.cdist(train, C)
        assign = dist.argmin(-1)
        onehot = F.one_hot(assign, M).to(train.dtype)
        counts = onehot.sum(1)
        sums = torch.einsum('bnm,bnd->bmd', onehot, train)
        C = torch.where((counts > 0).unsqueeze(-1), sums / counts.clamp(min=1).unsqueeze(-1), C)
    return C, assign


def _patched_forward(self: Encoder, src: torch.Tensor, train_size=None, *, M: int, iters: int, t: int):
    if train_size is None or train_size <= M:
        out = src
        for block in self.blocks:
            out = block(q=out, train_size=train_size, rope=self.rope)
        return out

    out = src
    B, T, d = out.shape
    for block in self.blocks:
        train = out[..., :train_size, :]
        C, assign = _cluster(train, M, iters)

        combined = torch.cat([train, C], dim=1)  # (B, train_size+M, d)
        counts = F.one_hot(assign, M).to(train.dtype).sum(1)  # (B, M)
        log_counts = torch.log(counts.clamp(min=1))

        q_scores_c = torch.einsum('btd,bmd->btm', out, C) / (d ** 0.5) + log_counts.unsqueeze(1)  # (B,T,M)
        top = q_scores_c.topk(min(t, M), dim=-1).indices  # (B,T,t) - clusters kept exact per query

        # exact-near mask: keep a real train row j (assigned to cluster c) iff c is in query i's top-t
        is_top = torch.zeros(B, T, M, dtype=torch.bool, device=out.device).scatter_(-1, top, True)  # (B,T,M)
        near_mask = torch.gather(is_top, 2, assign.unsqueeze(1).expand(-1, T, -1))  # (B,T,train_size)
        # monopole mask: keep cluster c's summary iff c is NOT in query i's top-t (else it's already exact above)
        far_mask = ~is_top  # (B,T,M)
        keep = torch.cat([near_mask, far_mask], dim=-1)  # (B,T,train_size+M)

        # log(count) bias on the monopole entries: an unpicked cluster of n real rows must count as
        # n rows' worth of softmax mass, not 1 - matching the monopole design already validated in
        # research/sim_anchor_attention.py (bare boolean masking, tried first, left the mass
        # undercounted - this is the fix, not the original design).
        bias = torch.zeros(B, T, train_size + M, device=out.device, dtype=out.dtype)
        bias[:, :, train_size:] = log_counts.unsqueeze(1)
        nheads = block.attn.num_heads
        attn_mask = bias.unsqueeze(1).expand(B, nheads, T, train_size + M).clone()
        attn_mask.masked_fill_(~keep.unsqueeze(1), float('-inf'))
        out = block(q=out, k=combined, v=combined, train_size=None, rope=self.rope, attn_mask=attn_mask)
    return out


def patch_tf_icl_bh(model, M: int = 64, iters: int = 3, t: int = 4):
    tf_icl = model.icl_predictor.tf_icl
    tf_icl.forward = types.MethodType(
        lambda self, src, train_size=None: _patched_forward(self, src, train_size, M=M, iters=iters, t=t), tf_icl
    )


def unpatch_tf_icl_bh(model):
    del model.icl_predictor.tf_icl.forward
