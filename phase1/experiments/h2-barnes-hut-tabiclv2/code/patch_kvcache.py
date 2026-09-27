"""Patches the REAL inference path TabICLClassifier.predict_proba() actually
uses (tf_icl.forward_with_cache), not the _train_forward bypass used in
earlier H2 scripts. Traced the call chain to be sure: predict_proba ->
_batch_forward_with_cache -> TabICL.forward_with_cache (cache_mode='kv'
default) -> ICLearning.forward_with_cache -> tf_icl.forward_with_cache
(Encoder.forward_with_cache, layers.py).

Two phases, both real (not the same call twice):
- store_cache=True (during .fit()): src has train+first test batch. Clusters
  each block's own current representations (per-layer, like every version in
  this project so far), builds the combined [train_rows ++ centroids] K/V
  and near/far attn_mask for this batch, and additionally caches per-layer
  (centroids, log_counts, cluster assignment, train_size) in a side-channel
  dict - the library's own KVCache only holds key/value tensors, nothing for
  our cluster metadata.
- use_cache=True (later .predict_proba() calls reusing the fitted model):
  src is a NEW test batch only. Rebuilds this batch's own near/far mask
  against the CACHED centroids (cheap: M dot products per query, not a full
  train_size distance computation), then attends against the cached combined
  K/V via the block's cached_kv path with that mask.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from tabicl._model.kv_cache import KVCacheEntry
from tabicl._model.encoders import Encoder


def _cluster(train, M, iters):
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


def _build_mask(q, C, log_counts, assign, train_size, t, nheads):
    B, T, d = q.shape
    M = C.shape[1]
    scores = torch.einsum('btd,bmd->btm', q, C) / (d ** 0.5) + log_counts.unsqueeze(1)
    top = scores.topk(min(t, M), dim=-1).indices
    is_top = torch.zeros(B, T, M, dtype=torch.bool, device=q.device).scatter_(-1, top, True)
    near_mask = torch.gather(is_top, 2, assign.unsqueeze(1).expand(-1, T, -1))  # (B,T,train_size)
    keep = torch.cat([near_mask, ~is_top], dim=-1)
    bias = torch.zeros(B, T, train_size + M, device=q.device, dtype=q.dtype)
    bias[:, :, train_size:] = log_counts.unsqueeze(1)
    attn_mask = bias.unsqueeze(1).expand(B, nheads, T, train_size + M).clone()
    attn_mask.masked_fill_(~keep.unsqueeze(1), float('-inf'))
    return attn_mask


class BarnesHutKVState:
    """Side-channel per-layer cluster metadata, separate from the library's KVCache
    (which only knows about key/value tensors)."""
    def __init__(self):
        self.layers: dict[int, dict] = {}
        self.reduced = False  # False if train_size <= M (fallback: plain exact caching, no BH)


_ORIGINAL_FORWARD_WITH_CACHE = Encoder.forward_with_cache


def patch_tf_icl_kvcache(M: int = 64, iters: int = 3, t: int = 4):
    """Class-level patch, applied BEFORE clf.fit() - the model (and tf_icl instance)
    doesn't exist until fit() creates it, and fit() itself calls forward_with_cache
    internally (the store_cache pass) before returning, so there's no instance to
    patch beforehand. Confirmed safe to patch at the class level (not just tf_icl)
    by reading interaction.py: tf_row (the other Encoder instance, row-wise
    interaction) never calls .forward_with_cache() as a whole method - it loops over
    tf_row.blocks directly with its own bespoke logic - so only tf_icl exercises this
    generic method in the current library version. Call unpatch_tf_icl_kvcache() to
    restore the class's original method."""
    state = BarnesHutKVState()

    def _forward_with_cache(self: Encoder, src, icl_cache, train_size=None, use_cache=False, store_cache=True):
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache:
            state.reduced = train_size is not None and train_size > M
            out = src
            for layer_idx, block in enumerate(self.blocks):
                if not state.reduced:
                    out, k_proj, v_proj = block(q=out, train_size=train_size, rope=self.rope, need_kv=True)
                    icl_cache.kv[layer_idx] = KVCacheEntry(key=k_proj, value=v_proj)
                    continue
                train = out[..., :train_size, :]
                C, assign = _cluster(train, M, iters)
                counts = F.one_hot(assign, M).to(train.dtype).sum(1)
                log_counts = torch.log(counts.clamp(min=1))
                combined = torch.cat([train, C], dim=1)
                nheads = block.attn.num_heads
                attn_mask = _build_mask(out, C, log_counts, assign, train_size, t, nheads)
                out, k_proj, v_proj = block(q=out, k=combined, v=combined, train_size=None,
                                             rope=self.rope, attn_mask=attn_mask, need_kv=True)
                icl_cache.kv[layer_idx] = KVCacheEntry(key=k_proj, value=v_proj)
                state.layers[layer_idx] = dict(centroids=C, log_counts=log_counts, assign=assign,
                                                train_size=train_size)
            return out
        else:
            out = src
            for layer_idx, block in enumerate(self.blocks):
                if not state.reduced:
                    out = block(q=out, cached_kv=icl_cache.kv[layer_idx], rope=self.rope)
                    continue
                meta = state.layers[layer_idx]
                nheads = block.attn.num_heads
                attn_mask = _build_mask(out, meta["centroids"], meta["log_counts"], meta["assign"],
                                         meta["train_size"], t, nheads)
                out = block(q=out, cached_kv=icl_cache.kv[layer_idx], attn_mask=attn_mask, rope=self.rope)
            return out

    Encoder.forward_with_cache = _forward_with_cache
    return state  # exposed for inspection/debugging, not required by callers


def unpatch_tf_icl_kvcache():
    Encoder.forward_with_cache = _ORIGINAL_FORWARD_WITH_CACHE
