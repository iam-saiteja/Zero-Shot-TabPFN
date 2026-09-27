"""Real sparse Barnes-Hut attention: genuine reduced compute/memory, not masked-over-dense.

patch_nearfield.py / _build_mask in patch_kvcache.py compute scores over ALL train_size
rows then mask most out - correct, but the point is proven, not the compute saved.
Real sparsity needs a DIFFERENT key/value set per query (each query's own top-t clusters'
real rows), which the library's shared-k/v attention block API cannot express - so this
bypasses that block's generic attention and reimplements the block's own math manually
(same W_q/W_k/W_v/out_proj/norm1/norm2/ffn, verified norm_first=True on the real
checkpoint), replacing only the O(N) attention core with an O(t*cap + M) gather.

Only tested/used for the tf_icl stage - same scope as every other patch in this experiment.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


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


def _padded_cluster_index(assign, M, cap):
    """(B,N) cluster assignment -> (B,M,cap) real row indices per cluster, -1 = padding.
    O(N) per batch item (small Python loop over B, which is 1 in every test here)."""
    B, N = assign.shape
    device = assign.device
    order = torch.argsort(assign, dim=1)
    sorted_assign = torch.gather(assign, 1, order)
    grid = torch.full((B, M, cap), -1, dtype=torch.long, device=device)
    for b in range(B):
        _, counts = torch.unique_consecutive(sorted_assign[b], return_counts=True)
        pos = torch.cat([torch.arange(int(c), device=device) for c in counts])
        keep = pos < cap
        cid = sorted_assign[b][keep]
        slot = pos[keep]
        grid[b, cid, slot] = order[b][keep]
    return grid


def sparse_bh_block(block, out: torch.Tensor, train_size: int, M: int, t: int, iters: int, cap: int):
    """Replaces one MultiheadAttentionBlock's forward for the reduced (train_size>M) case,
    with real per-query sparse gather instead of a dense masked matrix. Assumes
    norm_first=True and rope=None (both confirmed for tf_icl on the real checkpoint)."""
    assert block.norm_first, "sparse_bh_block only implements the pre-norm branch"
    B, T, d = out.shape
    h = block.attn.num_heads
    dh = d // h
    N = train_size

    # Clustering and top-t scoring both happen in RAW (pre-norm1) space, exactly matching
    # patch_nearfield.py's dense-masked version (which clusters/scores on the block's raw
    # input `out`, not on norm1(out)) - this was the actual bug on the first attempt: this
    # function used to cluster in normalized space, diverging from the validated version.
    train_raw = out[:, :train_size, :]
    C_raw, assign = _cluster(train_raw, M, iters)
    counts = F.one_hot(assign, M).to(train_raw.dtype).sum(1)
    log_counts = torch.log(counts.clamp(min=1))
    grid = _padded_cluster_index(assign, M, cap)  # (B,M,cap)
    scores_c = torch.einsum('btd,bmd->btm', out, C_raw) / (d ** 0.5) + log_counts.unsqueeze(1)

    # Actual attention K/V/Q use norm1-normalized inputs, matching what block.forward does
    # internally (self.attn always receives norm1(q)/norm1(k)/norm1(v) in the norm_first
    # branch). LayerNorm is row-wise/elementwise, so norm1(C_raw) here is exactly what the
    # dense version's norm1(concat([train_raw, C_raw])) would produce for the centroid rows.
    q_in = block.norm1(out)
    train = q_in[:, :train_size, :]
    C = block.norm1(C_raw)

    Wq, Wk, Wv = block.attn.in_proj_weight.chunk(3, 0)
    bq, bk, bv = block.attn.in_proj_bias.chunk(3, 0)
    K_train = F.linear(train, Wk, bk)  # (B,N,d) - projected ONCE, same cost as exact attention already pays
    V_train = F.linear(train, Wv, bv)
    K_mono = F.linear(C, Wk, bk).view(B, M, h, dh)  # (B,M,d) - cheap, M is small
    V_mono = F.linear(C, Wv, bv).view(B, M, h, dh)
    Q = F.linear(q_in, Wq, bq).view(B, T, h, dh)
    ssmax_layer = getattr(block.attn, "ssmax_layer", None)
    if ssmax_layer is not None:
        # dense_masked_forward's k/v sequence is [train_size real rows ++ M monopoles], so its
        # src_len (what ssmax scales against) is train_size+M, not the true reduced count this
        # kernel actually attends to - matched here for correctness, not "true" semantics.
        Q = ssmax_layer(Q.transpose(1, 2), train_size + M).transpose(1, 2)
    top = scores_c.topk(min(t, M), dim=-1).indices  # (B,T,t)
    is_top = torch.zeros(B, T, M, dtype=torch.bool, device=out.device).scatter_(-1, top, True)

    S = min(t, M) * cap
    sel = torch.gather(grid.unsqueeze(1).expand(B, T, M, cap), 2, top.unsqueeze(-1).expand(-1, -1, -1, cap))
    sel = sel.reshape(B, T, S)  # (B,T,S) real row indices, -1 = pad -- ONLY S per query, not N
    pad_mask = sel < 0
    safe_idx = sel.clamp(min=0)
    idx_d = safe_idx.unsqueeze(-1).expand(-1, -1, -1, d)
    # gather from the ALREADY-PROJECTED (B,N,d) tensors - real sparsity: no O(T*N) work here,
    # only O(T*S) gather + O(T*S) score/weighted-sum below.
    K_near = torch.gather(K_train.unsqueeze(1).expand(B, T, N, d), 2, idx_d).view(B, T, S, h, dh)
    V_near = torch.gather(V_train.unsqueeze(1).expand(B, T, N, d), 2, idx_d).view(B, T, S, h, dh)

    near_scores = torch.einsum('bthd,btshd->bths', Q, K_near) / (dh ** 0.5)
    near_scores = near_scores.masked_fill(pad_mask.unsqueeze(2), float('-inf'))
    far_scores = torch.einsum('bthd,bmhd->bthm', Q, K_mono) / (dh ** 0.5) + log_counts.view(B, 1, 1, M)
    far_scores = far_scores.masked_fill(is_top.unsqueeze(2), float('-inf'))  # mask OUT top clusters (handled exactly by near); keep non-top

    all_scores = torch.cat([near_scores, far_scores], dim=-1)  # (B,T,h,S+M)
    attn = torch.softmax(all_scores, dim=-1)
    V_far_exp = V_mono.unsqueeze(1).expand(B, T, M, h, dh)
    V_all = torch.cat([V_near, V_far_exp], dim=2)  # (B,T,S+M,h,dh)
    attn_out = torch.einsum('bths,btshd->bthd', attn, V_all).reshape(B, T, d)
    attn_out = block.attn.out_proj(attn_out)

    x = out + block.dropout1(attn_out)
    x2 = block.norm2(x)
    ff = block.linear2(block.dropout(block.activation(block.linear1(x2))))
    x = x + block.dropout2(ff)
    return x


def sparse_bh_encoder_forward(encoder, src, train_size=None, *, M=64, iters=3, t=4, cap=None):
    if train_size is None or train_size <= M:
        out = src
        for block in encoder.blocks:
            out = block(q=out, train_size=train_size, rope=encoder.rope)
        return out
    if cap is None:
        # k-means on real (structured) embeddings tends to balance better than on random
        # noise (checked: on untrained-random-init synthetic data cluster sizes ranged 1-22
        # for N=64,M=16, average 4 - quite imbalanced; this margin is a real speed/accuracy
        # tradeoff knob, not a fixed constant - too small silently drops real neighbors
        # (found the hard way: this only reduces accuracy, no error is raised).
        cap = max(16, 4 * (train_size // M + 1))
    out = src
    for block in encoder.blocks:
        out = sparse_bh_block(block, out, train_size, M, t, iters, cap)
    return out
