"""Barnes-Hut attention via PyTorch's FlexAttention (torch.nn.attention.flex_attention,
built into torch>=2.5, already installed - no CUDA/Triton hand-written). torch.compile
fuses the mask_mod/score_mod into a real block-sparse kernel that SKIPS fully-masked
blocks, unlike patch_nearfield.py (materializes the full mask, still O(N) compute) or
sparse_kernel.py (correct but an unfused Python loop with real per-step overhead).

Combined K/V = [real train rows ++ M monopole centroids], same structure as
patch_nearfield.py/sparse_kernel.py - mask_mod encodes: a real row is visible iff its
cluster is in the query's top-t; a monopole is visible iff its cluster is NOT in the
query's top-t (avoids double-counting). score_mod adds the log(count) bias to monopole
entries - same math already validated, different execution engine.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

_flex_compiled = torch.compile(flex_attention, dynamic=False)


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


def flex_bh_block(block, out: torch.Tensor, train_size: int, M: int, t: int, iters: int):
    assert block.norm_first
    B, T, d = out.shape
    h = block.attn.num_heads
    dh = d // h
    N = train_size

    train_raw = out[:, :train_size, :]
    C_raw, assign = _cluster(train_raw, M, iters)
    counts = F.one_hot(assign, M).to(train_raw.dtype).sum(1)
    log_counts = torch.log(counts.clamp(min=1))
    scores_c = torch.einsum('btd,bmd->btm', out, C_raw) / (d ** 0.5) + log_counts.unsqueeze(1)
    top = scores_c.topk(min(t, M), dim=-1).indices
    is_top = torch.zeros(B, T, M, dtype=torch.bool, device=out.device).scatter_(-1, top, True)

    q_in = block.norm1(out)
    train = q_in[:, :train_size, :]
    C = block.norm1(C_raw)
    Wq, Wk, Wv = block.attn.in_proj_weight.chunk(3, 0)
    bq, bk, bv = block.attn.in_proj_bias.chunk(3, 0)
    combined_raw = torch.cat([train, C], dim=1)  # (B, N+M, d)
    K = F.linear(combined_raw, Wk, bk).view(B, N + M, h, dh).transpose(1, 2)  # (B,h,N+M,dh)
    V = F.linear(combined_raw, Wv, bv).view(B, N + M, h, dh).transpose(1, 2)
    Q = F.linear(q_in, Wq, bq).view(B, T, h, dh).transpose(1, 2)  # (B,h,T,dh)

    # is_top: (B,T,M) bool; assign: (B,N) long -> both captured by closure for mask_mod/score_mod
    def mask_mod(b, hh, q_idx, kv_idx):
        is_real = kv_idx < N
        real_cluster = assign[b, kv_idx.clamp(max=N - 1)]
        mono_cluster = (kv_idx - N).clamp(min=0)
        real_ok = is_top[b, q_idx, real_cluster]
        mono_ok = ~is_top[b, q_idx, mono_cluster.clamp(max=M - 1)]
        return torch.where(is_real, real_ok, mono_ok)

    def score_mod(score, b, hh, q_idx, kv_idx):
        is_mono = kv_idx >= N
        bias = log_counts[b, (kv_idx - N).clamp(min=0, max=M - 1)]
        return torch.where(is_mono, score + bias, score)

    block_mask = create_block_mask(mask_mod, B, None, T, N + M, device=out.device)
    attn_out = _flex_compiled(Q, K, V, score_mod=score_mod, block_mask=block_mask)
    attn_out = attn_out.transpose(1, 2).reshape(B, T, d)
    attn_out = block.attn.out_proj(attn_out)

    x = out + block.dropout1(attn_out)
    x2 = block.norm2(x)
    ff = block.linear2(block.dropout(block.activation(block.linear1(x2))))
    x = x + block.dropout2(ff)
    return x


def flex_bh_encoder_forward(encoder, src, train_size=None, *, M=64, iters=3, t=4):
    if train_size is None or train_size <= M:
        out = src
        for block in encoder.blocks:
            out = block(q=out, train_size=train_size, rope=encoder.rope)
        return out
    out = src
    for block in encoder.blocks:
        out = flex_bh_block(block, out, train_size, M, t, iters)
    return out
