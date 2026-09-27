"""Chunked EXACT attention for tf_icl - not an approximation. Same online-softmax
accumulator technique already validated in zsisab/engine.py for TabPFN v1 (and
matching the real "Chunked TabPFN" prior art in research/landscape.md), adapted to
TabICLv2's block (norm1/ssmax/out_proj/ffn). Caps peak memory to O(chunk_size)
instead of O(train_size) for the K/V materialization, while being mathematically
identical to full softmax attention - built specifically to address the paging
wall found at ~300k rows on this 4GB GPU (exact_ceiling_check.py), which needs a
zero-cost fix before any accuracy-for-speed tradeoff (Barnes-Hut) is justified.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def chunked_exact_block(block, out: torch.Tensor, train_size: int, chunk_size: int = 16384,
                        query_chunk_size: int | None = None):
    """Nested chunking (query blocks x key/value blocks), like real FlashAttention -
    chunking only the key dimension (first attempt) still leaves an O(T * chunk_size)
    scores tensor per step, T being ALL queries (~N here) - a 97GB allocation attempt
    at N=200k, chunk_size=16384. Both dimensions must be bounded independently."""
    assert block.norm_first
    B, T, d = out.shape
    h = block.attn.num_heads
    dh = d // h
    if query_chunk_size is None:
        query_chunk_size = chunk_size

    q_in = block.norm1(out)  # O(T*d), cheap - not the O(T*N) problem
    train_normed = q_in[:, :train_size, :]

    Wq, Wk, Wv = block.attn.in_proj_weight.chunk(3, 0)
    bq, bk, bv = block.attn.in_proj_bias.chunk(3, 0)
    ssmax_layer = getattr(block.attn, "ssmax_layer", None)

    attn_out_full = torch.empty(B, T, h, dh, device=out.device, dtype=out.dtype)
    for qi in range(0, T, query_chunk_size):
        q_slice = q_in[:, qi:qi + query_chunk_size, :]
        Tq = q_slice.shape[1]
        Q = F.linear(q_slice, Wq, bq).view(B, Tq, h, dh)
        if ssmax_layer is not None:
            # exact semantics here: src_len is genuinely train_size (no approximation,
            # unlike sparse_kernel.py which had to fake this to match a masked reference)
            Q = ssmax_layer(Q.transpose(1, 2), train_size).transpose(1, 2)

        running_max = torch.full((B, Tq, h, 1), float("-inf"), device=out.device, dtype=out.dtype)
        running_sum = torch.zeros((B, Tq, h, 1), device=out.device, dtype=out.dtype)
        running_val = torch.zeros((B, Tq, h, dh), device=out.device, dtype=out.dtype)

        for i in range(0, train_size, chunk_size):
            chunk = train_normed[:, i:i + chunk_size, :]
            K_chunk = F.linear(chunk, Wk, bk).view(B, -1, h, dh)
            V_chunk = F.linear(chunk, Wv, bv).view(B, -1, h, dh)
            scores = torch.einsum("bthd,bshd->bths", Q, K_chunk) / (dh ** 0.5)
            chunk_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(running_max, chunk_max)
            scale_prev = torch.exp(running_max - new_max)
            scale_new = torch.exp(scores - new_max)
            running_sum = running_sum * scale_prev + scale_new.sum(-1, keepdim=True)
            running_val = running_val * scale_prev + torch.einsum("bths,bshd->bthd", scale_new, V_chunk)
            running_max = new_max
            del K_chunk, V_chunk, scores, chunk_max, new_max, scale_prev, scale_new

        attn_out_full[:, qi:qi + Tq] = running_val / running_sum
        del Q, running_max, running_sum, running_val

    attn_out = attn_out_full.reshape(B, T, d)
    attn_out = block.attn.out_proj(attn_out)

    x = out + block.dropout1(attn_out)
    x2 = block.norm2(x)
    ff = block.linear2(block.dropout(block.activation(block.linear1(x2))))
    x = x + block.dropout2(ff)
    return x


def chunked_exact_encoder_forward(encoder, src, train_size=None, *, chunk_size=2048, query_chunk_size=2048):
    if train_size is None:
        out = src
        for block in encoder.blocks:
            out = block(q=out, train_size=train_size, rope=encoder.rope)
        return out
    out = src
    for block in encoder.blocks:
        out = chunked_exact_block(block, out, train_size, chunk_size, query_chunk_size)
    return out
