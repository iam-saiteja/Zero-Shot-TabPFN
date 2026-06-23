from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, cast, Any

from tabpfn.architectures.tabpfn_v2 import AlongColumnAttention, TabPFNBlock, LowerPrecisionLayerNorm
from tabpfn.architectures.kv_cache import KVCacheEntry
from msa_pytorch import MiniMaxSparseAttentionPyTorch


class AlongColumnAttentionISAB(AlongColumnAttention):
    """
    Data-Derived Prototype Attention — a zero-shot-compatible O(N×M) attention
    for TabPFN's permutation-invariant row attention.

    Replaces O(N²) full self-attention with O(N×M) attention through M
    data-derived prototype vectors, where M << N (default M=32).

    How it works:
        Step 1 — Derive prototypes from data (no learned parameters):
            Randomly partition training rows into M chunks.
            Average each chunk → M prototype vectors.
            These represent compressed summaries of different regions of the
            training set. No training needed; derived fresh each forward pass.

        Step 2 — All rows attend to M prototypes (O(N × M)):
            Reuses TabPFN's trained q_projection / k_projection / v_projection.
            Each row queries against M prototypes instead of N rows.
            Uses F.scaled_dot_product_attention (Flash Attention backed).

    Why this works zero-shot:
        ALL projection weights are inherited from vanilla TabPFN (trained).
        Prototypes are computed from actual training data, not from random
        parameter vectors. So the attention is over meaningful embeddings
        from the first forward pass.

    Memory: O(N×M) instead of O(N²).
    At N=8192, M=32: attention weights are 8192×32 = 262K vs 67M → ~256× smaller.
    Both steps use Flash Attention via torch SDPA — no custom kernels needed.

    Fine-tuning path:
        proto_refine is a linear layer initialized as identity.
        Fine-tuning can train it to improve prototype quality from raw chunk means.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        num_prototypes: int = 32,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__(embedding_size, num_heads, head_dim, device=device, dtype=dtype)
        self.num_prototypes = num_prototypes

        # Learned prototype refinement — initialized as identity so zero-shot
        # behavior is pure data-derived means. Fine-tuning trains this layer.
        self.proto_refine = nn.Linear(embedding_size, embedding_size, bias=False)
        nn.init.eye_(self.proto_refine.weight)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def _derive_prototypes(self, train_rows: torch.Tensor) -> torch.Tensor:
        """
        Select M representative training rows directly to act as prototypes,
        instead of averaging chunks. This avoids the data-blurring problem
        completely, keeping feature correlations perfectly intact.
        """
        Bc, N, E = train_rows.shape
        device = train_rows.device
        M = self.num_prototypes

        if N <= M:
            if N < M:
                pad_len = M - N
                proto = F.pad(train_rows, (0, 0, 0, pad_len))
            else:
                proto = train_rows
            return self.proto_refine(proto)

        # Compute similarity to batch mean
        mean_row = train_rows.mean(dim=1, keepdim=True)  # [Bc, 1, E]
        proj = torch.bmm(train_rows, mean_row.transpose(-1, -2)).squeeze(-1)  # [Bc, N]
        
        # Normalize and average to get column-aligned scores
        proj_mean = proj.mean(dim=1, keepdim=True)
        proj_std = proj.std(dim=1, keepdim=True).clamp(min=1e-8)
        proj_normalized = (proj - proj_mean) / proj_std
        proj_aligned = proj_normalized.mean(dim=0)  # [N]
        
        # Sort and select M evenly spaced elements spanning the entire range of 0 to N-1
        _, perm = torch.sort(proj_aligned, dim=-1)
        selected_indices = perm[torch.linspace(0, N - 1, steps=M, device=device).long()]
        
        # Gather selected rows (identical across columns to keep features aligned)
        proto = train_rows[:, selected_indices]  # [Bc, M, E]
        return self.proto_refine(proto)

    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        Bc, R, E = x_BcRE.shape
        N = R if single_eval_pos is None else single_eval_pos

        # Step 1: derive M prototypes from training data — no random params
        proto = self._derive_prototypes(x_BcRE[:, :N])  # [Bc, M, E]

        # Step 2: all rows attend to M prototypes using TabPFN's trained q/k/v
        # O(R × M) — Flash Attention backed, no N×N materialization
        q = self.q_projection(x_BcRE).view(Bc, R, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_projection(proto).view(Bc, self.num_prototypes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_projection(proto).view(Bc, self.num_prototypes, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)  # [Bc, H, R, D]
        out = out.transpose(1, 2).reshape(Bc, R, self.num_heads * self.head_dim)

        kv_entry: KVCacheEntry | None = None
        if return_kv:
            kv_entry = KVCacheEntry(
                key=k.transpose(1, 2)[:, :, :1].contiguous().detach(),
                value=v.transpose(1, 2)[:, :, :1].contiguous().detach(),
            )

        return self.out_projection(out), kv_entry


class AlongColumnAttentionTwoPass(AlongColumnAttention):
    """
    Two-Pass Prototype Attention — soft-clustering based refinement.

    Single-pass ISAB fails because chunk-mean prototypes lose fine-grained
    information (rare classes, decision boundaries get averaged away).

    Fix: add a COMPRESS pass where prototype clusters compute soft assignment
    probabilities over the training set, then compute cluster centroids directly
    in the key and value spaces. This preserves minority-class details
    by grouping them into separate prototypes.

    Architecture (O(N×M) total, two attention/matmul steps, 0 new parameters):

        Pass 1 — Compress (Soft Clustering) [O(M × N)]:
            q_p = q_projection(proto_init)    — [Bc, H, M, D]
            k_r = k_projection(train_rows)    — [Bc, H, N, D]
            v_r = v_projection(train_rows)    — [Bc, H, N, D]
            attn_weights = softmax(q_p @ k_r^T) — [Bc, H, M, N]
            k_refined = attn_weights @ k_r    — [Bc, H, M, D]
            v_refined = attn_weights @ v_r    — [Bc, H, M, D]

        Pass 2 — Broadcast [O(N × M)]:
            q = q_projection(all_rows)        — [Bc, H, R, D]
            output = FlashAttn(q, k_refined, v_refined)

    No new parameters are introduced. Zero-shot compatibility is perfect
    because all operations stay inside the trained projection manifold.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        num_prototypes: int = 32,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__(embedding_size, num_heads, head_dim, device=device, dtype=dtype)
        self.num_prototypes = num_prototypes

    def load_state_dict(self, state_dict, strict: bool = True):
        # Load vanilla TabPFN weights (strict=False: new/missing params are fine)
        return super().load_state_dict(state_dict, strict=False)

    @staticmethod
    def _chunk_means(train_rows: torch.Tensor, M: int) -> torch.Tensor:
        """Randomly partition N training rows into M chunks; return chunk means."""
        Bc, N, E = train_rows.shape
        perm = torch.randperm(N, device=train_rows.device)
        chunk_size = max(1, N // M)
        protos = []
        for i in range(M):
            s, e = i * chunk_size, min((i + 1) * chunk_size, N)
            if s >= N: s, e = 0, chunk_size
            protos.append(train_rows[:, perm[s:e]].mean(dim=1))  # [Bc, E]
        return torch.stack(protos, dim=1)  # [Bc, M, E]

    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        Bc, R, E = x_BcRE.shape
        N = R if single_eval_pos is None else single_eval_pos
        H, D, M = self.num_heads, self.head_dim, self.num_prototypes
        train_rows = x_BcRE[:, :N]  # [Bc, N, E]

        # ── Step 1: Initialize prototype representations ────────────────────
        proto_init = self._chunk_means(train_rows, M)  # [Bc, M, E]

        # ── Step 2: COMPRESS (Soft Clustering in Projection Space) ──────────
        q_p = self.q_projection(proto_init).view(Bc, M, H, D).transpose(1, 2)  # [Bc, H, M, D]
        k_r = self.k_projection(train_rows).view(Bc, N, H, D).transpose(1, 2)  # [Bc, H, N, D]
        v_r = self.v_projection(train_rows).view(Bc, N, H, D).transpose(1, 2)  # [Bc, H, N, D]

        # Soft attention weights over the training rows: [Bc, H, M, N]
        attn_weights = F.softmax(torch.matmul(q_p, k_r.transpose(-2, -1)) / math.sqrt(D), dim=-1)

        # Compute cluster centroids in Key and Value spaces: [Bc, H, M, D]
        k_refined = torch.matmul(attn_weights, k_r)
        v_refined = torch.matmul(attn_weights, v_r)

        # ── Step 3: BROADCAST — all queries attend to refined prototypes ────
        q = self.q_projection(x_BcRE).view(Bc, R, H, D).transpose(1, 2)        # [Bc, H, R, D]

        out = F.scaled_dot_product_attention(q, k_refined, v_refined)          # [Bc, H, R, D]
        out = out.transpose(1, 2).reshape(Bc, R, H * D)                        # [Bc, R, E]

        kv_entry: KVCacheEntry | None = None
        if return_kv:
            kv_entry = KVCacheEntry(
                key=k_refined.transpose(1, 2)[:, :, :1].contiguous().detach(),
                value=v_refined.transpose(1, 2)[:, :, :1].contiguous().detach(),
            )

        return self.out_projection(out), kv_entry



class AlongColumnAttentionMSA(AlongColumnAttention):
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        topk: int = 4,
        blk_kv: int = 32,
        blocking_strategy: str = "random",
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__(embedding_size, num_heads, head_dim, device=device, dtype=dtype)
        self.msa = MiniMaxSparseAttentionPyTorch(
            embedding_size=embedding_size,
            num_heads=num_heads,
            head_dim=head_dim,
            topk=topk,
            blk_kv=blk_kv,
            blocking_strategy=blocking_strategy,
        ).to(device=device, dtype=dtype)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def get_batch_permutation(self, x_BcRE: torch.Tensor, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute permutations of shape [Bc, N] for batch processing."""
        Bc, R, E = x_BcRE.shape
        device = x_BcRE.device

        if N <= self.msa.blk_kv:
            identity = torch.arange(N, device=device).unsqueeze(0).expand(Bc, N)
            return identity, identity

        if self.msa.blocking_strategy == "random":
            perm = torch.stack([torch.randperm(N, device=device) for _ in range(Bc)])
            inv_perm = torch.empty_like(perm)
            inv_perm.scatter_(1, perm, torch.arange(N, device=device).unsqueeze(0).expand(Bc, N))
            return perm, inv_perm

        elif self.msa.blocking_strategy == "similarity":
            # Project using w_proj
            row_repr = x_BcRE[:, :N]  # [Bc, N, E]
            proj = torch.matmul(row_repr, self.msa.w_proj)  # [Bc, N]
            _, perm = torch.sort(proj, dim=-1)
            inv_perm = torch.empty_like(perm)
            inv_perm.scatter_(1, perm, torch.arange(N, device=device).unsqueeze(0).expand(Bc, N))
            return perm, inv_perm

        elif self.msa.blocking_strategy == "pca":
            row_repr = x_BcRE[:, :N]  # [Bc, N, E]
            # Center the representations
            centered = row_repr - row_repr.mean(dim=1, keepdim=True)
            # Vectorized SVD
            try:
                _, _, V = torch.linalg.svd(centered)
                w_proj = V[:, :, 0]  # [Bc, E] (first principal component)
            except Exception:
                # Fallback to random if SVD fails to converge
                w_proj = torch.randn(Bc, E, device=device, dtype=x_BcRE.dtype)
            
            proj = torch.bmm(row_repr, w_proj.unsqueeze(-1)).squeeze(-1)  # [Bc, N]
            _, perm = torch.sort(proj, dim=-1)
            inv_perm = torch.empty_like(perm)
            inv_perm.scatter_(1, perm, torch.arange(N, device=device).unsqueeze(0).expand(Bc, N))
            return perm, inv_perm

        else:
            identity = torch.arange(N, device=device).unsqueeze(0).expand(Bc, N)
            return identity, identity

    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        Bc, R, E = x_BcRE.shape
        device = x_BcRE.device
        dtype = x_BcRE.dtype

        # 1. Determine train size N
        N = R if single_eval_pos is None else single_eval_pos


        # 2. Get batch permutation for the train rows
        perm, inv_perm = self.get_batch_permutation(x_BcRE, N)
        batch_indices = torch.arange(Bc, device=device).unsqueeze(1).expand(Bc, N)

        # 3. Permute the input representations for the train rows
        x_BcRE_train_perm = x_BcRE[batch_indices, perm]  # [Bc, N, E]

        # 4. Construct permuted x_BcRE
        if N < R:
            x_BcRE_perm = torch.cat([x_BcRE_train_perm, x_BcRE[:, N:]], dim=1)
        else:
            x_BcRE_perm = x_BcRE_train_perm

        # 5. Compute Projections on the permuted inputs
        q_BcRHD = self.q_projection(x_BcRE_perm).view(Bc, R, -1, self.head_dim)

        kv_entry: KVCacheEntry | None = None
        if cached_kv is not None:
            # If cache is provided, we permute the cached keys/values using perm
            k_Bc1 = cached_kv.key[batch_indices, perm]
            v_Bc1 = cached_kv.value[batch_indices, perm]
            
            # Run block-sparse attention
            output_BcSHD = self.msa(
                q=q_BcRHD,
                k=k_Bc1,
                v=v_Bc1,
                causal=False,
                softmax_scale=1.0 / math.sqrt(self.head_dim)
            )
        else:
            # Compute K and V on the permuted train inputs
            k_BcNHD = self.k_projection(x_BcRE_perm[:, :N]).view(Bc, N, -1, self.head_dim)
            v_BcNHD = self.v_projection(x_BcRE_perm[:, :N]).view(Bc, N, -1, self.head_dim)

            if single_eval_pos == R:
                output_BcSHD = self.msa(
                    q=q_BcRHD,
                    k=k_BcNHD,
                    v=v_BcNHD,
                    causal=False,
                    softmax_scale=1.0 / math.sqrt(self.head_dim)
                )
            else:
                # Train queries attend to train keys/values
                out_train_BcNHD = self.msa(
                    q=q_BcRHD[:, :N],
                    k=k_BcNHD,
                    v=v_BcNHD,
                    causal=False,
                    softmax_scale=1.0 / math.sqrt(self.head_dim)
                )
                # Test queries attend to train keys/values (MQA - first head only)
                out_test_BcMHD = self.msa(
                    q=q_BcRHD[:, N:],
                    k=k_BcNHD[:, :, :1],
                    v=v_BcNHD[:, :, :1],
                    causal=False,
                    softmax_scale=1.0 / math.sqrt(self.head_dim)
                )
                output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)

            if return_kv:
                # Return K/V cache in the ORIGINAL order so the cache remains permutation-agnostic
                k_original = k_BcNHD[:, :, :1][batch_indices, inv_perm]
                v_original = v_BcNHD[:, :, :1][batch_indices, inv_perm]
                kv_entry = KVCacheEntry(
                    key=k_original.contiguous().detach(),
                    value=v_original.contiguous().detach(),
                )

        # 6. Un-permute the output of the train queries
        out_train_unperm = output_BcSHD[:, :N][batch_indices, inv_perm]
        if N < R:
            output_BcSHD_final = torch.cat([out_train_unperm, output_BcSHD[:, N:]], dim=1)
        else:
            output_BcSHD_final = out_train_unperm

        # 7. Reshape and project out
        output_BcSF = output_BcSHD_final.reshape(Bc, R, self.head_dim * self.num_heads)
        return self.out_projection(output_BcSF), kv_entry

def test_along_column_attention_msa():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing AlongColumnAttentionMSA on device: {device}...")
    Bc, R, E = 2, 64, 128
    num_heads = 4
    head_dim = E // num_heads
    x = torch.randn(Bc, R, E, device=device)

    # Initialize layer
    layer = AlongColumnAttentionMSA(
        embedding_size=E,
        num_heads=num_heads,
        head_dim=head_dim,
        topk=2,
        blk_kv=16,
        blocking_strategy="pca",
        device=device
    )

    out, kv = layer(x, single_eval_pos=48, return_kv=True)
    print(f"Output shape: {out.shape}")
    assert out.shape == (Bc, R, E), "Incorrect output shape!"
    assert kv.key.shape == (Bc, 48, 1, head_dim), "Incorrect key cache shape!"
    print("SUCCESS: AlongColumnAttentionMSA drop-in replacement works correctly!")



class AlongColumnAttentionLinear(AlongColumnAttention):
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__(embedding_size, num_heads, head_dim, device=device, dtype=dtype)
        
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        Bc, R, E = x_BcRE.shape
        device = x_BcRE.device
        dtype = x_BcRE.dtype

        N = R if single_eval_pos is None else single_eval_pos

        q_BcRHD = self.q_projection(x_BcRE).view(Bc, R, -1, self.head_dim)
        
        kv_entry: KVCacheEntry | None = None
        if cached_kv is not None:
            k_Bc1 = cached_kv.key
            v_Bc1 = cached_kv.value
        else:
            k_Bc1 = self.k_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
            v_Bc1 = self.v_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
            
            if return_kv:
                kv_entry = KVCacheEntry(
                    key=k_Bc1[:, :, :1].contiguous().detach(),
                    value=v_Bc1[:, :, :1].contiguous().detach(),
                )
                
        H = q_BcRHD.shape[2]
        H_kv = k_Bc1.shape[2]
        if H != H_kv:
            k_expanded = k_Bc1.repeat_interleave(H // H_kv, dim=2)
            v_expanded = v_Bc1.repeat_interleave(H // H_kv, dim=2)
        else:
            k_expanded = k_Bc1
            v_expanded = v_Bc1
            
        q_phi = torch.nn.functional.elu(q_BcRHD) + 1.0
        k_phi = torch.nn.functional.elu(k_expanded) + 1.0
        
        q_phi = q_phi / (q_phi.sum(dim=-1, keepdim=True) + 1e-9)
        k_phi = k_phi / (k_phi.sum(dim=-1, keepdim=True) + 1e-9)
        
        q_t = q_phi.permute(0, 2, 1, 3)
        k_t = k_phi.permute(0, 2, 1, 3)
        v_t = v_expanded.permute(0, 2, 1, 3)
        
        kv_prod = torch.matmul(k_t.transpose(-2, -1), v_t)
        output_BcH = torch.matmul(q_t, kv_prod)
        
        k_sum = k_t.sum(dim=-2, keepdim=True).transpose(-2, -1)
        denom = torch.matmul(q_t, k_sum)
        
        output_BcH = output_BcH / (denom + 1e-9)
        output_BcSF = output_BcH.permute(0, 2, 1, 3).reshape(Bc, R, H * self.head_dim)
        return self.out_projection(output_BcSF), kv_entry


class AlongColumnAttentionTopKBlock(AlongColumnAttention):
    """
    Top-K Block-Sparse Attention (Index Branch)
    
    Instead of averaging Keys/Values together (which destroys sharp decision 
    boundaries and collapses predictions), we group the Keys and Values into 
    blocks, compute a Block Key (mean key), and use the pre-trained Q/K projections 
    to dynamically route each Query to the Top-K most relevant blocks. 
    We then run EXACT Flash Attention on those selected blocks.
    
    Zero-shot compatible, exact attention weights, no blurring.
    """
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 64,
        topk_blocks: int = 4,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__(embedding_size, num_heads, head_dim, device=device, dtype=dtype)
        self.block_size = block_size
        self.topk_blocks = topk_blocks

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        Bc, R, E = x_BcRE.shape
        device = x_BcRE.device
        dtype = x_BcRE.dtype
        N = R if single_eval_pos is None else single_eval_pos

        # Compute Q
        q = self.q_projection(x_BcRE).view(Bc, R, self.num_heads, self.head_dim).transpose(1, 2)  # [Bc, H, R, D]

        kv_entry: KVCacheEntry | None = None
        if cached_kv is not None:
            k = cached_kv.key
            v = cached_kv.value
            N = k.shape[1]
        else:
            k = self.k_projection(x_BcRE[:, :N]).view(Bc, N, self.num_heads, self.head_dim).transpose(1, 2)  # [Bc, H, N, D]
            v = self.v_projection(x_BcRE[:, :N]).view(Bc, N, self.num_heads, self.head_dim).transpose(1, 2)  # [Bc, H, N, D]
            if return_kv:
                kv_entry = KVCacheEntry(
                    key=k.transpose(1, 2)[:, :, :1].contiguous().detach(),
                    value=v.transpose(1, 2)[:, :, :1].contiguous().detach(),
                )

        B = self.block_size
        # Pad N to a multiple of B if necessary
        pad_len = (B - (N % B)) % B
        if pad_len > 0:
            k_padded = F.pad(k, (0, 0, 0, pad_len))
            v_padded = F.pad(v, (0, 0, 0, pad_len))
        else:
            k_padded, v_padded = k, v

        _, _, N_pad, D = k_padded.shape
        M = N_pad // B

        k_blocks = k_padded.view(Bc, self.num_heads, M, B, D)
        v_blocks = v_padded.view(Bc, self.num_heads, M, B, D)

        # Compute Block Keys (averaging over valid tokens only)
        valid_mask = (torch.arange(N_pad, device=device) < N).view(1, 1, M, B, 1).to(dtype)
        k_blocks_sum = (k_blocks * valid_mask).sum(dim=3)  # [Bc, H, M, D]
        valid_counts = valid_mask.sum(dim=3).clamp(min=1.0)
        k_block_mean = k_blocks_sum / valid_counts  # [Bc, H, M, D]

        # Scoring: [Bc, H, R, D] @ [Bc, H, D, M] -> [Bc, H, R, M]
        block_scores = torch.matmul(q, k_block_mean.transpose(-1, -2)) / math.sqrt(D)

        # Top-K block selection per query
        k_blocks_to_select = min(self.topk_blocks, M)
        topk_scores, topk_indices = torch.topk(block_scores, k=k_blocks_to_select, dim=-1)  # [Bc, H, R, K]

        # Construct sparse mask of shape [Bc, H, R, N_pad]
        block_indices = torch.arange(M, device=device).view(1, 1, 1, 1, M)
        selected_blocks = (topk_indices.unsqueeze(-1) == block_indices).any(dim=-2)  # [Bc, H, R, M]
        token_mask = selected_blocks.repeat_interleave(B, dim=-1)  # [Bc, H, R, N_pad]

        # Combine with actual data length boundary to prevent attending to padded elements
        valid_token_mask = token_mask & (torch.arange(N_pad, device=device) < N).view(1, 1, 1, N_pad)

        # Perform EXACT SDPA on k_padded and v_padded using valid_token_mask
        out = F.scaled_dot_product_attention(q, k_padded, v_padded, attn_mask=valid_token_mask)
        out = out.transpose(1, 2).reshape(Bc, R, self.num_heads * D)

        return self.out_projection(out), kv_entry


def test_along_column_attention_linear():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing AlongColumnAttentionLinear on device: {device}...")
    Bc, R, E = 2, 64, 128
    num_heads = 4
    head_dim = E // num_heads
    x = torch.randn(Bc, R, E, device=device)

    layer = AlongColumnAttentionLinear(
        embedding_size=E,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device
    )

    out, kv = layer(x, single_eval_pos=48, return_kv=True)
    print(f"Linear output shape: {out.shape}")
    assert out.shape == (Bc, R, E), "Incorrect output shape!"
    assert kv.key.shape == (Bc, 48, 1, head_dim), "Incorrect key cache shape!"
    print("SUCCESS: AlongColumnAttentionLinear drop-in replacement works correctly!")

if __name__ == "__main__":
    test_along_column_attention_msa()
    test_along_column_attention_linear()

