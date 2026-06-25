from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabpfn.architectures.tabpfn_v2 import AlongColumnAttention
from tabpfn.architectures.kv_cache import KVCacheEntry
from zsisab.msa_pytorch import MiniMaxSparseAttentionPyTorch


class AlongColumnAttentionLinear(AlongColumnAttention):
    """
    Linear Attention Baseline
    Uses the elu(x) + 1 kernel trick from Katharopoulos et al. (2020) "Transformers are RNNs".
    """
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__(embedding_size, num_heads, head_dim, device=device, dtype=dtype)
        
    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return super().load_state_dict(state_dict, strict=False, **kwargs)

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


class AlongColumnAttentionISAB(AlongColumnAttention):
    """
    Standard ISAB Attention Baseline (Lee et al., 2019)
    Uses data-derived prototypes (chunk means) but lacks the ZS-ISAB corrections 
    (Norm alignment, Logit scaling, MQA routing), causing zero-shot degradation.
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
        self.proto_refine = nn.Linear(embedding_size, embedding_size, bias=False)
        nn.init.eye_(self.proto_refine.weight)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return super().load_state_dict(state_dict, strict=False, **kwargs)

    def _derive_prototypes(self, train_rows: torch.Tensor) -> torch.Tensor:
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
        mean_row = train_rows.mean(dim=1, keepdim=True)
        proj = torch.bmm(train_rows, mean_row.transpose(-1, -2)).squeeze(-1)
        proj_mean = proj.mean(dim=1, keepdim=True)
        proj_std = proj.std(dim=1, keepdim=True).clamp(min=1e-8)
        proj_normalized = (proj - proj_mean) / proj_std
        proj_aligned = proj_normalized.mean(dim=0)
        _, perm = torch.sort(proj_aligned, dim=-1)
        selected_indices = perm[torch.linspace(0, N - 1, steps=M, device=device).long()]
        proto = train_rows[:, selected_indices]
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
        proto = self._derive_prototypes(x_BcRE[:, :N])
        q = self.q_projection(x_BcRE).view(Bc, R, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_projection(proto).view(Bc, self.num_prototypes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_projection(proto).view(Bc, self.num_prototypes, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(Bc, R, self.num_heads * self.head_dim)
        kv_entry: KVCacheEntry | None = None
        if return_kv:
            kv_entry = KVCacheEntry(
                key=k.transpose(1, 2)[:, :, :1].contiguous().detach(),
                value=v.transpose(1, 2)[:, :, :1].contiguous().detach(),
            )
        return self.out_projection(out), kv_entry


class AlongColumnAttentionMSA(AlongColumnAttention):
    """
    MiniMax Sparse Attention (MSA) Baseline
    Routes attention dynamically to top-K selected blocks based on similarity.
    """
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

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return super().load_state_dict(state_dict, strict=False, **kwargs)

    def get_batch_permutation(self, x_BcRE: torch.Tensor, N: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            row_repr = x_BcRE[:, :N]
            proj = torch.matmul(row_repr, self.msa.w_proj)
            _, perm = torch.sort(proj, dim=-1)
            inv_perm = torch.empty_like(perm)
            inv_perm.scatter_(1, perm, torch.arange(N, device=device).unsqueeze(0).expand(Bc, N))
            return perm, inv_perm
        elif self.msa.blocking_strategy == "pca":
            row_repr = x_BcRE[:, :N]
            centered = row_repr - row_repr.mean(dim=1, keepdim=True)
            try:
                _, _, V = torch.linalg.svd(centered)
                w_proj = V[:, :, 0]
            except Exception:
                w_proj = torch.randn(Bc, E, device=device, dtype=x_BcRE.dtype)
            proj = torch.bmm(row_repr, w_proj.unsqueeze(-1)).squeeze(-1)
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
        N = R if single_eval_pos is None else single_eval_pos

        perm, inv_perm = self.get_batch_permutation(x_BcRE, N)
        batch_indices = torch.arange(Bc, device=device).unsqueeze(1).expand(Bc, N)
        x_BcRE_train_perm = x_BcRE[batch_indices, perm]
        if N < R:
            x_BcRE_perm = torch.cat([x_BcRE_train_perm, x_BcRE[:, N:]], dim=1)
        else:
            x_BcRE_perm = x_BcRE_train_perm

        q_BcRHD = self.q_projection(x_BcRE_perm).view(Bc, R, -1, self.head_dim)
        kv_entry: KVCacheEntry | None = None

        if cached_kv is not None:
            k_Bc1 = cached_kv.key[batch_indices, perm]
            v_Bc1 = cached_kv.value[batch_indices, perm]
            output_BcSHD = self.msa(q=q_BcRHD, k=k_Bc1, v=v_Bc1, causal=False, softmax_scale=1.0 / math.sqrt(self.head_dim))
        else:
            k_BcNHD = self.k_projection(x_BcRE_perm[:, :N]).view(Bc, N, -1, self.head_dim)
            v_BcNHD = self.v_projection(x_BcRE_perm[:, :N]).view(Bc, N, -1, self.head_dim)
            if single_eval_pos == R:
                output_BcSHD = self.msa(q=q_BcRHD, k=k_BcNHD, v=v_BcNHD, causal=False, softmax_scale=1.0 / math.sqrt(self.head_dim))
            else:
                out_train_BcNHD = self.msa(q=q_BcRHD[:, :N], k=k_BcNHD, v=v_BcNHD, causal=False, softmax_scale=1.0 / math.sqrt(self.head_dim))
                out_test_BcMHD = self.msa(q=q_BcRHD[:, N:], k=k_BcNHD[:, :, :1], v=v_BcNHD[:, :, :1], causal=False, softmax_scale=1.0 / math.sqrt(self.head_dim))
                output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)
            if return_kv:
                k_original = k_BcNHD[:, :, :1][batch_indices, inv_perm]
                v_original = v_BcNHD[:, :, :1][batch_indices, inv_perm]
                kv_entry = KVCacheEntry(key=k_original.contiguous().detach(), value=v_original.contiguous().detach())

        out_train_unperm = output_BcSHD[:, :N][batch_indices, inv_perm]
        if N < R:
            output_BcSHD_final = torch.cat([out_train_unperm, output_BcSHD[:, N:]], dim=1)
        else:
            output_BcSHD_final = out_train_unperm
        output_BcSF = output_BcSHD_final.reshape(Bc, R, self.head_dim * self.num_heads)
        return self.out_projection(output_BcSF), kv_entry


class AlongColumnAttentionTopKBlock(AlongColumnAttention):
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

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return super().load_state_dict(state_dict, strict=False, **kwargs)

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

        q = self.q_projection(x_BcRE).view(Bc, R, self.num_heads, self.head_dim).transpose(1, 2)
        kv_entry: KVCacheEntry | None = None
        if cached_kv is not None:
            k = cached_kv.key
            v = cached_kv.value
            N = k.shape[1]
        else:
            k = self.k_projection(x_BcRE[:, :N]).view(Bc, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_projection(x_BcRE[:, :N]).view(Bc, N, self.num_heads, self.head_dim).transpose(1, 2)
            if return_kv:
                kv_entry = KVCacheEntry(
                    key=k.transpose(1, 2)[:, :, :1].contiguous().detach(),
                    value=v.transpose(1, 2)[:, :, :1].contiguous().detach(),
                )

        B = self.block_size
        pad_len = (B - (N % B)) % B
        if pad_len > 0:
            k_padded = F.pad(k, (0, 0, 0, pad_len))
            v_padded = F.pad(v, (0, 0, 0, pad_len))
        else:
            k_padded, v_padded = k, v

        _, _, N_pad, D = k_padded.shape
        M = N_pad // B

        k_blocks = k_padded.view(Bc, self.num_heads, M, B, D)
        valid_mask = (torch.arange(N_pad, device=device) < N).view(1, 1, M, B, 1).to(dtype)
        k_blocks_sum = (k_blocks * valid_mask).sum(dim=3)
        valid_counts = valid_mask.sum(dim=3).clamp(min=1.0)
        k_block_mean = k_blocks_sum / valid_counts

        block_scores = torch.matmul(q, k_block_mean.transpose(-1, -2)) / math.sqrt(D)
        k_blocks_to_select = min(self.topk_blocks, M)
        _, topk_indices = torch.topk(block_scores, k=k_blocks_to_select, dim=-1)

        block_indices = torch.arange(M, device=device).view(1, 1, 1, 1, M)
        selected_blocks = (topk_indices.unsqueeze(-1) == block_indices).any(dim=-2)
        token_mask = selected_blocks.repeat_interleave(B, dim=-1)
        valid_token_mask = token_mask & (torch.arange(N_pad, device=device) < N).view(1, 1, 1, N_pad)

        out = F.scaled_dot_product_attention(q, k_padded, v_padded, attn_mask=valid_token_mask)
        out = out.transpose(1, 2).reshape(Bc, R, self.num_heads * D)
        return self.out_projection(out), kv_entry
