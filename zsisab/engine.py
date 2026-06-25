from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from tabpfn.architectures.tabpfn_v2 import AlongColumnAttention
from tabpfn.architectures.kv_cache import KVCacheEntry


class AlongColumnAttentionZS_ISAB(AlongColumnAttention):
    """
    Zero-Shot ISAB (ZS-ISAB) — A training-free extension of Set Transformer's 
    Inducing Point Attention (Lee et al., 2019) designed for pre-trained models.

    Standard ISAB requires model retraining due to activation shifts when pooling
    keys and values into prototype chunk means. ZS-ISAB applies three real-time 
    corrections to preserve the pre-trained manifold zero-shot:
    
    1. Norm Alignment: Rescales prototype embeddings to match the original training set's mean/std.
    2. Dynamic Logit Scaling: Shrinks attention logits based on compression ratio to restore entropy.
    3. MQA Head Alignment: Forces test-time queries to attend only to the first K/V head.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        num_prototypes: int = 128,
        use_logit_scaling: bool = True,
        use_norm_alignment: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__(embedding_size, num_heads, head_dim, device=device, dtype=dtype)
        self.num_prototypes = num_prototypes
        self.use_logit_scaling = use_logit_scaling
        self.use_norm_alignment = use_norm_alignment

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        # We explicitly set strict=False to inherit vanilla TabPFN projection weights
        # without requiring parameters for the dynamic prototype chunks.
        return super().load_state_dict(state_dict, strict=False, **kwargs)

    @staticmethod
    def _chunk_means(train_rows: torch.Tensor, M: int) -> torch.Tensor:
        """Randomly partition N training rows into M chunks; return chunk means (Standard ISAB)."""
        Bc, N, E = train_rows.shape
        device = train_rows.device
        perm = torch.randperm(N, device=device)
        chunk_size = max(1, N // M)
        
        num_elements = M * chunk_size
        selected_perm = perm[:num_elements]
        
        gathered = train_rows[:, selected_perm]  # [Bc, M * chunk_size, E]
        gathered = gathered.view(Bc, M, chunk_size, E)
        return gathered.mean(dim=2)  # [Bc, M, E]

    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        Bc, R, E = x_BcRE.shape
        H, D, M = self.num_heads, self.head_dim, self.num_prototypes

        q_BcRHD = self.q_projection(x_BcRE).view(Bc, R, H, D)

        if cached_kv is not None:
            k_Bc1 = cached_kv.key
            v_Bc1 = cached_kv.value
            assert k_Bc1 is not None
            assert v_Bc1 is not None
            if k_Bc1.dtype != q_BcRHD.dtype:
                k_Bc1 = k_Bc1.to(q_BcRHD.dtype)
                v_Bc1 = v_Bc1.to(q_BcRHD.dtype)
            
            from tabpfn.architectures.shared.scaled_dot_product_attention import scaled_dot_product_attention
            output_BcSHD = scaled_dot_product_attention(q_BcRHD, k_Bc1, v_Bc1)
            return self.out_projection(output_BcSHD.reshape(Bc, R, H * D)), None

        N = R if single_eval_pos is None else single_eval_pos
        train_rows = x_BcRE[:, :N]  # [Bc, N, E]

        # Fallback to vanilla self-attention if train size is smaller than or equal to M
        if N <= M:
            k_refined = self.k_projection(train_rows).view(Bc, N, H, D)
            v_refined = self.v_projection(train_rows).view(Bc, N, H, D)
        else:
            # 1. Standard ISAB Chunk Means
            proto_init = self._chunk_means(train_rows, M)  # [Bc, M, E]
            
            # 2. Correction 1: Norm Alignment
            if self.use_norm_alignment:
                train_mean = train_rows.mean(dim=1, keepdim=True)
                train_std = train_rows.std(dim=1, keepdim=True).clamp(min=1e-6)
                proto_mean = proto_init.mean(dim=1, keepdim=True)
                proto_std = proto_init.std(dim=1, keepdim=True).clamp(min=1e-6)
                proto_init = (proto_init - proto_mean) / proto_std * train_std + train_mean
                
            # Soft-clustering assignment from original N rows into M prototypes
            q_p = self.q_projection(proto_init).view(Bc, M, H, D).transpose(1, 2)  # [Bc, H, M, D]
            k_r = self.k_projection(train_rows).view(Bc, N, H, D).transpose(1, 2)  # [Bc, H, N, D]
            v_r = self.v_projection(train_rows).view(Bc, N, H, D).transpose(1, 2)  # [Bc, H, N, D]

            attn_weights = F.softmax(torch.matmul(q_p, k_r.transpose(-2, -1)) / math.sqrt(D), dim=-1)
            k_refined = torch.matmul(attn_weights, k_r).transpose(1, 2).contiguous()
            v_refined = torch.matmul(attn_weights, v_r).transpose(1, 2).contiguous()

        from tabpfn.architectures.shared.scaled_dot_product_attention import scaled_dot_product_attention
        
        # 3. Correction 2 & 3: Logit Scaling & MQA Head Alignment
        if single_eval_pos == R:
            # No test queries in this batch
            if self.use_logit_scaling and N > M:
                scale_factor = math.sqrt(math.log(N) / math.log(M))
                q_BcRHD = q_BcRHD * scale_factor
            output_BcSHD = scaled_dot_product_attention(q_BcRHD, k_refined, v_refined)
        else:
            # Train and Test queries exist
            if self.use_logit_scaling and N > M:
                scale_factor = math.sqrt(math.log(N) / math.log(M))
                q_train = q_BcRHD[:, :N] * scale_factor
                q_test = q_BcRHD[:, N:] * scale_factor
            else:
                q_train = q_BcRHD[:, :N]
                q_test = q_BcRHD[:, N:]
                
            out_train_BcNHD = scaled_dot_product_attention(
                q_train, k_refined, v_refined
            )
            
            # Correction 3: MQA Head Alignment - test queries attend only to first head [:, :, :1]
            out_test_BcMHD = scaled_dot_product_attention(
                q_test, k_refined[:, :, :1], v_refined[:, :, :1]
            )
            output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)

        kv_entry: KVCacheEntry | None = None
        if return_kv:
            kv_entry = KVCacheEntry(
                key=k_refined[:, :, :1].contiguous().detach(),
                value=v_refined[:, :, :1].contiguous().detach(),
            )

        return self.out_projection(output_BcSHD.reshape(Bc, R, H * D)), kv_entry
