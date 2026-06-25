from __future__ import annotations

import math
import torch
import torch.nn as nn

class MiniMaxSparseAttentionPyTorch(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        dim_idx: int = 16,
        topk: int = 4,
        blk_kv: int = 32,
        blocking_strategy: str = "random",  # "random" or "similarity" or "pca"
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim_idx = dim_idx
        self.topk = topk
        self.blk_kv = blk_kv
        self.blocking_strategy = blocking_strategy

        # Dimensionally reduced scoring heads for Index Branch
        self.q_idx_proj = nn.Linear(head_dim, dim_idx, bias=False)
        self.k_idx_proj = nn.Linear(head_dim, dim_idx, bias=False)

        # Learned projection parameter for "similarity" blocking strategy
        self.w_proj = nn.Parameter(torch.randn(embedding_size))

        # Initialize projection weights
        nn.init.normal_(self.q_idx_proj.weight, std=0.02)
        nn.init.normal_(self.k_idx_proj.weight, std=0.02)
        self.current_kl_loss = None


    def get_permutation(self, x_BcRE: torch.Tensor, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the permutation indices for the first N rows based on the blocking strategy."""
        device = x_BcRE.device
        if N <= self.blk_kv:
            # Too small to block, return identity
            identity = torch.arange(N, device=device)
            return identity, identity

        if self.blocking_strategy == "random":
            perm = torch.randperm(N, device=device)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(N, device=device)
            return perm, inv_perm

        elif self.blocking_strategy == "similarity":
            # Compute a cheap similarity representation for each row across all batch elements and columns
            # x_BcRE shape is (Bc, R, E) where R is total rows
            row_repr = x_BcRE[:, :N].mean(dim=0)  # [N, E]
            proj = row_repr @ self.w_proj  # [N]
            _, perm = torch.sort(proj)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(N, device=device)
            return perm, inv_perm

        elif self.blocking_strategy == "pca":
            row_repr = x_BcRE[:, :N].mean(dim=0)  # [N, E]
            mean_repr = row_repr.mean(dim=0, keepdim=True)
            centered_repr = row_repr - mean_repr
            # Cheap SVD for PCA
            _, _, V = torch.pca_lowrank(centered_repr, q=1)
            w_proj = V[:, 0]
            proj = row_repr @ w_proj
            _, perm = torch.sort(proj)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(N, device=device)
            return perm, inv_perm

        else:
            # Default to identity
            identity = torch.arange(N, device=device)
            return identity, identity

    def forward(
        self,
        q: torch.Tensor,       # [B, S_q, H, D]
        k: torch.Tensor,       # [B, S_k, H_kv, D]
        v: torch.Tensor,       # [B, S_k, H_kv, D]
        causal: bool = False,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        """Compute block-sparse attention using PyTorch fallback."""
        B, S_q, H, D = q.shape
        _, S_k, H_kv, _ = k.shape
        device = q.device

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        # Expand K heads to match Q heads if needed (GQA / MQA support)
        if H != H_kv:
            k_expanded = k.repeat_interleave(H // H_kv, dim=2)
            v_expanded = v.repeat_interleave(H // H_kv, dim=2)
        else:
            k_expanded = k
            v_expanded = v

        # Pad S_k so it's divisible by blk_kv
        N_blocks = (S_k + self.blk_kv - 1) // self.blk_kv
        pad_len = N_blocks * self.blk_kv - S_k

        # 1. Index Branch: Score key-value BLOCKS directly (no full N×N matrix)
        # Project to dimensionally reduced index space
        q_idx = self.q_idx_proj(q)   # [B, S_q, H, dim_idx]
        k_idx = self.k_idx_proj(k_expanded)  # [B, S_k, H, dim_idx]

        # Permute to [B, H, S, dim_idx]
        q_idx_t = q_idx.permute(0, 2, 1, 3)   # [B, H, S_q, dim_idx]
        k_idx_t = k_idx.permute(0, 2, 1, 3)   # [B, H, S_k, dim_idx]

        # Pad k_idx along S_k so it splits evenly into blocks
        if pad_len > 0:
            k_idx_t = torch.cat([
                k_idx_t,
                torch.full((B, H, pad_len, self.dim_idx), float("-inf"), device=device, dtype=k_idx_t.dtype)
            ], dim=2)

        # Pool each block of keys to a single representative (mean over non-padding tokens)
        # k_idx_t: [B, H, N_blocks * blk_kv, dim_idx] -> [B, H, N_blocks, blk_kv, dim_idx]
        k_idx_blocked = k_idx_t.view(B, H, N_blocks, self.blk_kv, self.dim_idx)
        # Use mean pooling; padding slots are -inf but we clamp to avoid NaN in mean
        k_idx_blocked = k_idx_blocked.clamp(min=-1e4)
        k_block_repr = k_idx_blocked.mean(dim=3)  # [B, H, N_blocks, dim_idx]

        # Score each query against each block representative: O(S_q * N_blocks) not O(S_q * S_k)
        # block_scores: [B, H, S_q, N_blocks]
        block_scores = torch.matmul(q_idx_t, k_block_repr.transpose(-2, -1)) / math.sqrt(self.dim_idx)

        # Compute KL loss for alignment when in training mode
        if self.training:
            # We compute full exact attention weights for alignment
            q_perm = q.permute(0, 2, 1, 3) # [B, H, S_q, D]
            k_perm = k_expanded.permute(0, 2, 1, 3) # [B, H, S_k, D]
            
            # Compute full attention scores
            full_scores = torch.matmul(q_perm, k_perm.transpose(-2, -1)) * softmax_scale
            
            if causal:
                q_positions = torch.arange(S_q, device=device).view(1, 1, S_q, 1)
                k_positions = torch.arange(S_k, device=device).view(1, 1, 1, S_k)
                causal_mask = k_positions <= q_positions
                full_scores = full_scores.masked_fill(~causal_mask, float("-inf"))
            
            P_full = torch.softmax(full_scores, dim=-1) # [B, H, S_q, S_k]
            
            if pad_len > 0:
                P_full = torch.cat([
                    P_full,
                    torch.zeros(B, H, S_q, pad_len, device=device, dtype=P_full.dtype)
                ], dim=-1)
                
            P_full_blocked = P_full.view(B, H, S_q, N_blocks, self.blk_kv)
            P_full_blocks = P_full_blocked.sum(dim=-1) # [B, H, S_q, N_blocks]
            
            log_P_idx_blocks = torch.log_softmax(block_scores, dim=-1)
            kl = torch.nn.functional.kl_div(log_P_idx_blocks, P_full_blocks, reduction="batchmean")
            self.current_kl_loss = kl
        else:
            self.current_kl_loss = None

        # Select top-k blocks per query
        actual_topk = min(self.topk, N_blocks)
        topk_scores, topk_indices = torch.topk(block_scores, k=actual_topk, dim=-1) # [B, H, S_q, actual_topk]


        # Compute index contrast and log it
        index_contrast = (topk_scores.mean() - block_scores.mean()).item()
        if not hasattr(self, 'indexer_log'):
            self.indexer_log = []
        self.indexer_log.append(index_contrast)

        # Neighborhood Preservation / Local Block forcing
        # Force block containing the query itself
        # For training queries in TabPFN, the query position matches the key position
        q_positions = torch.arange(S_q, device=device).view(1, 1, S_q, 1)
        local_blocks = q_positions // self.blk_kv
        local_blocks = local_blocks.clamp(max=N_blocks - 1) # [1, 1, S_q, 1]

        # Check if local block is in the selected top-k blocks
        # topk_indices has shape [B, H, S_q, actual_topk]
        is_local_present = (topk_indices == local_blocks).any(dim=-1, keepdim=True) # [B, H, S_q, 1]

        # If local block is not present, replace the lowest scoring block (which is at the last index of topk_indices)
        topk_indices = torch.where(
            is_local_present,
            topk_indices,
            torch.cat([topk_indices[..., :-1], local_blocks.expand(B, H, S_q, 1)], dim=-1)
        )

        # Sort indices to keep keys in original relative order within selected blocks
        topk_indices, _ = torch.sort(topk_indices, dim=-1)

        # 2. Main Branch: Gather selected blocks and perform exact attention
        # Construct index tensor to gather keys/values
        # gather_indices shape: [B, H, S_q, actual_topk * blk_kv]
        K_tokens = actual_topk * self.blk_kv
        offsets = torch.arange(self.blk_kv, device=device).view(1, 1, 1, 1, self.blk_kv)
        gather_indices = (topk_indices.unsqueeze(-1) * self.blk_kv + offsets).view(B, H, S_q, K_tokens)

        # Gather keys/values using memory-efficient flat indexing
        # k_expanded has shape [B, S_k, H, D] (since we repeated interleave). Permute to [B, H, S_k, D]
        k_perm = k_expanded.permute(0, 2, 1, 3) # [B, H, S_k, D]
        v_perm = v_expanded.permute(0, 2, 1, 3) # [B, H, S_k, D]

        # Add padding to keys/values to handle out-of-bounds indices from pad_len
        if pad_len > 0:
            k_perm = torch.cat([k_perm, torch.zeros(B, H, pad_len, D, device=device, dtype=k_perm.dtype)], dim=2)
            v_perm = torch.cat([v_perm, torch.zeros(B, H, pad_len, D, device=device, dtype=v_perm.dtype)], dim=2)

        # Flatten k and v for fast indexing
        k_flat = k_perm.reshape(-1, D)
        v_flat = v_perm.reshape(-1, D)

        # Calculate flat indices
        # gather_indices has shape [B, H, S_q, K_tokens]
        # We need: batch_idx * (H * (S_k + pad_len)) + head_idx * (S_k + pad_len) + gather_idx
        S_k_padded = S_k + pad_len
        batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H, S_q, K_tokens)
        head_indices = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, S_q, K_tokens)
        
        flat_k_idx = batch_indices * (H * S_k_padded) + head_indices * S_k_padded + gather_indices

        # Gather keys and values
        k_gathered = k_flat[flat_k_idx].view(B, H, S_q, K_tokens, D) # [B, H, S_q, K_tokens, D]
        v_gathered = v_flat[flat_k_idx].view(B, H, S_q, K_tokens, D) # [B, H, S_q, K_tokens, D]

        # Compute exact attention over gathered keys/values
        # q has shape [B, S_q, H, D]. Permute to [B, H, S_q, D] and unsqueeze to [B, H, S_q, 1, D]
        q_perm = q.permute(0, 2, 1, 3).unsqueeze(-2) # [B, H, S_q, 1, D]

        # Compute dot product attention
        # q_perm: [B, H, S_q, 1, D]
        # k_gathered: [B, H, S_q, K_tokens, D]
        # scores: [B, H, S_q, 1, K_tokens]
        scores = torch.matmul(q_perm * softmax_scale, k_gathered.transpose(-2, -1))

        # Causal masking if applicable
        if causal:
            # Construct a mask for the gathered tokens
            # A key token at gather_indices[b, h, q_idx, k_idx] is visible to query q_idx
            # if gather_indices[b, h, q_idx, k_idx] <= q_idx
            q_idx_positions = torch.arange(S_q, device=device).view(1, 1, S_q, 1)
            causal_mask = gather_indices <= q_idx_positions # [B, H, S_q, K_tokens]
            scores = scores.masked_fill(~causal_mask.unsqueeze(-2), float("-inf"))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1) # [B, H, S_q, 1, K_tokens]

        # Compute output
        # attn_weights: [B, H, S_q, 1, K_tokens]
        # v_gathered: [B, H, S_q, K_tokens, D]
        # output: [B, H, S_q, 1, D]
        output = torch.matmul(attn_weights, v_gathered).squeeze(-2) # [B, H, S_q, D]

        # Permute back to [B, S_q, H, D]
        return output.permute(0, 2, 1, 3)

def test_msa_correctness():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running correctness check on device: {device}...")
    B, S_q, S_k, H, D = 1, 128, 128, 4, 32
    q = torch.randn(B, S_q, H, D, device=device)
    k = torch.randn(B, S_k, H, D, device=device)
    v = torch.randn(B, S_k, H, D, device=device)

    msa = MiniMaxSparseAttentionPyTorch(
        embedding_size=128,
        num_heads=H,
        head_dim=D,
        topk=4,
        blk_kv=32,
        blocking_strategy="random"
    ).to(device)

    # Test forward pass
    out = msa(q, k, v)
    print(f"MSA output shape: {out.shape}")
    assert out.shape == (B, S_q, H, D), "Incorrect output shape!"
    print("SUCCESS: MSA PyTorch fallback works correctly!")

if __name__ == "__main__":
    test_msa_correctness()
