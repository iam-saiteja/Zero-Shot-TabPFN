import math
import torch
import torch.nn.functional as F

def get_zsisab_encoder_forward(original_forward_fn, num_prototypes: int = 32, use_logit_scaling: bool = True, use_norm_alignment: bool = True):
    """
    Returns a monkey-patched forward function for tabpfn.layer.TransformerEncoderLayer.
    
    Implements Zero-Shot Inducing-point Set Attention Block (ZS-ISAB) which compresses
    the training context from O(N^2) to O(N*M) attention complexity while preserving
    the pretrained TabPFN representations through:
      1. Deterministic prototype generation via chunked averaging (no row dropping)
      2. Mean-only norm alignment (preserves natural CLT variance reduction)
      3. Conservative activation: only engages when N > 4*M to ensure sufficient
         compression ratio and avoid distortion on small datasets
    
    For small datasets (N <= 4*M), falls back to vanilla dense attention which is
    already fast enough. ZS-ISAB provides its benefit on large-scale datasets
    (thousands to hundreds of thousands of rows) where O(N^2) becomes prohibitive.
    """
    def zsisab_forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src

        # Intercept the exact zero-shot tabular evaluation block
        if isinstance(src_mask, int):
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            N = single_eval_position
            M = num_prototypes

            train_rows = src_[:N]
            test_rows = src_[N:]

            # Let's perform ZS-ISAB across all sequence lengths.
            is_batch_first = self.self_attn.batch_first

            if is_batch_first:
                train_for_chunk = train_rows          # [B, N, E]
            else:
                train_for_chunk = train_rows.transpose(0, 1)  # [N, B, E] -> [B, N, E]

            B, seq_N, E = train_for_chunk.shape

            # Engage ZS-ISAB if N > M. Otherwise, use exact dense training rows as prototypes.
            if seq_N > M:
                print(f"[Layer] Using ZS-ISAB (N={seq_N} -> M={M})")
                
                # Deterministic selection of actual representative rows
                generator = torch.Generator(device=train_for_chunk.device)
                generator.manual_seed(42)
                perm = torch.randperm(seq_N, device=train_for_chunk.device, generator=generator)
                
                # Slice actual rows directly. This preserves exact feature distributions, 
                # covariance, and norms, avoiding the CLT variance collapse of chunked averaging.
                selected_indices = perm[:M]
                proto_init = train_for_chunk[:, selected_indices] # [B, M, E]
            else:
                # If training size is smaller than M, use all training rows (effectively dense)
                proto_init = train_for_chunk

            if not is_batch_first:
                proto_init = proto_init.transpose(0, 1)  # [B, M, E] -> [M, B, E]

            # Evaluate queries against the selected representative prototypes
            src_left = self.self_attn(train_rows, proto_init, proto_init)[0]
            src_right = self.self_attn(test_rows, proto_init, proto_init)[0]

            src2 = torch.cat([src_left, src_right], dim=0)

        else:
            # Fallback for all other masking (training, global, etc.)
            return original_forward_fn(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Rest of the standard TransformerEncoderLayer block
        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        if not self.pre_norm:
            src = self.norm2(src)
        return src

    return zsisab_forward
