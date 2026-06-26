import math
import torch
import torch.nn.functional as F

def get_nsa_encoder_forward(original_forward_fn, num_prototypes: int = 32):
    """
    Returns a monkey-patched forward function for tabpfn.layer.TransformerEncoderLayer.
    
    Implements Zero-Shot Nystrom Softmax Attention (NSA-TabPFN) which compresses
    the training context from O(N^2) to O(N*M) attention complexity.
    
    For small datasets (N <= M), falls back to vanilla dense attention.
    """
    def nsa_forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
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

            # Perform NSA across sequence lengths
            is_batch_first = self.self_attn.batch_first

            if is_batch_first:
                train_for_chunk = train_rows          # [B, N, E]
            else:
                train_for_chunk = train_rows.transpose(0, 1)  # [N, B, E] -> [B, N, E]

            B, seq_N, E = train_for_chunk.shape

            # Engage NSA if N > M. Otherwise, use exact dense training rows.
            if seq_N > M:
                # 1. Deterministic selection of prototype anchor locations
                generator = torch.Generator(device=train_for_chunk.device)
                generator.manual_seed(42)
                perm = torch.randperm(seq_N, device=train_for_chunk.device, generator=generator)
                selected_indices = perm[:M]
                
                prototypes = train_for_chunk[:, selected_indices] # [B, M, E]
                
                # 2. Compute similarity/interpolation weights W [B, N, M]
                #    Measures how each of the N training rows maps onto the M anchors.
                scale = 1.0 / math.sqrt(E)
                scores = torch.bmm(train_for_chunk, prototypes.transpose(1, 2)) * scale  # [B, N, M]
                W = F.softmax(scores, dim=1)  # Normalize across N dimension -> [B, N, M]
                
                # 3. Project Key and Value vectors down to size M [B, M, E]
                #    This pools details from all N training rows into the M slots.
                proto_init = torch.bmm(W.transpose(1, 2), train_for_chunk)  # [B, M, E]
                
                if not is_batch_first:
                    proto_init = proto_init.transpose(0, 1)  # [B, M, E] -> [M, B, E]
            else:
                # If training size is smaller than M, use all training rows (effectively dense)
                proto_init = train_rows

            # Evaluate queries against the constructed prototypes (K=proto_init, V=proto_init)
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

    return nsa_forward
