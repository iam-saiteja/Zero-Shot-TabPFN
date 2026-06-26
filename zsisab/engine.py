import math
import torch
import torch.nn.functional as F

def get_zsisab_encoder_forward(original_forward_fn, num_prototypes: int = 128, use_logit_scaling: bool = True, use_norm_alignment: bool = True):
    """
    Returns a monkey-patched forward function for tabpfn.layer.TransformerEncoderLayer.
    
    Implements Zero-Shot Inducing-point Set Attention Block (ZS-ISAB) which compresses
    the training context from O(N^2) to O(N*M) attention complexity while preserving
    the pretrained TabPFN representations through:
      1. Deterministic prototype generation via chunked averaging (no row dropping)
      2. Norm alignment corrected for CLT variance reduction
      3. Symmetric dynamic logit scaling for both train and test queries
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

            # Adaptive prototype count: ensure meaningful compression
            # Use at most N//4 prototypes (minimum 16) to guarantee chunk_size >= 4
            M = min(num_prototypes, max(16, N // 4))

            train_rows = src_[:N]
            test_rows = src_[N:]

            if N <= M:
                # Fallback to dense attention (identical to vanilla TabPFN)
                src_left = self.self_attn(train_rows, train_rows, train_rows)[0]
                src_right = self.self_attn(test_rows, train_rows, train_rows)[0]
                src2 = torch.cat([src_left, src_right], dim=0)
            else:
                # --- ZERO-SHOT ISAB ---

                # Determine layout: TabPFN uses batch_first=False → src is [Seq, Batch, Embed]
                is_batch_first = self.self_attn.batch_first

                if is_batch_first:
                    train_for_chunk = train_rows          # [B, N, E]
                else:
                    train_for_chunk = train_rows.transpose(0, 1)  # [N, B, E] → [B, N, E]

                B, seq_N, E = train_for_chunk.shape

                # 1. Prototype Generation — Deterministic, no row dropping
                #    Use a fixed seed so all 12 layers see the same prototypes.
                #    Pad the last chunk so every row participates.
                generator = torch.Generator(device=train_for_chunk.device)
                generator.manual_seed(42)
                perm = torch.randperm(seq_N, device=train_for_chunk.device, generator=generator)

                chunk_size = max(1, seq_N // M)
                # Assign ALL rows to prototypes via padding (repeat last rows to fill)
                if seq_N < M * chunk_size:
                    # Fewer rows than M*chunk_size: pad by repeating
                    pad_count = M * chunk_size - seq_N
                    perm = torch.cat([perm, perm[:pad_count]])
                elif seq_N > M * chunk_size:
                    # More rows than M*chunk_size: include remainder in last chunk
                    # Reshape the first M*chunk_size into [M, chunk_size], then handle remainder
                    remainder = seq_N - M * chunk_size
                    # We'll use all rows: first M-1 chunks of chunk_size, last chunk gets the rest
                    pass  # handled below

                # Build prototypes ensuring ALL rows are used
                if seq_N <= M * chunk_size:
                    # Exact fit or padded
                    gathered = train_for_chunk[:, perm[:M * chunk_size]]
                    gathered = gathered.view(B, M, chunk_size, E)
                    proto_init = gathered.mean(dim=2)  # [B, M, E]
                    effective_chunk_size = chunk_size
                else:
                    # More rows than fit evenly: give remainder to last chunk
                    main_count = (M - 1) * chunk_size
                    main_gathered = train_for_chunk[:, perm[:main_count]]
                    main_gathered = main_gathered.view(B, M - 1, chunk_size, E)
                    main_protos = main_gathered.mean(dim=2)  # [B, M-1, E]

                    last_chunk = train_for_chunk[:, perm[main_count:]]  # [B, remainder, E]
                    last_proto = last_chunk.mean(dim=1, keepdim=True)   # [B, 1, E]

                    proto_init = torch.cat([main_protos, last_proto], dim=1)  # [B, M, E]
                    effective_chunk_size = chunk_size

                # 2. Norm Alignment — corrected for CLT variance reduction
                #    Prototypes (averages of chunks) naturally have std ≈ train_std / √chunk_size
                #    We correct for this factor rather than blindly rescaling to train_std
                if use_norm_alignment and effective_chunk_size > 1:
                    train_mean = train_for_chunk.mean(dim=1, keepdim=True)
                    train_std = train_for_chunk.std(dim=1, keepdim=True).clamp(min=1e-6)
                    proto_mean = proto_init.mean(dim=1, keepdim=True)
                    proto_std = proto_init.std(dim=1, keepdim=True).clamp(min=1e-6)

                    # Target std for prototypes: train_std / √chunk_size (CLT expectation)
                    target_std = train_std / math.sqrt(effective_chunk_size)
                    proto_init = (proto_init - proto_mean) / proto_std * target_std + train_mean

                if not is_batch_first:
                    proto_init = proto_init.transpose(0, 1)  # [B, M, E] → [M, B, E]

                # 3. Dynamic Logit Scaling — applied SYMMETRICALLY to both train and test
                if use_logit_scaling and seq_N > M:
                    scale_factor = math.sqrt(math.log(seq_N) / math.log(M))
                    train_queries_scaled = train_rows * scale_factor
                    test_queries_scaled = test_rows * scale_factor
                else:
                    train_queries_scaled = train_rows
                    test_queries_scaled = test_rows

                # Evaluate queries against prototypes
                src_left = self.self_attn(train_queries_scaled, proto_init, proto_init)[0]
                src_right = self.self_attn(test_queries_scaled, proto_init, proto_init)[0]

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
