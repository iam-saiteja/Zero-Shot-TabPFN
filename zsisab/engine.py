import math
import torch
import torch.nn.functional as F

def get_zsisab_encoder_forward(original_forward_fn, num_prototypes: int = 128, chunk_size: int = 16384, verbose: bool = False):
    def zsisab_forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src

        if isinstance(src_mask, int):
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            N = single_eval_position
            M = num_prototypes

            is_batch_first = self.self_attn.batch_first
            seq_dim = 1 if is_batch_first else 0
            seq_len = src_.shape[seq_dim]

            if N <= M:
                return original_forward_fn(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

            if not is_batch_first:
                src_ = src_.transpose(0, 1)
            
            B = src_.shape[0]
            E = self.self_attn.embed_dim
            num_heads = self.self_attn.num_heads
            head_dim = E // num_heads

            W_q, W_k, W_v = self.self_attn.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = self.self_attn.in_proj_bias.chunk(3, dim=0)

            train_rows = src_[:, :N, :]

            generator = torch.Generator(device=src_.device)
            generator.manual_seed(42)
            perm = torch.randperm(N, device=src_.device, generator=generator)
            selected_indices = perm[:M]
            
            I = train_rows[:, selected_indices, :]
            
            Q_I = F.linear(I, W_q, b_q)
            Q_I = Q_I.view(B, M, num_heads, head_dim).transpose(1, 2)
            
            H_num = torch.zeros((B, num_heads, M, head_dim), device=src_.device)
            H_max = torch.full((B, num_heads, M, 1), -float('inf'), device=src_.device)
            H_den = torch.zeros((B, num_heads, M, 1), device=src_.device)

            for i in range(0, N, chunk_size):
                X_chunk = train_rows[:, i:i+chunk_size, :]
                
                K_chunk = F.linear(X_chunk, W_k, b_k).view(B, -1, num_heads, head_dim).transpose(1, 2)
                V_chunk = F.linear(X_chunk, W_v, b_v).view(B, -1, num_heads, head_dim).transpose(1, 2)
                
                scores = torch.matmul(Q_I, K_chunk.transpose(-2, -1)) / math.sqrt(head_dim)
                
                chunk_max = torch.max(scores, dim=-1, keepdim=True)[0]
                new_max = torch.maximum(H_max, chunk_max)
                
                scale_prev = torch.exp(H_max - new_max)
                scale_new = torch.exp(scores - new_max)
                
                H_den = H_den * scale_prev + scale_new.sum(dim=-1, keepdim=True)
                H_num = H_num * scale_prev + torch.matmul(scale_new, V_chunk)
                
                H_max = new_max

            H_attn = H_num / H_den
            H_attn = H_attn.transpose(1, 2).contiguous().view(B, M, E)
            H = self.self_attn.out_proj(H_attn)

            K_H = F.linear(H, W_k, b_k).view(B, M, num_heads, head_dim).transpose(1, 2)
            V_H = F.linear(H, W_v, b_v).view(B, M, num_heads, head_dim).transpose(1, 2)

            output_chunks = []
            src_original = src
            
            for i in range(0, seq_len, chunk_size):
                if is_batch_first:
                    src_chunk = src_original[:, i:i+chunk_size, :]
                else:
                    src_chunk = src_original[i:i+chunk_size, :, :]
                    
                if self.pre_norm:
                    src_norm_chunk = self.norm1(src_chunk)
                else:
                    src_norm_chunk = src_chunk
                    
                if not is_batch_first:
                    src_norm_chunk = src_norm_chunk.transpose(0, 1)
                    
                Q_chunk = F.linear(src_norm_chunk, W_q, b_q).view(B, -1, num_heads, head_dim).transpose(1, 2)
                scores = torch.matmul(Q_chunk, K_H.transpose(-2, -1)) / math.sqrt(head_dim)
                attn_weights = F.softmax(scores, dim=-1)
                
                attn_out = torch.matmul(attn_weights, V_H)
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, E)
                attn_out = self.self_attn.out_proj(attn_out)
                
                if not is_batch_first:
                    attn_out = attn_out.transpose(0, 1)
                    
                src_chunk = src_chunk + self.dropout1(attn_out)
                if not self.pre_norm:
                    src_chunk = self.norm1(src_chunk)
                    
                if self.pre_norm:
                    src_norm2 = self.norm2(src_chunk)
                else:
                    src_norm2 = src_chunk
                    
                mlp_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
                
                src_chunk = src_chunk + self.dropout2(mlp_out)
                if not self.pre_norm:
                    src_chunk = self.norm2(src_chunk)
                    
                output_chunks.append(src_chunk)

            concat_dim = 1 if is_batch_first else 0
            src = torch.cat(output_chunks, dim=concat_dim)
            return src

        else:
            return original_forward_fn(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    return zsisab_forward
