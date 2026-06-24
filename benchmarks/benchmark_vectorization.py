import torch
import time

def _chunk_means_loop(train_rows, M):
    Bc, N, E = train_rows.shape
    perm = torch.randperm(N, device=train_rows.device)
    chunk_size = max(1, N // M)
    protos = []
    for i in range(M):
        s, e = i * chunk_size, min((i + 1) * chunk_size, N)
        if s >= N: s, e = 0, chunk_size
        protos.append(train_rows[:, perm[s:e]].mean(dim=1))
    return torch.stack(protos, dim=1)

def _chunk_means_vectorized(train_rows, M):
    Bc, N, E = train_rows.shape
    device = train_rows.device
    perm = torch.randperm(N, device=device)
    chunk_size = max(1, N // M)
    num_elements = M * chunk_size
    selected_perm = perm[:num_elements]
    gathered = train_rows[:, selected_perm]
    gathered = gathered.view(Bc, M, chunk_size, E)
    return gathered.mean(dim=2)

Bc, N, E = 8, 398, 128
M = 128
x = torch.randn(Bc, N, E, device="cuda")

for _ in range(10):
    _ = _chunk_means_loop(x, M)
    _ = _chunk_means_vectorized(x, M)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    _ = _chunk_means_loop(x, M)
torch.cuda.synchronize()
t_loop = time.time() - t0

t0 = time.time()
for _ in range(100):
    _ = _chunk_means_vectorized(x, M)
torch.cuda.synchronize()
t_vec = time.time() - t0

print(f"Loop implementation (100 runs): {t_loop:.6f}s")
print(f"Vectorized implementation (100 runs): {t_vec:.6f}s")
print(f"Vectorization Speedup: {t_loop / t_vec:.1f}x")
