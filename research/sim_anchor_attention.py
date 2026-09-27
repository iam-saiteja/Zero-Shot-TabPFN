"""Synthetic-attention simulation: how well do anchor schemes approximate exact softmax attention?
Run: .venv/Scripts/python.exe research/sim_anchor_attention.py
Keys are an imbalanced Gaussian mixture (tabular embeddings are clustered + imbalanced).
This measures attention-output error only -- NOT a TabArena/accuracy result."""
import torch

def kmeans(x, M, iters=10, g=None):
    c = x[torch.randperm(len(x), generator=g)[:M]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(M):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c, torch.cdist(x, c).argmin(1)

def attn(q, k, v, bias=None):
    l = q @ k.T / k.shape[1] ** 0.5
    if bias is not None:
        l = l + bias
    return torch.softmax(l, 1) @ v

def run(N, M, scale, seed, d=32, nq=256, t=4):
    g = torch.Generator().manual_seed(seed)
    K_true = 24  # true number of latent clusters
    w = torch.rand(K_true, generator=g) ** 3 + 0.02  # imbalanced sizes
    lab = torch.multinomial(w / w.sum(), N, replacement=True, generator=g)
    mu = torch.randn(K_true, d, generator=g) * 2
    k = mu[lab] + 0.5 * torch.randn(N, d, generator=g)
    v = torch.randn(K_true, d, generator=g)[lab] + 0.3 * torch.randn(N, d, generator=g)
    qi = torch.randint(0, N, (nq,), generator=g)
    q = (k[qi] + 0.3 * torch.randn(nq, d, generator=g)) * scale  # scale = attention sharpness
    exact = attn(q, k, v)
    err = lambda o: ((o - exact).norm(dim=1) / exact.norm(dim=1)).mean().item()
    out = {}

    idx = torch.randperm(N, generator=g)[:M]
    out["subsample-M (exact on M rows)"] = err(attn(q, k[idx], v[idx]))

    A = k[idx]  # random anchors, two-step ISAB as in zsisab/engine.py
    out["ISAB random anchors"] = err(attn(q, A, attn(A, k, v)))

    C, a = kmeans(k, M, g=g)
    out["ISAB kmeans anchors (Track A)"] = err(attn(q, C, attn(C, k, v)))

    cnt = torch.bincount(a, minlength=M).float().clamp(min=1)
    mk = torch.stack([k[a == j].mean(0) if (a == j).any() else C[j] for j in range(M)])
    mv = torch.stack([v[a == j].mean(0) if (a == j).any() else torch.zeros(d) for j in range(M)])
    out["monopole (kmeans + log-count)"] = err(attn(q, mk, mv, bias=cnt.log()))

    # Barnes-Hut: expand top-t clusters per query exactly, monopole for the rest
    s = q @ mk.T / d ** 0.5 + cnt.log()
    top = s.topk(t, dim=1).indices
    E = torch.zeros(nq, M, dtype=torch.bool).scatter_(1, top, True)
    le = q @ k.T / d ** 0.5
    near = E[:, a]
    lm = torch.where(E, torch.full_like(s, -1e9), s)
    lall = torch.cat([torch.where(near, le, torch.full_like(le, -1e9)), lm], 1)
    vall = torch.cat([v, mv], 0)
    out[f"Barnes-Hut (t={t} clusters exact)"] = err(torch.softmax(lall, 1) @ vall)
    return out

if __name__ == "__main__":
    N, seeds = 8000, 5
    for scale in (0.5, 1.0, 2.0):
        print(f"\nN={N}  attention sharpness scale={scale}  (mean relative L2 error, {seeds} seeds; lower=better)")
        for M in (16, 64):
            rs = [run(N, M, scale, s) for s in range(seeds)]
            print(f"  M={M}")
            for name in rs[0]:
                print(f"    {name:36s} {sum(r[name] for r in rs) / seeds:.4f}")
