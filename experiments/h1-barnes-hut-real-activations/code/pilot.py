"""H1 pilot: swap TabPFN v1 encoder self-attention for anchor schemes, compare to exact.
Run from repo root: .venv/Scripts/python.exe experiments/h1-barnes-hut-real-activations/code/pilot.py
Protocol: experiments/h1-barnes-hut-real-activations/protocol.md (locked before this ran)."""
import csv, math, os, sys, time, warnings
import numpy as np, torch, torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")
import tabpfn.layer as L
from tabpfn import TabPFNClassifier
from zsisab.engine import _refine_inducing_points

import tabpfn.scripts.transformer_prediction_interface as _tpi  # tabpfn 0.1.11 predates sklearn's force_all_finite rename
for _n in ("check_X_y", "check_array"):
    if hasattr(_tpi, _n):
        _f = getattr(_tpi, _n)
        setattr(_tpi, _n, (lambda f: lambda *a, force_all_finite=None, **k: f(*a, **({"ensure_all_finite": force_all_finite} if force_all_finite is not None else {}), **k))(_f))

ORIG = L.TransformerEncoderLayer.forward
CFG = dict(mode="vanilla", M=64, t=4, iters=3, calls=0)
QC = 256  # query chunk


def _anchors(train, Wq_00, iters):
    B, N, E = train.shape
    g = torch.Generator(device=train.device).manual_seed(42 + int(abs(Wq_00 * 10000)) % 10007)
    idx = torch.randperm(N, device=train.device, generator=g)[: CFG["M"]]
    return _refine_inducing_points(train, idx, CFG["M"], 4096, iters)


def pilot_forward(self, src, src_mask=None, src_key_padding_mask=None):
    if CFG["mode"] == "vanilla" or not isinstance(src_mask, int):
        return ORIG(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    CFG["calls"] += 1
    mode, M, t = CFG["mode"], CFG["M"], CFG["t"]
    with torch.no_grad():
        src_ = self.norm1(src) if self.pre_norm else src
        N = src_mask
        x = src_.transpose(0, 1)  # (B,S,E)
        B, S, E = x.shape
        h = self.self_attn.num_heads
        dh = E // h
        Wq, Wk, Wv = self.self_attn.in_proj_weight.chunk(3, 0)
        bq, bk, bv = self.self_attn.in_proj_bias.chunk(3, 0)
        proj = lambda z, W, b: F.linear(z, W, b).view(z.shape[0], z.shape[1], h, dh).transpose(1, 2)
        train = x[:, :N]
        K, V, Q = proj(train, Wk, bk), proj(train, Wv, bv), proj(x, Wq, bq)
        sc = 1.0 / math.sqrt(dh)
        out = torch.empty_like(Q)

        if mode == "exact":
            for i in range(0, S, QC):
                out[:, :, i:i + QC] = torch.softmax(Q[:, :, i:i + QC] @ K.transpose(-2, -1) * sc, -1) @ V
        elif mode in ("isab_random", "isab_kmeans"):
            I = _anchors(train, Wq[0, 0].item(), 0 if mode == "isab_random" else CFG["iters"])
            QI, KI = proj(I, Wq, bq), proj(I, Wk, bk)
            H = torch.softmax(QI @ K.transpose(-2, -1) * sc, -1) @ V
            for i in range(0, S, QC):
                out[:, :, i:i + QC] = torch.softmax(Q[:, :, i:i + QC] @ KI.transpose(-2, -1) * sc, -1) @ H
        elif mode == "bh":
            C = _anchors(train, Wq[0, 0].item(), CFG["iters"])
            a = torch.cdist(train, C).argmin(-1)  # (B,N)
            oh = F.one_hot(a, M).to(train.dtype)  # (B,N,M)
            n = oh.sum(1)  # (B,M)
            mu = torch.einsum("bnm,bne->bme", oh, train) / n.clamp(min=1).unsqueeze(-1)
            muK, muV = proj(mu, Wk, bk), proj(mu, Wv, bv)  # exact cluster-mean key/value (linear maps)
            logn = torch.log(n)[:, None, None, :]  # (B,1,1,M)
            Vall = torch.cat([V, muV], 2)
            aidx = a[:, None, None, :].expand(B, h, 1, N)
            for i in range(0, S, QC):
                q = Q[:, :, i:i + QC]
                qc = q.shape[2]
                s = q @ muK.transpose(-2, -1) * sc + logn  # (B,h,qc,M)
                top = s.topk(t, -1).indices
                Ex = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, top, True)
                near = Ex.gather(-1, aidx.expand(B, h, qc, N))
                le = (q @ K.transpose(-2, -1) * sc).masked_fill(~near, float("-inf"))
                lm = s.masked_fill(Ex, float("-inf"))
                out[:, :, i:i + QC] = torch.softmax(torch.cat([le, lm], -1), -1) @ Vall
        else:
            raise ValueError(mode)

        a_out = self.self_attn.out_proj(out.transpose(1, 2).reshape(B, S, E)).transpose(0, 1)
        src = src + self.dropout1(a_out)
        if not self.pre_norm:
            src = self.norm1(src)
        s2 = self.norm2(src) if self.pre_norm else src
        src = src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(s2)))))
        if not self.pre_norm:
            src = self.norm2(src)
        return src


L.TransformerEncoderLayer.forward = pilot_forward

DATASETS = {"adult": 1590, "bank-marketing": 1461, "electricity": 151, "MagicTelescope": 1120,
            "pendigits": 32, "phoneme": 1489, "spambase": 44, "satimage": 182, "jm1": 1053, "kr-vs-kp": 3}
CONFIGS = ([("exact", 0, 0)] + [(m, M, 4) for M in (32, 64, 128) for m in ("isab_random", "isab_kmeans", "bh")]
           + [("bh", 64, 8)])


def load(name, seed, n_train=3000, n_test=500):
    d = fetch_openml(data_id=DATASETS[name], as_frame=True, parser="auto")
    X = d.data.copy()
    for c in X.columns:
        if str(X[c].dtype) in ("category", "object"):
            X[c] = X[c].astype("category").cat.codes
    X = X.astype(float).fillna(-1).values
    y = d.target.astype("category").cat.codes.values
    rng = np.random.RandomState(seed)
    p = rng.permutation(len(X))
    nt = min(n_train, len(X) - n_test)
    return X[p[:nt]], y[p[:nt]], X[p[nt:nt + n_test]], y[p[nt:nt + n_test]]


def predict(Xtr, ytr, Xte):
    clf = TabPFNClassifier(device="cuda", N_ensemble_configurations=4)
    clf.fit(Xtr, ytr, overwrite_warning=True)
    return clf.predict_proba(Xte)


if __name__ == "__main__":
    out_path = "experiments/h1-barnes-hut-real-activations/results/pilot_results.csv"
    seeds = [0, 1]
    import pandas as pd
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    done = {(r["dataset"], r["seed"]) for r in rows}
    for name in DATASETS:
        for seed in seeds:
            if (name, seed) in done:
                continue
            Xtr, ytr, Xte, yte = load(name, seed)
            CFG["mode"] = "vanilla"
            pv = predict(Xtr, ytr, Xte)
            pv2 = predict(Xtr, ytr, Xte)
            det = float(np.abs(pv - pv2).max())
            labels = np.arange(pv.shape[1])
            base = dict(dataset=name, seed=seed, N=len(Xtr), det_check=det)
            rows.append(dict(base, mode="vanilla", M=0, t=0, acc=accuracy_score(yte, pv.argmax(1)),
                             logloss=log_loss(yte, pv, labels=labels), fid=0.0, maxdiff=0.0, sec=0, touched=len(Xtr)))
            for mode, M, t in CONFIGS:
                CFG.update(mode=mode, M=M, t=t, calls=0)
                t0 = time.time()
                p = predict(Xtr, ytr, Xte)
                sec = time.time() - t0
                assert CFG["calls"] > 0, "patched path never used"
                assert np.isfinite(p).all(), (name, mode, M, t)
                touched = len(Xtr) if mode == "exact" else (M if mode.startswith("isab") else M + t * len(Xtr) / M)
                rows.append(dict(base, mode=mode, M=M, t=t, acc=accuracy_score(yte, p.argmax(1)),
                                 logloss=log_loss(yte, np.clip(p, 1e-7, 1), labels=labels),
                                 fid=float(np.abs(p - pv).mean()), maxdiff=float(np.abs(p - pv).max()),
                                 sec=round(sec, 2), touched=round(touched)))
            print(name, seed, "N", len(Xtr), "det", det, "exact-vs-vanilla maxdiff",
                  [r["maxdiff"] for r in rows if r["dataset"] == name and r["seed"] == seed and r["mode"] == "exact"],
                  flush=True)
            for attempt in range(10):  # Windows can transiently lock the file (indexer/IDE)
                try:
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                    break
                except PermissionError:
                    time.sleep(3)
    print("done", len(rows), "rows")
