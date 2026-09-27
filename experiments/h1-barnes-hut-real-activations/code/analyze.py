"""Aggregate pilot_results.csv and evaluate the protocol's gate. Run from repo root."""
import pandas as pd

df = pd.read_csv("experiments/h1-barnes-hut-real-activations/results/pilot_results.csv")
print("determinism check (vanilla twice) max:", df.det_check.max())
ex = df[df["mode"] == "exact"]
print("exact-vs-vanilla max diff:", ex.maxdiff.max())
v = df[df["mode"] == "vanilla"].set_index(["dataset", "seed"])
df = df.join(v[["acc"]].rename(columns={"acc": "acc_vanilla"}), on=["dataset", "seed"])
df["acc_delta_pp"] = 100 * (df.acc - df.acc_vanilla)
d = df[df["mode"] != "vanilla"]
g = d.groupby(["mode", "M", "t"]).agg(fid_median=("fid", "median"), fid_mean=("fid", "mean"),
                                      acc_delta_pp=("acc_delta_pp", "mean"), touched=("touched", "mean"),
                                      sec=("sec", "mean")).round(4)
print(g.to_string())

print("\nper-dataset median fidelity error at M=64 (lower=better):")
p = d[(d.M == 64) & (d.t.isin([0, 4]))].pivot_table(index="dataset", columns="mode", values="fid", aggfunc="median")
p["bh/kmeans"] = p["bh"] / p["isab_kmeans"]
print(p.round(4).to_string())
wins = int((p["bh/kmeans"] <= 0.5).sum())
bh = d[(d["mode"] == "bh") & (d.M == 64) & (d.t == 4)]
print(f"\nGATE: bh<=0.5x kmeans fidelity on {wins}/{len(p)} datasets (need >=7); "
      f"mean bh acc delta vs vanilla = {bh.acc_delta_pp.mean():.2f}pp (need within 0.5pp); "
      f"median bh/kmeans ratio = {p['bh/kmeans'].median():.2f} (kill if > {1/1.5:.2f})")
