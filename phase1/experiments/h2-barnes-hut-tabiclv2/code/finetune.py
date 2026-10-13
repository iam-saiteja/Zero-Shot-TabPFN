"""Track B: episodic fine-tune of the real TabICLv2 checkpoint on real OpenML
datasets, disjoint from the 30-dataset real TabArena eval set already used
for the Barnes-Hut baseline (see docs/research-log.md, 2026-09-29 entries).

Corpus: the 6 real TabArena-v0.1 classification tasks NOT in that 30-dataset
eval set (kddcup09_appetency, Diabetes130US, APSFailure, SDSS17,
customer_satisfaction_in_airline, GiveMeSomeCredit) - same distribution
family, zero overlap with eval, so re-evaluating on the original 30 after
this stays a legitimate zero-shot comparison (fine-tune corpus and eval
corpus are disjoint, per research/finetune-existing-models.md's explicit
requirement).

Objective matches the official trainer (tabicl/train/_run.py): sample an
episode (train_size context rows + query rows) from a dataset, forward
through TabICL(X, y_train), cross-entropy on the query rows, backward, step.
Low LR (this is fine-tuning a converged checkpoint, not pretraining from
scratch) with AdamW, since the paper this project already reviewed
(On Finetuning Tabular Foundation Models) found full fine-tune beats LoRA
when the model fits in memory - ours does (27.6M params, ~440MB with AdamW
states, trivial next to what episodes at seq_len<=1024 need for activations).

Checkpoints saved periodically in the exact format TabICLClassifier expects
({"config":..., "state_dict":...}) so this is resumable and the result can
be loaded straight back into TabICLClassifier(model_path=...) for real eval.
"""
import sys, os, time, random, argparse
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import openml
from huggingface_hub import hf_hub_download

from tabicl._model.tabicl import TabICL
from tabicl._sklearn.preprocessing import TransformToNumerical, EnsembleGenerator

HERE = os.path.dirname(__file__)
META_CSV = os.path.join(HERE, "..", "..", "..", "harness", "tabarena", "src", "tabarena",
                        "benchmark", "task", "metadata", "sources", "data", "TabArena-v0.1_tasks_metadata.csv")
CKPT_OUT_DIR = os.path.join(HERE, "..", "results", "finetuned_ckpts")
os.makedirs(CKPT_OUT_DIR, exist_ok=True)

# Disjoint from the 30-dataset real eval corpus - see docstring.
FT_TASKS = {
    "kddcup09_appetency": 363683,
    "Diabetes130US": 363630,
    "APSFailure": 363616,
    "SDSS17": 363699,
    "customer_satisfaction_in_airline": 363628,
    "GiveMeSomeCredit": 363673,
}

MAX_SEQ_LEN = 1024   # matches the official trainer's default max_seq_len
MIN_TRAIN_FRAC = 0.5
MAX_TRAIN_FRAC = 0.9


def load_base_model(device):
    """recompute=True enables the model's built-in gradient checkpointing
    (torch.utils.checkpoint in encoders.py/interaction.py, gated on this
    flag) - it's a pure forward-pass-mode setting, safe to override from
    the checkpoint's saved config (irrelevant to loading trained weights,
    per tabicl.py's own docstring). Needed: a first backward-pass smoke
    test OOM'd on this 4GB card even at ~1k rows, since grad tracking
    retains every layer's activations simultaneously, unlike inference."""
    ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename="tabicl-classifier-v2-20260212.ckpt",
                                 local_files_only=True)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    config["recompute"] = True
    model = TabICL(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint["config"]  # save the ORIGINAL config (recompute is a forward-mode toggle, not part of the trained architecture)


def fetch_datasets():
    """Download once, cache in memory as (X_df, y_series) per task."""
    data = {}
    for name, task_id in FT_TASKS.items():
        task = openml.tasks.get_task(task_id, download_splits=False, download_data=True)
        X, y = task.get_X_and_y(dataset_format="dataframe")
        data[name] = (X, y)
        print(f"loaded {name}: {X.shape}", flush=True)
    return data


def sample_episode(X_df, y_ser, rng):
    """One episodic in-context split from one real dataset: subsample rows,
    carve train/query, fit numeric+normalization preprocessing on the train
    portion only (no leakage), return tensors + true query labels."""
    n = len(X_df)
    seq_len = min(MAX_SEQ_LEN, n)
    idx = rng.sample(range(n), seq_len)
    train_frac = rng.uniform(MIN_TRAIN_FRAC, MAX_TRAIN_FRAC)
    train_size = max(4, min(seq_len - 4, int(seq_len * train_frac)))

    Xs = X_df.iloc[idx].reset_index(drop=True)
    ys = y_ser.iloc[idx].reset_index(drop=True)
    y_enc = pd.factorize(ys)[0]
    n_classes = len(np.unique(y_enc))
    if n_classes < 2:
        return None

    X_train_raw, X_test_raw = Xs.iloc[:train_size], Xs.iloc[train_size:]
    y_train_raw, y_test_raw = y_enc[:train_size], y_enc[train_size:]

    enc = TransformToNumerical(verbose=False)
    Xn_train = enc.fit_transform(X_train_raw)
    Xn_test = enc.transform(X_test_raw)

    eg = EnsembleGenerator(classification=True, n_estimators=1, norm_methods=["none"],
                            random_state=rng.randint(0, 2**31))
    eg.fit(Xn_train, y_train_raw)
    data = eg.transform(Xn_test, mode="both")
    Xs_ens, ys_ens = data["none"]  # (1, seq_len, feat), (1, train_size)
    y_pattern = eg.ensemble_configs_["none"][0][1]
    y_test_shuf = np.array(y_pattern)[y_test_raw.astype(int)]

    return Xs_ens, ys_ens, y_test_shuf


def run(steps, lr, save_every, log_path):
    device = "cuda"
    model, config = load_base_model(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    datasets = fetch_datasets()
    names = list(datasets.keys())
    rng = random.Random(0)

    log = open(log_path, "a")
    t_start = time.time()
    losses, accs = [], []
    for step in range(1, steps + 1):
        name = names[step % len(names)]
        X_df, y_ser = datasets[name]
        ep = None
        tries = 0
        while ep is None and tries < 5:
            ep = sample_episode(X_df, y_ser, rng)
            tries += 1
        if ep is None:
            continue
        Xs_ens, ys_ens, y_test_shuf = ep

        X_t = torch.from_numpy(Xs_ens).float().to(device)
        y_train_t = torch.from_numpy(ys_ens).float().to(device)
        y_test_t = torch.from_numpy(y_test_shuf).long().to(device)

        try:
            logits = model(X=X_t, y_train=y_train_t, d=None, return_logits=True)
            loss = F.cross_entropy(logits.flatten(end_dim=-2), y_test_t.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except torch.cuda.OutOfMemoryError as e:
            msg = f"step {step} {name}: OOM, skipping - {str(e).splitlines()[0][:80]}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            torch.cuda.empty_cache()
            continue

        with torch.no_grad():
            acc = (logits.argmax(-1).flatten() == y_test_t.flatten()).float().mean().item()
        losses.append(loss.item()); accs.append(acc)

        if step % 10 == 0:
            elapsed = time.time() - t_start
            msg = (f"step {step}/{steps} {name} loss={np.mean(losses[-10:]):.4f} "
                   f"acc={np.mean(accs[-10:]):.4f} elapsed={elapsed/60:.1f}min "
                   f"mem={torch.cuda.max_memory_allocated()/1e6:.0f}MB")
            print(msg, flush=True); log.write(msg + "\n"); log.flush()

        if step % save_every == 0 or step == steps:
            out_path = os.path.join(CKPT_OUT_DIR, f"tabicl_v2_ft_step{step}.ckpt")
            torch.save({"config": config, "state_dict": model.state_dict()}, out_path)
            print(f"saved {out_path}", flush=True); log.write(f"saved {out_path}\n"); log.flush()

    log.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_path", type=str, default=os.path.join(HERE, "..", "results", "finetune.log"))
    args = p.parse_args()
    run(args.steps, args.lr, args.save_every, args.log_path)
