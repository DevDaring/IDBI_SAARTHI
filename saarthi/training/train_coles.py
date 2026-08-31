"""
CoLES-style transaction-sequence encoder (self-contained, GPU).

Implements Contrastive Learning for Event Sequences (Babaev et al., SIGMOD'22)
without the heavy pytorch-lifestream / hydra stack, so the job runs unattended
on a rented GPU with only torch + pandas + pyarrow installed.

Idea
----
Two disjoint sub-sequences sampled from the SAME client should embed close
together; sub-sequences from DIFFERENT clients should embed apart. That trains
a GRU encoder on unlabelled transaction streams. The resulting 256-d embedding
is then a feature vector for the downstream default model.

Stages
------
  pretrain  : contrastive on unlabelled sequences (TabFormer + Berka + Amex)
  embed     : dump per-client embeddings to parquet
  probe     : quick supervised AUC check on labelled corpora (Berka / Amex)

Usage
-----
  python train_coles.py --stage pretrain --epochs 15 --out /workspace/out
  python train_coles.py --stage embed  --ckpt /workspace/out/coles.pt
  python train_coles.py --stage probe  --ckpt /workspace/out/coles.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_DIR = os.environ.get("SAARTHI_SEQ",
                         "/home/Debz/Hackathon/IDBI_Hackathon/Dataset/sequences")
SEED = 20260502
MAX_LEN = 128
N_AMT, N_KIND, N_CHAN, N_ERR, N_DT = 32, 64, 512, 64, 12

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class SeqStore:
    """Ragged client sequences held as flat arrays + offsets (memory-efficient)."""

    def __init__(self, files):
        frames = []
        for f in files:
            if os.path.exists(f):
                d = pd.read_parquet(f)
                frames.append(d)
                print(f"  loaded {f}: {len(d):,} events", flush=True)
        if not frames:
            raise SystemExit("no sequence parquet files found")
        df = pd.concat(frames, ignore_index=True)
        df["cid"] = df["client_id"].astype("category").cat.codes
        df = df.sort_values("cid", kind="stable").reset_index(drop=True)

        self.client_ids = (df.groupby("cid")["client_id"].first()
                           .reindex(range(df["cid"].max() + 1)).values)
        self.feat = np.stack([
            np.clip(df["amt_bucket"].values, 0, N_AMT - 1),
            np.clip(df["kind"].values, 0, N_KIND - 1),
            np.clip(df["channel"].values % N_CHAN, 0, N_CHAN - 1),
            np.clip(df["err"].values, 0, N_ERR - 1),
            np.clip(df["dt_bucket"].values, 0, N_DT - 1),
        ], axis=1).astype(np.int16)
        self.amount = np.nan_to_num(df["amount"].values.astype(np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0)
        counts = df.groupby("cid").size().reindex(
            range(df["cid"].max() + 1), fill_value=0).values
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.n = len(counts)
        # only clients with enough events to split into two views
        self.usable = np.where(counts >= 8)[0]
        print(f"  store: {self.n:,} clients ({len(self.usable):,} usable), "
              f"{len(self.amount):,} events", flush=True)

    def slice(self, cid):
        a, b = self.offsets[cid], self.offsets[cid + 1]
        return self.feat[a:b], self.amount[a:b]


def _pad(views, L=MAX_LEN):
    B = len(views)
    f = np.zeros((B, L, 5), dtype=np.int64)
    a = np.zeros((B, L), dtype=np.float32)
    m = np.zeros((B, L), dtype=np.float32)
    for i, (vf, va) in enumerate(views):
        n = min(len(va), L)
        if n == 0:
            continue
        f[i, :n] = vf[-n:]
        a[i, :n] = va[-n:]
        m[i, :n] = 1.0
    return torch.from_numpy(f), torch.from_numpy(a), torch.from_numpy(m)


def sample_pairs(store: SeqStore, batch: int, rng: np.random.RandomState):
    """Two disjoint random windows per client -> positive pair."""
    cids = rng.choice(store.usable, size=batch, replace=False)
    v1, v2 = [], []
    for c in cids:
        f, a = store.slice(c)
        n = len(a)
        half = n // 2
        lo = rng.randint(4, max(5, half + 1))
        hi = rng.randint(4, max(5, n - half + 1))
        v1.append((f[:half][-lo:], a[:half][-lo:]))
        v2.append((f[half:][-hi:], a[half:][-hi:]))
    return _pad(v1), _pad(v2)


def full_views(store: SeqStore, cids):
    return _pad([store.slice(c) for c in cids])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class TxnEncoder(nn.Module):
    """Embed each event, run a GRU, mean+max pool -> L2-normalised vector."""

    def __init__(self, dim=256, emb=32):
        super().__init__()
        self.e_amt = nn.Embedding(N_AMT, emb)
        self.e_kind = nn.Embedding(N_KIND, emb // 2)
        self.e_chan = nn.Embedding(N_CHAN, emb)
        self.e_err = nn.Embedding(N_ERR, emb // 4)
        self.e_dt = nn.Embedding(N_DT, emb // 4)
        in_dim = emb + emb // 2 + emb + emb // 4 + emb // 4 + 1
        self.gru = nn.GRU(in_dim, dim, num_layers=1, batch_first=True,
                          bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(dim * 4, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, f, a, m):
        x = torch.cat([
            self.e_amt(f[..., 0]), self.e_kind(f[..., 1]), self.e_chan(f[..., 2]),
            self.e_err(f[..., 3]), self.e_dt(f[..., 4]), a.unsqueeze(-1),
        ], dim=-1)
        h, _ = self.gru(x)
        mm = m.unsqueeze(-1)
        mean = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
        # dtype-safe mask value: -1e9 overflows fp16 under AMP autocast
        mx = h.masked_fill(mm == 0, torch.finfo(h.dtype).min).max(1).values
        z = self.proj(torch.cat([mean, mx], dim=-1))
        return F.normalize(z, dim=-1)


def nt_xent(z1, z2, temp=0.1):
    """Symmetric InfoNCE: positives are matching indices across the two views."""
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(torch.finfo(sim.dtype).min)
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, tgt)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_pretrain(args, dev):
    files = [f"{SEQ_DIR}/{n}_seq.parquet" for n in args.corpora]
    store = SeqStore(files)
    model = TxnEncoder(dim=args.dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    rng = np.random.RandomState(SEED)

    hist = []
    for ep in range(args.epochs):
        model.train()
        tot, t0 = 0.0, time.time()
        for _ in range(steps):
            (f1, a1, m1), (f2, a2, m2) = sample_pairs(store, args.batch, rng)
            f1, a1, m1 = f1.to(dev), a1.to(dev), m1.to(dev)
            f2, a2, m2 = f2.to(dev), a2.to(dev), m2.to(dev)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                loss = nt_xent(model(f1, a1, m1), model(f2, a2, m2), args.temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        msg = (f"epoch {ep+1}/{args.epochs}  loss={tot/steps:.4f}  "
               f"{time.time()-t0:.0f}s")
        print(msg, flush=True)
        hist.append({"epoch": ep + 1, "loss": tot / steps})
        os.makedirs(args.out, exist_ok=True)
        torch.save({"model": model.state_dict(), "dim": args.dim,
                    "epoch": ep + 1, "history": hist},
                   f"{args.out}/coles.pt")
        with open(f"{args.out}/pretrain_history.json", "w") as fh:
            json.dump(hist, fh, indent=2)
    print("PRETRAIN_DONE", flush=True)


@torch.no_grad()
def embed_corpus(model, store, dev, batch=512) -> pd.DataFrame:
    model.eval()
    rows, ids = [], []
    counts = np.diff(store.offsets)
    valid = np.where(counts > 0)[0]
    for i in range(0, len(valid), batch):
        cids = valid[i:i + batch]
        f, a, m = full_views(store, cids)
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
            z = model(f.to(dev), a.to(dev), m.to(dev))
        rows.append(z.float().cpu().numpy())
        ids.extend(store.client_ids[cids])
    Z = np.concatenate(rows, 0)
    out = pd.DataFrame(Z, columns=[f"seq_{i}" for i in range(Z.shape[1])])
    out.insert(0, "client_id", ids)
    return out


def _load(args, dev):
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = TxnEncoder(dim=ck.get("dim", args.dim)).to(dev)
    model.load_state_dict(ck["model"])
    return model


def stage_embed(args, dev):
    model = _load(args, dev)
    os.makedirs(args.out, exist_ok=True)
    for name in args.corpora:
        f = f"{SEQ_DIR}/{name}_seq.parquet"
        if not os.path.exists(f):
            continue
        store = SeqStore([f])
        emb = embed_corpus(model, store, dev)
        p = f"{args.out}/{name}_embeddings.parquet"
        emb.to_parquet(p, index=False)
        print(f"  {name}: {emb.shape} -> {p}", flush=True)
    print("EMBED_DONE", flush=True)


def stage_probe(args, dev):
    """Does the unsupervised embedding actually predict default?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    model = _load(args, dev)
    results = {}
    for name in args.corpora:
        lf = f"{SEQ_DIR}/{name}_labels.parquet"
        sf = f"{SEQ_DIR}/{name}_seq.parquet"
        if not (os.path.exists(lf) and os.path.exists(sf)):
            continue
        store = SeqStore([sf])
        emb = embed_corpus(model, store, dev)
        lab = pd.read_parquet(lf)
        m = emb.merge(lab, on="client_id", how="inner")
        if m["target"].nunique() < 2 or len(m) < 50:
            continue
        X = m.drop(columns=["client_id", "target"]).values
        y = m["target"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        results[name] = {"auc": round(float(auc), 4), "n": int(len(m))}
        print(f"  PROBE {name}: linear-probe AUC={auc:.4f} on n={len(m):,}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/probe_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("PROBE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pretrain",
                    choices=["pretrain", "embed", "probe"])
    ap.add_argument("--corpora", nargs="*", default=["tabformer", "berka", "amex"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--ckpt", default="out/coles.pt")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  "
          f"{torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'}", flush=True)
    {"pretrain": stage_pretrain, "embed": stage_embed, "probe": stage_probe}[args.stage](args, dev)


if __name__ == "__main__":
    main()
