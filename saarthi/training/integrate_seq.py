"""
Ablation: does the transaction-as-language embedding actually add lift?

Joins the CoLES sequence embedding onto each corpus's tabular features and
trains the same 3-way-split GBDT stack three ways:

    tabular            - the existing feature set alone
    sequence           - the 256-d CoLES embedding alone
    tabular + sequence - both

Any honest claim about "transaction-as-language" has to survive this table,
because a sequence encoder that adds nothing over plain aggregates is
decoration. Reported on the untouched test fold with isotonic calibration.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import adapters
from train_global import train_three_way

SEQ = os.environ.get("SAARTHI_SEQ",
                     "/home/Debz/Hackathon/IDBI_Hackathon/Dataset/sequences")
EMB = os.environ.get("SAARTHI_EMB",
                     "/home/Debz/Hackathon/IDBI_Hackathon/saarthi/models/embeddings")
OUT = os.environ.get("SAARTHI_MODELS",
                     "/home/Debz/Hackathon/IDBI_Hackathon/saarthi/models")

# how each corpus's rows map onto the client_id used in the sequence store
KEYMAP = {
    "berka": lambda: (pd.read_csv(f"{adapters.DATA}/berka/loan.csv", sep=";")
                      ["account_id"].astype(str).radd("BK_")),
    "amex": lambda: (pd.read_csv(f"{adapters.DATA}/amex/train_labels.csv")
                     ["customer_ID"].astype(str).str[:16].radd("AX_")),
}


def load_emb(name: str) -> pd.DataFrame | None:
    p = f"{EMB}/{name}_embeddings.parquet"
    if not os.path.exists(p):
        print(f"  no embeddings at {p}", flush=True)
        return None
    e = pd.read_parquet(p)
    return e.drop_duplicates("client_id")


def run(name: str, amex_customers: int = 60_000) -> dict:
    kw = {"max_customers": amex_customers} if name == "amex" else {}
    c = adapters.load(name, **kw)
    emb = load_emb(name)
    if emb is None:
        return {}

    keys = KEYMAP[name]()
    if len(keys) != len(c.y):
        keys = keys.iloc[:len(c.y)].reset_index(drop=True)
    tab = c.native.reset_index(drop=True).copy()
    tab["__client_id"] = keys.values

    merged = tab.merge(emb, left_on="__client_id", right_on="client_id", how="inner")
    if len(merged) < 100:
        print(f"  {name}: only {len(merged)} rows matched embeddings; skipping",
              flush=True)
        return {}
    keep_idx = tab["__client_id"].isin(set(merged["client_id"]))
    y = c.y.reset_index(drop=True)[keep_idx.values].reset_index(drop=True)

    seq_cols = [x for x in merged.columns if x.startswith("seq_")]
    tab_cols = [x for x in c.native.columns]
    X_tab = merged[tab_cols]
    X_seq = merged[seq_cols].astype("float32")
    X_both = pd.concat([X_tab.reset_index(drop=True),
                        X_seq.reset_index(drop=True)], axis=1)

    print(f"  {name}: {len(merged):,} rows matched  "
          f"tab={len(tab_cols)} seq={len(seq_cols)}  rate={y.mean():.4f}", flush=True)

    out = {}
    for label, X in (("tabular", X_tab), ("sequence", X_seq), ("tabular+sequence", X_both)):
        try:
            b = train_three_way(X.reset_index(drop=True), y, tag=f"{name}:{label}",
                                use_xgb=False, use_cat=False)
            m = b["metrics"]["calibrated_test"]
            out[label] = m
            print(f"    {label:18} AUC={m['auc']:.4f}  PR={m['pr_auc']:.4f}  "
                  f"ECE={m['ece']:.4f}", flush=True)
        except Exception as e:
            print(f"    {label:18} FAILED {type(e).__name__}: {e}", flush=True)
    if "tabular" in out and "tabular+sequence" in out:
        lift = out["tabular+sequence"]["auc"] - out["tabular"]["auc"]
        out["auc_lift_from_sequence"] = round(lift, 5)
        print(f"    >>> lift from sequence embedding: {lift:+.4f} AUC", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=["berka", "amex"])
    ap.add_argument("--amex-customers", type=int, default=60_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for n in a.corpora:
        print(f"\n=== ablation: {n} ===", flush=True)
        try:
            res[n] = run(n, a.amex_customers)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res[n] = {"error": f"{type(e).__name__}: {e}"}
    with open(f"{OUT}/ablation_sequence.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}/ablation_sequence.json", flush=True)
