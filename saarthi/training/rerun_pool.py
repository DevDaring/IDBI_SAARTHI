"""
Re-train the pooled global model with within-corpus rank normalisation and
re-measure leave-one-dataset-out transfer.

Why: the first pooled run concatenated raw magnitudes across corpora denominated
in different currencies and eras (USD / DM / NT$ / CZK), so a "loan amount" of
50,000 meant something completely different per corpus. LODO transfer collapsed
to a mean AUC of 0.549 -- several hold-outs below chance, i.e. anti-predictive.
Converting every numeric feature to its within-corpus percentile keeps the
credit ordering and drops the untransferable scale.

Writes both variants into metrics.json so the comparison itself is reportable.
"""
from __future__ import annotations

import json
import os

import joblib

import adapters
from train_global import leave_one_out, train_three_way

OUT = os.environ.get("SAARTHI_MODELS",
                     "/home/Debz/Hackathon/IDBI_Hackathon/saarthi/models")


def main():
    print("=== pooled global WITH within-corpus rank normalisation ===", flush=True)
    pool = adapters.build_pool(adapters.POOLABLE, cap_per_dataset=300_000,
                               normalise=True)
    b = train_three_way(pool["X"], pool["y"], tag="global_norm")
    m = b["metrics"]
    cal = m["calibrated_test"]
    print(f"  GLOBAL(norm) AUC={cal['auc']:.4f}  ECE={cal['ece']:.4f}  "
          f"n={cal['n']:,}", flush=True)

    print("\n=== leave-one-dataset-out (normalised) ===", flush=True)
    lodo = leave_one_out(pool)
    mean_auc = sum(v["auc"] for v in lodo.values()) / max(1, len(lodo))
    print(f"  mean transfer AUC = {mean_auc:.4f}", flush=True)

    b["pool_datasets"] = adapters.POOLABLE
    b["canonical"] = adapters.CANON_ALL
    b["normalised"] = True
    joblib.dump(b, f"{OUT}/global_canonical.joblib", compress=3)

    p = f"{OUT}/metrics.json"
    rep = json.load(open(p)) if os.path.exists(p) else {}
    rep["pooled_raw"] = rep.get("pooled", {})          # keep the un-normalised run
    rep["lodo_raw"] = rep.get("lodo", {})
    rep["pooled"] = m
    rep["pooled"]["datasets"] = adapters.POOLABLE
    rep["pooled"]["normalised"] = True
    rep["lodo"] = lodo
    rep["lodo_mean_auc"] = round(mean_auc, 4)
    raw_l = rep.get("lodo_raw", {})
    if raw_l:
        rep["lodo_raw_mean_auc"] = round(
            sum(v["auc"] for v in raw_l.values()) / len(raw_l), 4)
    json.dump(rep, open(p, "w"), indent=2)
    print(f"\npatched {p}", flush=True)


if __name__ == "__main__":
    main()
