"""
Re-train the Berka specialist with the leakage fix and patch metrics.json.

The first full run used the pre-fix adapter, which aggregated transactions from
the entire account history -- including the 71% that occur AFTER the loan was
granted -- and therefore reported a meaningless AUC of 1.0000. This rebuilds
that single entry using pre-origination transactions only.
"""
from __future__ import annotations

import json
import os

import joblib

import adapters
from train_global import train_three_way

OUT = os.environ.get("SAARTHI_MODELS",
                     "/home/Debz/Hackathon/IDBI_Hackathon/saarthi/models")


def main():
    c = adapters.load("berka")           # preloan_only=True by default now
    print(c.summary(), flush=True)
    b = train_three_way(c.native, c.y, tag="berka")
    m = b["metrics"]
    cal = m["calibrated_test"]
    print(f"berka (pre-origination only): AUC={cal['auc']:.4f} "
          f"PR={cal['pr_auc']:.4f} ECE={cal['ece']:.4f} n_test={cal['n']}", flush=True)

    joblib.dump(b, f"{OUT}/specialist_berka.joblib", compress=3)

    p = f"{OUT}/metrics.json"
    rep = json.load(open(p)) if os.path.exists(p) else {"specialists": {}}
    rep.setdefault("specialists", {})["berka"] = m
    rep.setdefault("notes", {})["berka_leakage_fix"] = (
        "Berka aggregates ONLY pre-origination transactions. Using the full "
        "history leaks the label (71% of transactions post-date the loan) and "
        "yields AUC 1.0000, which is not reported.")
    json.dump(rep, open(p, "w"), indent=2)
    print(f"patched {p}", flush=True)


if __name__ == "__main__":
    main()
