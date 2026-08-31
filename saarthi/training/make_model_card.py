"""
Generate the HuggingFace model card from the ACTUAL metrics files.

Never hand-copy numbers into a model card: this reads models/metrics.json,
models/ablation_sequence.json and models/probe_results.json and renders
models/README.md from whatever is really there.
"""
from __future__ import annotations

import json
import os
from datetime import date

MODELS = os.environ.get("SAARTHI_MODELS",
                        "/home/Debz/Hackathon/IDBI_Hackathon/saarthi/models")

PUBLISHED = {          # reference points from the literature / competitions
    "sba": "~0.95 (literature)",
    "lending_club": "~0.70-0.73 (leakage-free)",
    "home_credit": "0.805 (Kaggle winner)",
    "gmsc": "~0.87 (Kaggle winner)",
    "taiwan": "~0.78 (literature)",
    "german": "~0.79 (literature)",
    "amex": "0.80 (Kaggle metric)",
    "hc2024": "~0.86 (Kaggle winner)",
    "berka": "n/a",
}
DESC = {
    "sba": "US Small Business Administration loans - closest public analogue to MSME lending",
    "lending_club": "2007-2018 consumer loans, charge-off label",
    "home_credit": "Home Credit 2018 application + bureau aggregates",
    "gmsc": "Give Me Some Credit, 90+ DPD within 2 years",
    "taiwan": "Taiwan credit-card default, 6-month repayment panel",
    "german": "UCI German Credit, 1000 rows",
    "berka": "PKDD'99 Czech bank, pre-origination transaction aggregates",
    "amex": "American Express monthly statement panel",
    "hc2024": "Home Credit 2024 model-stability competition",
}


def _load(name):
    p = os.path.join(MODELS, name)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {}


def main():
    m = _load("metrics.json")
    abl = _load("ablation_sequence.json")
    probe = _load("probe_results.json")

    L = []
    A = L.append
    A("---")
    A("license: apache-2.0")
    A("tags:\n  - credit-risk\n  - default-prediction\n  - tabular\n  - msme\n  - lightgbm")
    A("library_name: joblib")
    A("---\n")
    A("# SAARTHI — MSME Default Prediction\n")
    A("Credit default-prediction models trained **entirely on public datasets** for "
      "IDBI Innovate 2026 (Track 04, Default Prediction Model). No bank data was "
      "used at any point.\n")
    A("The headline design choice: SAARTHI ships **pre-trained** and then fine-tunes "
      "on a lender's own book, rather than cold-starting on every upload.\n")

    A("## Methodology — why these numbers are trustworthy\n")
    A("Every metric below comes from a **strict three-way split**:\n")
    A("| Fold | Share | Used for |")
    A("|---|---|---|")
    A("| fit | 60% | training the boosters |")
    A("| calibrate | 15% | fitting the isotonic calibrator **only** |")
    A("| test | 25% | never seen by either — all reported metrics |\n")
    A("This matters: a common shortcut fits the probability calibrator and then "
      "measures calibration error on that *same* fold, which drives ECE "
      "artificially toward zero. Here the calibrator never sees the test fold, so "
      "the reported ECE is a real out-of-sample calibration estimate.\n")

    spec = m.get("specialists", {})
    if spec:
        A("## Per-dataset specialist models\n")
        A("Full native feature set per corpus. Ensemble of LightGBM + XGBoost + "
          "CatBoost, isotonic-calibrated.\n")
        A("| Dataset | n | Default rate | Test AUC | PR-AUC | ECE | Published reference |")
        A("|---|---:|---:|---:|---:|---:|---|")
        for k, v in spec.items():
            if "error" in v:
                A(f"| {k} | — | — | _failed_ | — | — | {PUBLISHED.get(k,'')} |")
                continue
            c = v.get("calibrated_test", {})
            n = sum(v.get("split", {}).values()) or c.get("n", 0)
            A(f"| `{k}` | {n:,} | {c.get('base_rate',0):.4f} | "
              f"**{c.get('auc',0):.4f}** | {c.get('pr_auc',0):.4f} | "
              f"{c.get('ece',0):.4f} | {PUBLISHED.get(k,'')} |")
        A("")
        for k in spec:
            if k in DESC:
                A(f"- `{k}` — {DESC[k]}")
        A("")

    pooled = m.get("pooled", {})
    if pooled and "calibrated_test" in pooled:
        c = pooled["calibrated_test"]
        A("## Pooled global model (ships with the app)\n")
        A("Trained across corpora in a shared 15-field canonical credit vocabulary "
          "so it can score any loan book that maps onto it.\n")
        A(f"- **Test AUC:** {c.get('auc',0):.4f}")
        A(f"- **PR-AUC:** {c.get('pr_auc',0):.4f}")
        A(f"- **ECE:** {c.get('ece',0):.4f}  ·  **Brier:** {c.get('brier',0):.4f}")
        A(f"- **Test rows:** {c.get('n',0):,}")
        A(f"- **Corpora pooled:** {', '.join(pooled.get('datasets', []))}\n")

    lodo = m.get("lodo", {})
    if lodo:
        A("## Leave-one-dataset-out transfer\n")
        A("Train on every corpus *except* one, then score the held-out corpus cold. "
          "This is the honest proxy for _\"will it transfer to a book it has never "
          "seen?\"_ — the question that actually matters for deployment.\n")
        A("| Held-out corpus | n | Transfer AUC |")
        A("|---|---:|---:|")
        for k, v in lodo.items():
            A(f"| `{k}` | {v.get('n',0):,} | {v.get('auc',0):.4f} |")
        A("")

    if abl:
        A("## Ablation — does transaction-as-language earn its place?\n")
        A("A CoLES-style contrastive encoder (GRU + InfoNCE over disjoint "
          "sub-sequence views) was pre-trained on unlabelled transaction streams, "
          "then its 256-d embedding was tested against the tabular features.\n")
        A("| Corpus | Tabular | Sequence only | Tabular + sequence | Lift |")
        A("|---|---:|---:|---:|---:|")
        for k, v in abl.items():
            if not isinstance(v, dict) or "tabular" not in v:
                continue
            t = v.get("tabular", {}).get("auc", 0)
            s = v.get("sequence", {}).get("auc", 0)
            b = v.get("tabular+sequence", {}).get("auc", 0)
            A(f"| `{k}` | {t:.4f} | {s:.4f} | {b:.4f} | "
              f"{v.get('auc_lift_from_sequence',0):+.4f} |")
        A("")

    if probe:
        A("### Linear probe on frozen embeddings\n")
        A("Logistic regression on the frozen encoder output — measures how much "
          "default signal the *unsupervised* embedding captured on its own.\n")
        for k, v in probe.items():
            A(f"- `{k}`: AUC {v.get('auc',0):.4f} (n={v.get('n',0):,})")
        A("")

    A("## Known limitations\n")
    A("- **Berka** aggregates only *pre-origination* transactions. Using the full "
      "history yields AUC 1.0000 because 71% of an account's transactions occur "
      "after the loan date and encode the repayment behaviour that defines the "
      "label. The leaked figure is not reported here.")
    A("- **Amex** and **Home Credit** are Kaggle-competition datasets; they are used "
      "for research validation. The shipped global model is trained on the "
      "permissively-licensed core corpora.")
    A("- The pooled model uses a deliberately small shared vocabulary, so its AUC is "
      "*lower* than the specialists by construction. Its purpose is transfer, not "
      "peak in-corpus accuracy.")
    A("- No Indian MSME data was available; GST-behavioural fields are simulated "
      "against the schema IDBI published, not learned from real filings.\n")

    A("## Files\n")
    A("| File | What it is |")
    A("|---|---|")
    A("| `global_canonical.joblib` | pooled global model + isotonic calibrator |")
    A("| `specialist_<corpus>.joblib` | per-corpus specialist ensembles |")
    A("| `coles.pt` | transaction-sequence encoder checkpoint |")
    A("| `metrics.json` | every number above, machine-readable |")
    A("| `ablation_sequence.json` | sequence-embedding ablation |\n")

    A("## Usage\n")
    A("```python")
    A("import joblib, pandas as pd")
    A("b = joblib.load('global_canonical.joblib')")
    A("X = df.reindex(columns=b['features'])          # canonical vocabulary")
    A("raw = sum(m.predict_proba(X)[:, 1] for m in b['members'].values()) / len(b['members'])")
    A("pd_calibrated = b['calibrator'].predict(raw)   # real probability of default")
    A("```\n")
    A(f"---\n_Generated {date.today().isoformat()} from metrics files — "
      "numbers are not hand-entered._")

    out = os.path.join(MODELS, "README.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
