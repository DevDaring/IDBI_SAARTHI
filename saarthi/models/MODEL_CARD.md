---
license: apache-2.0
tags:
  - credit-risk
  - default-prediction
  - tabular
  - msme
  - lightgbm
library_name: joblib
---

# SAARTHI — MSME Default Prediction

Credit default-prediction models trained **entirely on public datasets** for IDBI Innovate 2026 (Track 04, Default Prediction Model). No bank data was used at any point.

The headline design choice: SAARTHI ships **pre-trained** and then fine-tunes on a lender's own book, rather than cold-starting on every upload.

## Methodology — why these numbers are trustworthy

Every metric below comes from a **strict three-way split**:

| Fold | Share | Used for |
|---|---|---|
| fit | 60% | training the boosters |
| calibrate | 15% | fitting the isotonic calibrator **only** |
| test | 25% | never seen by either — all reported metrics |

This matters: a common shortcut fits the probability calibrator and then measures calibration error on that *same* fold, which drives ECE artificially toward zero. Here the calibrator never sees the test fold, so the reported ECE is a real out-of-sample calibration estimate.

## Per-dataset specialist models

Full native feature set per corpus. Ensemble of LightGBM + XGBoost + CatBoost, isotonic-calibrated.

| Dataset | n | Default rate | Test AUC | PR-AUC | ECE | Published reference |
|---|---:|---:|---:|---:|---:|---|
| `sba` | 897,167 | 0.1756 | **0.9800** | 0.9126 | 0.0015 | ~0.95 (literature) |
| `lending_club` | 1,369,566 | 0.2124 | **0.7263** | 0.4104 | 0.0024 | ~0.70-0.73 (leakage-free) |
| `home_credit` | 307,511 | 0.0807 | **0.7632** | 0.2449 | 0.0011 | 0.805 (Kaggle winner) |
| `gmsc` | 150,000 | 0.0668 | **0.8522** | 0.3629 | 0.0041 | ~0.87 (Kaggle winner) |
| `taiwan` | 30,000 | 0.2212 | **0.7698** | 0.5243 | 0.0074 | ~0.78 (literature) |
| `german` | 1,000 | 0.3000 | **0.7668** | 0.5505 | 0.0680 | ~0.79 (literature) |
| `berka` | 682 | 0.1111 | **0.8558** | 0.5725 | 0.0295 | n/a |
| `amex` | 120,000 | 0.2598 | **0.9580** | 0.8858 | 0.0050 | ~0.96 AUC (winners; the 0.80 headline is a different metric) |
| hc2024 | — | — | _failed_ | — | — | ~0.86 (Kaggle winner) |

- `sba` — US Small Business Administration loans - closest public analogue to MSME lending
- `lending_club` — 2007-2018 consumer loans, charge-off label
- `home_credit` — Home Credit 2018 application + bureau aggregates
- `gmsc` — Give Me Some Credit, 90+ DPD within 2 years
- `taiwan` — Taiwan credit-card default, 6-month repayment panel
- `german` — UCI German Credit, 1000 rows
- `berka` — PKDD'99 Czech bank, pre-origination transaction aggregates
- `amex` — American Express monthly statement panel
- `hc2024` — Home Credit 2024 model-stability competition

## Pooled global model (ships with the app)

Trained across corpora in a shared 15-field canonical credit vocabulary so it can score any loan book that maps onto it.

- **Test AUC:** 0.8605
- **PR-AUC:** 0.5795
- **ECE:** 0.0018  ·  **Brier:** 0.0868
- **Test rows:** 270,421
- **Corpora pooled:** sba, lending_club, home_credit, gmsc, taiwan, german, berka

## Leave-one-dataset-out transfer — a negative result

Train on every corpus *except* one, then score the held-out corpus cold. This is the honest proxy for _"will it transfer to a book it has never seen?"_ — the question that actually matters for deployment.

**It does not transfer.** Most hold-outs land at or *below* chance, which means the pooled model is not merely uninformative on an unseen corpus, it is anti-predictive: the feature→outcome relationships invert across lending domains.

| Held-out corpus | n | Raw pooling | Rank-normalised |
|---|---:|---:|---:|
| `berka` | 682 | 0.4612 | 0.5155 |
| `german` | 1,000 | 0.4646 | 0.4022 |
| `gmsc` | 150,000 | 0.7786 | 0.7711 |
| `home_credit` | 300,000 | 0.4991 | 0.5785 |
| `lending_club` | 300,000 | 0.5228 | 0.5399 |
| `sba` | 300,000 | 0.4103 | 0.2574 |
| `taiwan` | 30,000 | 0.7038 | 0.6989 |
| **mean** | | **0.5486** | **0.5377** |

### Normalisation did NOT rescue it

The obvious hypothesis is scale mismatch — these corpora are denominated in USD, DM, NT$ and CZK, so a "loan amount" of 50,000 means different things in each. Converting every numeric feature to its within-corpus percentile rank tests that hypothesis directly, and **it fails**: mean transfer AUC moved 0.5486 → 0.5377, i.e. no better. SBA in particular degrades from 0.4103 to 0.2574.

The mechanism is therefore **relational inversion, not scale**. SBA is small-business lending, where a longer term and a larger SBA-guaranteed principal typically indicate a better-vetted, collateral-backed loan — the opposite of the consumer-credit corpora that dominate the pooled training set. Rank-normalising preserves that inverted ordering perfectly, which is why it cannot help.

**Practical consequence:** a single pooled "foundation" model for credit risk is not supported by this evidence. Domain-matched training plus fine-tuning on the lender's own book is the defensible architecture, which is what SAARTHI does.

## Ablation — does transaction-as-language earn its place?

A CoLES-style contrastive encoder (GRU + InfoNCE over disjoint sub-sequence views) was pre-trained on unlabelled transaction streams, then its 256-d embedding was tested against the tabular features.

| Corpus | Tabular | Sequence only | Tabular + sequence | Lift |
|---|---:|---:|---:|---:|
| `berka` | 0.8575 | 0.8097 | 0.8842 | +0.0267 |
| `amex` | 0.9590 | 0.9156 | 0.9585 | -0.0006 |

**When the sequence encoder earns its place:** on Berka (16 tabular features, raw bank transactions) it adds **+0.027 AUC**. On Amex (941 hand-engineered aggregates over the *same* statement data) it adds nothing — the aggregates already capture what the encoder learns. The embedding substitutes for feature engineering rather than adding to it; it is most valuable exactly where a lender has raw transaction streams but few curated fields, which is the realistic MSME case.

Note also that on Amex the 256-d embedding **alone** reaches 0.9156 AUC with no hand-engineered features at all, versus 0.9590 for 941 engineered columns.

### Linear probe on frozen embeddings

Logistic regression on the frozen encoder output — measures how much default signal the *unsupervised* embedding captured on its own.

- `amex`: AUC 0.8717 (n=60,000)
- `berka`: AUC 0.7783 (n=682)

## Known limitations

- **Berka** aggregates only *pre-origination* transactions. Using the full history yields AUC 1.0000 because 71% of an account's transactions occur after the loan date and encode the repayment behaviour that defines the label. The leaked figure is not reported here.
- **Amex** and **Home Credit** are Kaggle-competition datasets; they are used for research validation. The shipped global model is trained on the permissively-licensed core corpora.
- The pooled model uses a deliberately small shared vocabulary, so its AUC is *lower* than the specialists by construction. Its purpose is transfer, not peak in-corpus accuracy.
- No Indian MSME data was available; GST-behavioural fields are simulated against the schema IDBI published, not learned from real filings.

## Files

| File | What it is |
|---|---|
| `global_canonical.joblib` | pooled global model + isotonic calibrator |
| `specialist_<corpus>.joblib` | per-corpus specialist ensembles |
| `coles.pt` | transaction-sequence encoder checkpoint |
| `metrics.json` | every number above, machine-readable |
| `ablation_sequence.json` | sequence-embedding ablation |

## Usage

```python
import joblib, pandas as pd
b = joblib.load('global_canonical.joblib')
X = df.reindex(columns=b['features'])          # canonical vocabulary
raw = sum(m.predict_proba(X)[:, 1] for m in b['members'].values()) / len(b['members'])
pd_calibrated = b['calibrator'].predict(raw)   # real probability of default
```

---
_Generated 2026-08-31 from metrics files — numbers are not hand-entered._