# SAARTHI — MSME Loan Default Early-Warning System

> *सारथी — the charioteer who guides.*
> **Predict who defaults, see it 12 months early, say why in the same words for every loan, and show the one move that fixes it.**

### 🔗 Live app: **https://koushikdeb.duckdns.org**
### 🤗 Trained models: **https://huggingface.co/Debk/saarthi-default-prediction**

Open the link from any device — no install, no login. Drop in a loan CSV (or use
the built-in samples) and watch the whole pipeline run. A guided walkthrough lives
at **https://koushikdeb.duckdns.org/how-it-works**.

Built for **IDBI Innovate 2026 (Track 04 — Default Prediction Model)** by
**Team SAARTHI** (Koushik Deb, solo). **No bank data was used at any stage** —
every model here is trained on public research datasets.

---

## What it does, in one paragraph

For every borrower SAARTHI answers three questions: **who is likely to default**,
**how soon** (a 12-month curve, not one static number), and **the single change
that measurably lowers that risk**. The design rule is a hard separation of
duties: **the ML model owns every number** (probability of default, SHAP drivers,
the projected effect of any remedy), while **the LLM owns only words** — it picks
a code from a fixed 10-item taxonomy and writes the prose. A **second model from a
different family** then audits that prose against the model's own SHAP evidence,
so nothing reaching a credit officer is invented.

---

## 🆕 What changed in the second prototype phase

Phase 1 was an excellent adaptive tool that trained on whatever you uploaded.
Phase 2 turned it into a **trained model** — which is what the problem statement
actually asks for — and, in the process, found and fixed three things that were
quietly wrong.

| | |
| --- | --- |
| **Trained a real model** | 3.3M loans across nine public corpora. The app now **arrives pre-trained** and feeds that prior into training on the lender's own book, instead of cold-starting on every upload. |
| **Fixed a calibration-evaluation flaw** | The old code fitted the isotonic calibrator and then measured ECE **on the same fold**. Isotonic is flexible enough to drive that number to ~0 on data it has already seen. Now a strict 60/15/25 **fit / calibrate / test** split. |
| **Found and removed a data leak** | On the Berka corpus, **71% of an account's transactions post-date the loan** and encode the outcome. Naive aggregation scored a meaningless **AUC 1.0000**. Restricted to pre-origination activity, the honest figure is **0.8558**. |
| **Built transaction-as-language** | A self-supervised contrastive encoder (CoLES-style) trained on **13.25M transaction events**. On a frozen embedding alone, a linear probe reaches **0.8717 AUC** on Amex. |
| **Measured instead of assuming** | Ensembling correlated boosters was tested, found to add **nothing** (−0.0029 AUC), and reported as such. |

The calibration fix is visible in the live app. On the same 3,000-loan demo book:

| | AUC | ECE |
| --- | ---: | ---: |
| before (calibrator scored on its own fold) | 0.799 | **0.000** ← not real |
| after (held-out test fold) | 0.782 | **0.070** ← honest |

A side effect worth noting: with an honest split the **fairness audit now actually
fires** (region flagged for review at 14.1% equalised-odds gap) where previously
everything passed.

---

## 📊 Results

All figures come from a fold that **neither the model nor the calibrator has seen**.

### Per-dataset specialist models

| Dataset | n (test) | Test AUC | ECE | Published reference |
| --- | ---: | ---: | ---: | --- |
| **SBA** (US small business — the MSME analogue) | 224,292 | **0.9800** | 0.0015 | ~0.95 |
| **Amex** (statement panel) | 30,000 | **0.9580** | 0.0050 | ~0.96 AUC |
| Berka (pre-origination only) | 171 | 0.8558 | 0.0295 | — |
| Give Me Some Credit | 37,500 | 0.8522 | 0.0041 | ~0.87 |
| Taiwan Default | 7,500 | 0.7698 | 0.0074 | ~0.78 |
| German Credit | 250 | 0.7668 | 0.0680 | ~0.79 |
| Home Credit 2018 | 76,879 | 0.7632 | 0.0011 | 0.805 |
| Lending Club | 342,392 | 0.7263 | 0.0024 | 0.70–0.73 |
| **Pooled global model** (ships with the app) | 270,421 | 0.8605 | 0.0018 | — |

We beat the published reference on SBA, match it on Amex / Lending Club / Taiwan,
and sit below it on Home Credit (winners used all seven relational tables; we used
`application_train` plus light bureau aggregates), GMSC and German.

### Two negative results worth more than the AUCs

**1. Cross-domain transfer fails — and normalisation cannot fix it.**
Leave-one-dataset-out, training on every corpus except one:

| Held-out corpus | Raw pooling | Rank-normalised |
| --- | ---: | ---: |
| SBA | 0.4103 | 0.2574 |
| Berka | 0.4612 | 0.5155 |
| German | 0.4646 | 0.4022 |
| Home Credit | 0.4991 | 0.5785 |
| Lending Club | 0.5228 | 0.5399 |
| Taiwan | 0.7038 | 0.6989 |
| GMSC | 0.7786 | 0.7711 |
| **mean** | **0.5486** | **0.5377** |

Below 0.5 is not "uninformative" — it is **anti-predictive**. The obvious
hypothesis is scale mismatch (these corpora are denominated in USD, DM, NT$ and
CZK), so we converted every numeric feature to its within-corpus percentile rank.
**It did not help.** The mechanism is **relational inversion, not scale**: in
small-business lending a longer term and a larger guaranteed principal signal a
better-vetted, collateral-backed loan — the opposite of the consumer corpora that
dominate the pool. Rank-normalising preserves that inverted ordering exactly,
which is why it cannot rescue it.

*Consequence:* a single pooled "foundation model" for credit risk is **not
supported by this evidence**. Domain-matched training plus fine-tuning on the
lender's own book is the defensible architecture — which is what SAARTHI does.

**2. The sequence encoder substitutes for feature engineering rather than adding to it.**

| Corpus | Tabular features | Tabular | Sequence only | Both | Lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Berka | 16 | 0.8575 | 0.8097 | **0.8842** | **+0.0267** |
| Amex | 941 | 0.9590 | 0.9156 | 0.9585 | −0.0006 |

With 16 curated fields the embedding adds real signal; with 941 hand-engineered
aggregates over the same source it adds nothing. Most valuable exactly where a
lender has raw transaction streams but few curated fields — the realistic MSME
case, and precisely what IDBI's statement API provides.

**3. Ensembling correlated boosters is a wash.** Mean gain of LightGBM + XGBoost +
CatBoost over the best single member: **−0.0029 AUC**, worst on small corpora
(German −0.0155). Diversity of *inductive bias* pays; more GBDTs do not.

---

## ✨ Functionalities

**Data & mapping**
- Drag-and-drop `.csv` / `.xlsx` / `.parquet`; fast column profiling with polars.
- **AI column auto-mapping** onto a fixed canonical schema, with a deterministic heuristic seed + fallback. Robust target binarisation (`0/1`, `1/2`, `CHGOFF/P I F`, `default/paid`). Fully user-editable before running.

**Prediction**
- **Pre-trained global prior** — a pooled model trained offline on 3.3M public loans is loaded at startup and fed in as an extra feature, so a bank's book *refines* prior knowledge instead of starting from zero. Skipped automatically when field coverage is too low.
- **Calibrated PD** via LightGBM with isotonic calibration on a dedicated fold.
- Portfolio metrics — **AUC-ROC, PR-AUC, Brier, ECE** — all from a held-out test fold.

**Early warning**
- **12-month cumulative-risk curve** per loan. A lifelines **Cox** model shapes the timing when a time/vintage column exists, otherwise a Weibull-estimated curve (clearly labelled). The curve is anchored so its 12-month value equals the PD.
- **Alert**: the month risk crosses threshold, and the lead time it buys.

**Explanation (the trust story)**
- **SHAP** per loan → a **fixed 10-code reason taxonomy** with signed contributions and direction.
- **Grounded LLM explanation** — the LLM writes 2–3 plain sentences using *only* the model's real drivers. It cannot invent factors, flip a direction, assert trends, or output a probability of its own.
- **Faithfulness judge** — a different model family checks the prose against the SHAP evidence, backed by a deterministic structural gate; regenerates once on failure and badges ✓ Verified.
- **Consensus judge** (optional) — two families produce, a third adjudicates, for high-risk loans.

**Action & fairness**
- **Counterfactual recourse** — greedy minimal-change search over *actionable* levers only (never protected attributes or fixed history) → grounded projected post-action PD.
- **Difference-aware fairness audit** (fairlearn) — flags a disparity only when it persists among **same-risk** applicants.

**Reliability engineering**
- **Multi-provider LLM gateway** — key rotation, exponential backoff, role-based provider fallback, per-call timeouts, startup health-check with **model-id self-correction**, auto-disable of metered/empty providers, secret-free call trace.
- **5-layer JSON-safe pipeline** — strict JSON mode → direct parse → cleanup → `json-repair` → judge repair → pydantic validation → graceful *degrade* (never crash).
- In-process async job manager with live progress; top-K explanations precomputed in parallel, the rest lazy and cached.

---

## 🧩 Architecture

```
                 ┌──────── React + Vite + TS + Tailwind (SPA) ─────────┐
 Open the link ▶ │ Upload → Mapping → Processing → Dashboard → Loan    │
                 └────────────────────── /api ─────────────────────────┘
                                          │ (same origin)
     Internet ▶ Caddy :443 (trusted TLS) ▶ gunicorn 127.0.0.1:8080
                                          │
                 ┌──────────────── Flask API ──────────────────────────┐
                 │ ingest → map → features → [+global prior] →         │
                 │ train+score → survival → explain → judge →          │
                 │ recourse → fairness → assemble                      │
                 │                                                     │
                 │  LLM Gateway (DeepSeek ▸ Mistral ▸ OpenRouter ▸     │
                 │  Gemini) · rotation · fallback · JSON repair        │
                 └─────────────────────────────────────────────────────┘

   offline ▶ training/  →  models/global_canonical.joblib  →  loaded at startup
                        →  models/coles.pt (transaction encoder)
```

### Backend modules
| File | Role |
| --- | --- |
| `llm/gateway.py` | One OpenAI-compatible client per (provider, key); rotation, backoff, health-check + model-id self-correction, secret-free trace. |
| `llm/routes.py` | Role → ordered provider/model fallback chain. Enforces judge-family diversity. |
| `llm/json_safe.py` | 5-layer JSON pipeline → pydantic validation, else *degraded*. |
| `pipeline/ingest.py` | Fast CSV profiling (polars). |
| `pipeline/mapper.py` | LLM column → canonical schema, heuristic seed + fallback. |
| `pipeline/features.py` | Robust target binarisation, leakage-safe encoding, **protected attrs excluded**. |
| `pipeline/pretrained.py` | Loads the offline-trained global model; projects an upload onto its canonical vocabulary and scores it. |
| `pipeline/model.py` | LightGBM + **3-way fit/calibrate/test split** + isotonic calibration + AUC/PR-AUC/Brier/ECE. |
| `pipeline/survival.py` | 12-month curve; Cox timing when possible, else Weibull-estimated (labelled), anchored to PD. |
| `pipeline/explain.py` | SHAP → fixed reason-code taxonomy → grounded explainer LLM. |
| `pipeline/judges.py` | Faithfulness judge (+ deterministic pre-check) + optional consensus judge. |
| `pipeline/recourse.py` | Counterfactual minimal-change search → projected post-action PD. |
| `pipeline/fairness.py` | Difference-aware fairlearn audit. |
| `pipeline/orchestrator.py` | Runs every stage, emits progress, parallel eager explanations + lazy build. |

### Training pipeline (`saarthi/training/`)
| File | Role |
| --- | --- |
| `adapters.py` | Nine public corpora → **native** (full per-corpus features) and **canonical** (shared 15-field vocabulary) feature spaces. |
| `train_global.py` | Strict 3-way split, GBDT ensemble, isotonic calibration, leave-one-dataset-out transfer. |
| `seq_data.py` | Raw transactions → token sequences. TabFormer's deep-but-narrow histories are chunked into windows: 2,034 → **94,980** entities. |
| `train_coles.py` | Self-contained CoLES encoder (GRU + InfoNCE over disjoint sub-sequence views). Stages: pretrain / embed / probe. |
| `integrate_seq.py` | Ablation — tabular vs sequence vs both. |
| `rerun_berka.py`, `rerun_pool.py` | Leakage fix and normalisation re-runs, patching `metrics.json`. |
| `make_model_card.py` | Renders the HF model card **from the metrics files** — no number is ever typed by hand. |
| `hf_sync.py` | HuggingFace push/pull for models and derived sequences. |

### API endpoints
`POST /api/upload` · `POST /api/map` · `POST /api/run` (async job) ·
`GET /api/status/<job>` · `GET /api/results/<job>` · `GET /api/loan/<job>/<id>` ·
`GET /api/models` · `GET /api/pretrained` · `GET /api/health`

---

## 🛠️ Technologies used

| Layer | Stack |
| --- | --- |
| **Frontend** | React 18, Vite, TypeScript, Tailwind CSS v3, `recharts`, `axios`, `react-router-dom` |
| **Backend** | Python 3.11, Flask 3 + Flask-CORS, `gunicorn` (gthread), in-process `ThreadPoolExecutor` jobs |
| **ML** | LightGBM, XGBoost, CatBoost, scikit-learn (isotonic calibration), SHAP (TreeExplainer), lifelines (Cox PH), fairlearn, PyTorch (CoLES encoder), pandas, NumPy, polars, pyarrow |
| **Validation** | pydantic v2 (every JSON contract), `json-repair` |
| **LLM providers** | `openai` SDK as one OpenAI-compatible client → **DeepSeek** (primary), **Mistral** (fallback), **OpenRouter**, **Google Gemini** (diverse judge). AWS Bedrock adapter planned for in-VPC deployment. |
| **Tooling** | Playwright (end-to-end screenshot capture), python-pptx (deck automation), vast.ai (GPU), HuggingFace Hub |
| **Hosting / DevOps** | **Caddy** with **automatic HTTPS**, **systemd** services, **DuckDNS**, Ubuntu VPS |

---

## 📚 Datasets

`saarthi/scripts/download_data.py` fetches training data into `Dataset/`
(~15 GB, gitignored). Eleven corpora:

| Corpus | Scale | Role |
| --- | --- | --- |
| SBA National | 899K loans | MSME anchor |
| Lending Club | 2.26M accepted | temporal validation |
| Home Credit 2018 | 307K apps, 8 tables | multi-table joins |
| Home Credit 2024 | 1.53M cases, 33 tables | stability metric |
| Amex Default | 5.53M statements / 459K customers | behavioural panel |
| IBM TabFormer | 24.4M transactions | sequence pretraining |
| Give Me Some Credit | 150K | benchmark |
| Berka (PKDD'99) | 1.06M transactions, 682 loans | sequence → default |
| Taiwan / German / Kiva | 30K / 1K / 672K | benchmarks, text corpus |

Two ready-to-run synthetic books ship in `saarthi/data/`.

---

## 🚀 Run it locally

Prereqs: **Python 3.11** and **Node 20+**.

```bash
cd saarthi
scripts/setup.sh          # venv + backend deps + sample data + frontend deps
scripts/run_backend.sh    # http://localhost:5000
scripts/run_frontend.sh   # http://localhost:5173  (proxies /api → :5000)
```

Open **http://localhost:5173**, drop in `data/msme_demo.csv`, confirm the mapping,
**Run analysis**, then open the top-risk loan.

### Retrain the models
```bash
cd saarthi/training
python train_global.py                    # specialists + pooled + LODO
python seq_data.py berka amex tabformer   # build sequences
python train_coles.py --stage pretrain --epochs 40   # GPU, ~20 min
python integrate_seq.py                   # ablation
python make_model_card.py                 # regenerate the HF card
```

**Keys / `.env`** — all LLM keys are read **server-side only**. `config.py`
auto-detects existing key names. At startup the gateway pings each provider, lists
its real models, and self-corrects renamed ids; providers with no balance are
auto-disabled.

---

## ✅ Tests

```bash
source ../.venv/bin/activate
python scripts/smoke_test.py   # end-to-end acceptance (33 checks)
python scripts/http_test.py    # live HTTP surface (12 checks)
```

Covers: the mapper finds the target, the model trains with a real AUC, a high-PD
loan produces reason codes + action + fairness with a *faithful* explanation, the
JSON pipeline recovers malformed output, and disabling DeepSeek falls back to
Mistral automatically.

---

## 🔬 For the paper

Reproducible, deterministic runs (`RANDOM_SEED = 20260502`). The write-up-worthy
pieces are the **negative and methodological** results, not the AUCs:

- **Cross-domain transfer fails through relational inversion** — quantified across
  seven corpora, with the scale-mismatch explanation explicitly falsified.
- **Calibration-evaluation leakage** — fitting isotonic on the fold you then score
  drives ECE to ~0; demonstrated on the app's own demo book (0.000 → 0.070).
- **Label leakage in a widely-used dataset** — Berka gives AUC 1.0000 unless
  transaction aggregation is restricted to pre-origination activity.
- **Correlated ensembling is a wash** — −0.0029 AUC over the best single member.
- **Capability-diverse judge panel** — a producer model audited by a *different*
  family against SHAP ground truth, with a deterministic structural gate.

**Known gaps before submission:** single-seed results (needs ≥5 seeds with
mean ± std), no significance testing (DeLong / bootstrap CIs), no logistic
regression + WOE baseline, and no feature-distribution-shift analysis to explain
*why* transfer inverts.

---

## ⚖️ Honest limitations

- Runs are capped at `SCORE_ROW_CAP` (150k) rows in memory for a snappy demo.
- Per-loan LLM explanations are precomputed for the top-K riskiest loans and built lazily (cached) for the rest.
- **TabPFN is not installed**, so that branch of the model router never fires — LightGBM handles every run. The UI's "Trying TabPFN" progress message is aspirational.
- **DiCE is not used** despite the docstring in `recourse.py`; the greedy 1-D counterfactual search is the only implementation.
- The pooled global model transfers poorly to unseen domains (see above) — it is fed in as a *feature* the booster can learn to down-weight, never as a fixed offset.
- On very sparse loans the recourse "after" figure can look optimistic — the model is extrapolating from imputed values.
- Berka's honest test fold is 171 rows and German's 250; differences of ±0.015 there are within sampling noise.
- No Indian MSME data was available; GST-behavioural fields are modelled against IDBI's published schema, not learned from real filings.

---

<p align="center"><b>SAARTHI · IDBI Innovate 2026 · Track 04</b><br/>
<i>Predict who defaults, see it 12 months early, and show the one move that fixes it.</i></p>
