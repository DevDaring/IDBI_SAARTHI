# SAARTHI — MSME Loan Default Early-Warning System

> *सारथी — the charioteer who guides.*
> **Predict who defaults, see it 12 months early, say why in the same words for every loan, and show the one move that fixes it.**

### 🔗 Live app: **https://koushikdeb.duckdns.org**

Open the link from any device — no install, no login. Drop in a loan CSV (or use
the built-in samples) and watch the whole pipeline run. A guided walkthrough lives
at **https://koushikdeb.duckdns.org/how-it-works**.

---

## 🏆 The Hackathon — IDBI Innovate 2026, Track 04

SAARTHI is built for **IDBI Innovate 2026 (Track 04 — MSME credit-default early
warning)**. The core of the problem statement is not just *predicting* default —
it is producing **trustworthy, explained output that a credit officer can act on**,
using a **common interpretation framework** so every loan is described the same way.

### How SAARTHI adheres to the problem statement

| Requirement | How SAARTHI meets it |
| --- | --- |
| **Predict MSME default** | A calibrated LightGBM model outputs a real probability of default (PD) per loan, with AUC / PR-AUC / Brier / ECE reported. On the SBA book it scores AUC ≈ 0.97. |
| **See it early (12 months)** | A per-loan 12-month survival curve shows *when* risk builds and flags the onset month + lead time — not one static number. |
| **A "common interpretation framework"** | Every loan is explained in a **fixed 10-code reason taxonomy** (LIQUIDITY_STRESS, REPAYMENT_HISTORY_POOR, …), so any two loans are directly comparable. |
| **Trustworthy, not hallucinated** | The **ML model owns the number**; the LLM only writes words. A **faithfulness judge** (a *different* model family) verifies every explanation against the model's own SHAP evidence and badges it ✓ Verified. |
| **Actionable** | A **counterfactual recourse** engine finds the smallest realistic change (tenure, collateral, working capital) and shows *before-PD → action → after-PD*. |
| **Fair & responsible** | Protected attributes (gender, region, community) are **never** model inputs — used only for a difference-aware fairness audit that flags bias while sparing legitimate risk. |
| **Honest** | Estimated curves are labelled *estimated*; failed explanations *degrade* (never crash); a **model trace** shows exactly which provider/model wrote each explanation and whether its JSON was clean or repaired. |
| **Works on real data** | Validated end-to-end on real Kaggle/UCI books — SBA (899k rows), German Credit, Taiwan Default, plus a synthetic MSME demo set. |

**Non-negotiables we hold to:** the ML model outputs the probability; the LLM only
writes reasons/prose/actions; protected attributes are audit-only; every LLM JSON
response is schema-validated (or degrades); every run is reproducible.

---

## ✨ Functionalities

**Data & mapping**
- Drag-and-drop upload of `.csv` / `.xlsx` / `.parquet`; fast column profiling with polars (dtype, null %, sample values, uniqueness).
- **AI column auto-mapping** onto a fixed canonical loan schema, with a deterministic heuristic seed + fallback. Auto-detects the **target** (robust binarisation of `0/1`, `1/2`, `CHGOFF/P I F`, `default/paid`, …) and **protected** fields. Fully user-editable before running.

**Prediction**
- **Calibrated PD** per loan via LightGBM (TabPFN used automatically on small data if installed), **isotonic calibration** (`FrozenEstimator`), class-imbalance handling, train-sample cap with full-book scoring in batches.
- Portfolio metrics: **AUC-ROC, PR-AUC, Brier, ECE** (calibration error).

**Early warning**
- **12-month cumulative-risk curve** per loan — a lifelines **Cox** model shapes the timing when a time/vintage column exists, otherwise a Weibull-estimated curve (clearly labelled). The curve is anchored so its 12-month value equals the PD.
- **Alert**: the month risk crosses the threshold + the lead time it buys.

**Explanation (the trust story)**
- **SHAP** per loan → mapped onto a **fixed 10-code reason taxonomy** with signed contributions and direction.
- **Grounded LLM explanation**: the LLM writes 2–3 plain-English sentences using *only* the model's real drivers — it cannot invent factors, flip a direction, assert trends, or output a probability.
- **Faithfulness judge** (anti-hallucination): a different model family checks the prose against the SHAP evidence, backed by a deterministic structural check; it regenerates once on failure and badges the result ✓ Verified.
- **Consensus judge** (optional toggle): two model families produce and a third adjudicates, for high-risk loans.

**Action & fairness**
- **Counterfactual recourse**: greedy minimal-change search over *actionable* features → grounded projected post-action PD (before → after).
- **Difference-aware fairness audit** (fairlearn): demographic-parity & equalized-odds gaps + a within-risk-band residual, flagging only disparities that persist among same-risk applicants.

**Experience**
- **Portfolio "war-room" dashboard**: metric cards, risk-distribution donut, sortable/paginated risk table, fairness badges, plain-English run notes.
- **Per-loan drill-down**: risk curve with onset marker, ranked reason-code chips with evidence, hero explanation + verified badge, before→after recourse card, fairness badge, and a **model trace**.
- **"How it works"** explainer page with graphviz/matplotlib diagrams; **hover overlays** on each processing stage; a **Settings** drawer (resolved model-per-role, consensus toggle).

**Reliability engineering**
- **Multi-provider LLM gateway**: per-provider key rotation, exponential backoff, role-based provider fallback, per-call timeouts, startup **health-check** with **model-id self-correction**, auto-disable of metered/empty providers, and a secret-free call trace.
- **5-layer JSON-safe pipeline**: strict JSON mode → direct parse → cleanup → `json-repair` → judge repair → pydantic validation → graceful *degrade* (never crash).
- In-process **async job manager** (ThreadPoolExecutor) with live progress; top-K explanations precomputed in parallel, the rest built lazily and cached.
- Deterministic, fully-logged orchestrator (`RANDOM_SEED`) → reproducible runs.

---

## 🛠️ Technologies used

| Layer | Stack |
| --- | --- |
| **Frontend** | React 18, Vite, TypeScript, Tailwind CSS v3, `recharts` (charts), `axios`, `react-router-dom` |
| **Backend** | Python 3.11, Flask 3 + Flask-CORS, `gunicorn` (gthread), in-process `ThreadPoolExecutor` jobs |
| **ML** | LightGBM, scikit-learn (isotonic calibration + `FrozenEstimator`), SHAP (TreeExplainer), lifelines (Cox PH), fairlearn (MetricFrame), pandas, NumPy, polars, pyarrow; TabPFN + DiCE optional/best-effort |
| **Validation** | pydantic v2 (every JSON contract), `json-repair` |
| **LLM providers** | `openai` SDK as one OpenAI-compatible client → **DeepSeek** (primary), **Mistral** (fallback), **OpenRouter** (gpt-4o-mini / llama-3.1-8b / gemma-3-4b), **Google Gemini** 2.5-flash-lite (diverse judge); NanoGPT optional |
| **Diagrams** | graphviz (flowcharts) + matplotlib (charts) → static SVG |
| **Hosting / DevOps** | **Caddy** reverse proxy with **automatic HTTPS** (ACME → ZeroSSL/Let's Encrypt), **systemd** services (auto-restart on boot), **DuckDNS** free subdomain, Ubuntu VPS |
| **Datasets** | Kaggle: SBA, Berka, Lending Club, Kiva · UCI: German Credit (144), Taiwan Default (350) |

---

## 🧩 Architecture

```
                 ┌──────── React + Vite + TS + Tailwind (SPA) ─────────┐
 Open the link ▶ │ Upload → Mapping → Processing → Dashboard → Loan     │
                 │ + "How it works" explainer, live stage overlays      │
                 └────────────────────── /api ─────────────────────────┘
                                          │ (same origin)
     Internet ▶ Caddy :443 (trusted TLS) ▶ gunicorn 127.0.0.1:8080
                                          │
                 ┌──────────────── Flask API ──────────────────────────┐
                 │ ingest → map → features → train+score → survival →   │
                 │        explain → judge → recourse → fairness → assemble
                 │                                                       │
                 │  LLM Gateway (DeepSeek ▸ Mistral ▸ OpenRouter ▸ Gemini)
                 │   • key rotation · backoff · provider fallback        │
                 │   • 5-layer JSON-safe parsing + judge repair          │
                 │   • startup health-check & model-id self-correction   │
                 └───────────────────────────────────────────────────────┘
```

### Backend modules
| File | Role |
| --- | --- |
| `llm/gateway.py` | One OpenAI-compatible client per (provider, key); rotation, backoff, timeouts, health-check + model-id self-correction, secret-free trace. |
| `llm/routes.py` | Role → ordered provider/model fallback chain (mapper, explainer, json_judge, faithfulness_judge, consensus). |
| `llm/json_safe.py` | 5-layer JSON pipeline → pydantic validation, else *degraded*. |
| `pipeline/ingest.py` | Fast CSV profiling (polars). |
| `pipeline/mapper.py` | LLM column → canonical schema, heuristic seed + fallback, binary-target detection. |
| `pipeline/features.py` | Robust target binarisation, leakage-safe encoding, **protected attrs excluded**. |
| `pipeline/model.py` | LightGBM/TabPFN router + isotonic calibration + AUC/PR-AUC/Brier/ECE. |
| `pipeline/survival.py` | 12-month curve; Cox timing when possible, else Weibull-estimated (labelled), anchored to PD. |
| `pipeline/explain.py` | SHAP → fixed reason-code taxonomy → grounded explainer LLM. |
| `pipeline/judges.py` | Faithfulness judge (+ deterministic pre-check) + optional consensus judge. |
| `pipeline/recourse.py` | Counterfactual minimal-change search → projected post-action PD. |
| `pipeline/fairness.py` | Difference-aware fairlearn audit (DP/EO gaps + within-band residual). |
| `pipeline/orchestrator.py` | Runs every stage, emits progress, parallel eager explanations + lazy build. |

### API endpoints
`POST /api/upload` · `POST /api/map` · `POST /api/run` (async job) ·
`GET /api/status/<job>` · `GET /api/results/<job>` · `GET /api/loan/<job>/<id>` ·
`GET /api/models` · `GET /api/health`

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

**Keys / `.env`** — all LLM keys are read **server-side only** (the frontend holds
none). `config.py` auto-detects the existing key names (`DEEPSEEK_API_KEY_1`,
`MISTRAL_API_KEY1`, `OPENROUTER_API_KEY_1`, `GEMINI_API_KEY_*`, …). At startup the
gateway pings each provider, lists its real models, and **self-corrects** renamed
ids (e.g. `deepseek-chat` → `deepseek-v4-flash`); providers with no balance are
auto-disabled.

### Hosting (how the live site is served)
gunicorn serves the built frontend + API on `127.0.0.1:8080`; **Caddy** fronts it
on `:443` with an auto-renewing trusted certificate; both run as **systemd**
services that restart on boot. Details in `Host/saarthi/server_setup.md`.

---

## ✅ Tests

```bash
source ../.venv/bin/activate
python scripts/smoke_test.py   # end-to-end acceptance (33 checks)
python scripts/http_test.py    # live HTTP surface (12 checks)
```
Covers: mapper finds the target, model trains with AUC, a high-PD loan shows reason
codes + action + fairness with a *faithful* explanation, the fairness audit yields
a non-trivial disparity, the 5-layer JSON pipeline recovers malformed JSON, and
disabling DeepSeek falls back to Mistral automatically. Validated on real **SBA**
(AUC ≈ 0.97, leakage-free) and **German Credit** books.

---

## 📊 Datasets

`scripts/download_data.py` fetches training data into a `Dataset/` folder
(Kaggle: SBA, Berka, Lending Club, Kiva; UCI: German, Taiwan). Two ready-to-run
synthetic books ship in `data/` (`msme_demo.csv`, `credit_applicants.csv`).

---

## 🔬 For the paper

Deterministic, fully-logged runs (`RANDOM_SEED`). Novel, write-up-worthy pieces:
- **Capability-diverse judge panel** for tabular-explanation faithfulness — a producer model audited by a *different* family against SHAP ground truth, with a deterministic structural gate that cuts false positives.
- **Grounded LLM explanation** — the LLM is confined to a fixed taxonomy and may not author magnitudes or signs, decoupling *what* the model found from *how* it is phrased.
- **Difference-aware fairness** — flag only disparities that persist within a predicted-risk band.
- **Consistent survival** — the 12-month curve's endpoint equals the calibrated PD; time-to-event data shapes *timing*, never the level.

---

## ⚖️ Honest limitations
- Runs are capped at `SCORE_ROW_CAP` (150k) rows in memory for a snappy demo; raise it in `.env` for full-book scoring.
- Per-loan LLM explanations are precomputed for the top-K riskiest loans and built lazily (cached) for others.
- TabPFN / DiCE are best-effort; the reliable defaults are LightGBM + a greedy counterfactual search.
- On very sparse loans (mostly-missing rows) the recourse "after" figure can look optimistic — the model is extrapolating from imputed values.

---

<p align="center"><b>SAARTHI · IDBI Innovate 2026 · Track 04</b><br/>
<i>Predict who defaults, see it 12 months early, and show the one move that fixes it.</i></p>
