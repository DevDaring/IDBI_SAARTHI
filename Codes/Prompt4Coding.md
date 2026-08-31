
# SAARTHI — Full Build Prompt

> Paste this entire document into your coding agent (Claude Code / Cursor / etc.).
> Build it in the milestone order given in Section 13 so a working MVP exists early.
> Do not invent APIs. Where a model ID or endpoint is uncertain, call the provider's `GET /models` endpoint at startup and log what is available.

---

## 0. Your role

You are a senior full-stack engineer. Build a production-quality web application called **SAARTHI** — an MSME loan default early-warning system for the IDBI Innovate 2026 hackathon (Track 04). The user uploads a credit dataset (CSV), the app auto-maps the columns, trains a model, predicts default risk over the next 12 months, and shows every result **with a reason and a recommended action** in plain English. The hard requirement is **trustworthy, explained output** — never a bare number.

---

## 1. Product summary (what the user sees)

1. User drags in a CSV (e.g. a Kaggle credit dataset).
2. The app reads the columns and an LLM maps them to a fixed internal schema. The user can correct the mapping.
3. The app trains a default-prediction model on the data and scores every loan.
4. For each loan it produces: a 12-month risk curve, a probability of default, the **reasons** (in a fixed, comparable format), a **recommended action** to reduce the risk, and a **fairness flag**.
5. The dashboard shows a portfolio overview plus a per-loan drill-down. Every explanation is written in simple English and is checked by a second "judge" model so it cannot hallucinate.

Tagline to keep in mind: *predict who defaults, see it 12 months early, say why in the same words for every loan, and show the one move that fixes it.*

---

## 2. Tech stack (pinned)

- **Frontend:** React 18 + Vite, TypeScript, Tailwind CSS, `recharts` for charts, `axios` for API calls. No backend secrets ever reach the frontend.
- **Backend:** Python 3.11 + Flask + Flask-CORS. Use the `openai` Python SDK as the single HTTP client for every LLM provider (all providers below are OpenAI-compatible — only `base_url` and key change).
- **Agent/LLM orchestration:** LangChain is allowed but optional. Keep the core pipeline a plain, deterministic Python orchestrator (reliable and easy to debug). Use LLMs only for the steps in Section 6.
- **ML:** `lightgbm`, `scikit-learn`, `shap`, `lifelines` (survival), `dice-ml` (recourse, optional), `fairlearn` (fairness audit), `pandas`, `polars` (fast CSV), `numpy`, `pydantic` (schema validation), `json-repair` (malformed-JSON fallback).
- **Optional pretrained model:** `tabpfn` (a pretrained tabular foundation model) for small datasets. Treat it as best-effort: if it installs and the dataset is within its size limit, use it; otherwise fall back to LightGBM. LightGBM is the reliable default.
- **Jobs:** simple in-process job manager using `concurrent.futures.ThreadPoolExecutor` + an in-memory job store, with status polling. (Redis + RQ is an optional scale-up; do not require it.)

---

## 3. Environment variables (`.env`, backend only)

Read these in the backend. **Never expose them to React.** All keys already exist in the user's `.env`.

```
# DeepSeek — PRIMARY (2 keys, round-robin)
DEEPSEEK_API_KEY_1=
DEEPSEEK_API_KEY_2=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash          # primary worker model
DEEPSEEK_MODEL_PRO=deepseek-v4-pro        # stronger model for judges
# NOTE: legacy IDs deepseek-chat / deepseek-reasoner retire 2026-07-24.
# At startup, call GET {DEEPSEEK_BASE_URL}/models and verify the configured IDs exist;
# if not, log and fall back to whatever the endpoint lists.

# Mistral — FALLBACK (2 keys, round-robin)
MISTRAL_API_KEY_1=
MISTRAL_API_KEY_2=
MISTRAL_BASE_URL=https://api.mistral.ai/v1
MISTRAL_MODEL=mistral-large-latest

# NanoGPT — free models (subscription), OpenAI-compatible
NANOGPT_API_KEY=
NANOGPT_BASE_URL=https://nano-gpt.com/api/v1
NANOGPT_MODEL=                            # pick a free model id from GET /models at startup
NANOGPT_EMBED_MODEL=text-embedding-3-small  # NanoGPT exposes an embeddings endpoint

# OpenRouter — paid/strong models (2 keys, round-robin), OpenAI-compatible
OPENROUTER_API_KEY_1=
OPENROUTER_API_KEY_2=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=                         # set to a strong model id for the judge

# App
FLASK_PORT=5000
MAX_UPLOAD_MB=512
TRAIN_ROW_CAP=100000                      # sample above this for fast training
```

---

## 4. Repository structure

```
saarthi/
├── backend/
│   ├── app.py                 # Flask entry, routes, CORS
│   ├── config.py              # loads .env, model routes
│   ├── jobs.py                # ThreadPoolExecutor job manager + status store
│   ├── llm/
│   │   ├── gateway.py         # the LLM gateway (Section 6) — most important file
│   │   ├── routes.py          # role -> ordered provider/model fallback chain
│   │   └── json_safe.py       # 5-layer JSON parsing + judge repair (Section 6.3)
│   ├── pipeline/
│   │   ├── ingest.py          # read CSV, profile columns
│   │   ├── mapper.py          # LLM column mapping -> canonical schema
│   │   ├── features.py        # encode, impute, build matrix
│   │   ├── model.py           # TabPFN/LightGBM router + calibration + PD
│   │   ├── survival.py        # 12-month hazard curve
│   │   ├── explain.py         # SHAP -> causal reason codes -> LLM prose
│   │   ├── judges.py          # judge panel (Section 9)
│   │   ├── recourse.py        # counterfactual action
│   │   ├── fairness.py        # difference-aware audit
│   │   └── orchestrator.py    # runs all stages, emits progress
│   ├── schemas.py             # pydantic models for every JSON contract
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/ (Upload, Mapping, Processing, Dashboard, LoanDetail)
│   │   ├── components/ (RiskCurve, ReasonCodes, RecourseCard, FairnessBadge, ModelTrace, ...)
│   │   ├── api/client.ts
│   │   └── types.ts
│   └── package.json
└── README.md
```

---

## 5. Canonical loan schema (map every dataset to this)

The mapper's job is to map arbitrary uploaded columns onto these fields. All optional except the target, which is required for training.

| Canonical field         | Meaning                             | Used for                          |
| ----------------------- | ----------------------------------- | --------------------------------- |
| `loan_id`             | unique id (generate if absent)      | reference                         |
| `loan_amount`         | principal                           | risk                              |
| `term_months`         | loan tenure                         | risk + survival horizon           |
| `interest_rate`       | rate                                | risk                              |
| `income_or_turnover`  | borrower income / firm turnover     | risk                              |
| `dscr`                | debt service coverage               | risk                              |
| `credit_score`        | bureau score                        | risk                              |
| `sector`              | industry / MSME category            | risk                              |
| `collateral_value`    | security value                      | risk + recourse                   |
| `prior_delinquencies` | past late counts                    | risk                              |
| `employment_length`   | stability                           | risk                              |
| `text_purpose`        | loan purpose / description text     | text signal                       |
| `region`              | state / district                    | **protected — audit only** |
| `gender`              | proprietor gender                   | **protected — audit only** |
| `community`           | caste/community if present          | **protected — audit only** |
| `time_observed`       | months observed / vintage           | survival                          |
| `target`              | default / charge-off label (binary) | **required** for training   |

Protected fields are used **only** by the fairness audit and are **excluded** from model features.

---

## 6. LLM Gateway — the most important module (`llm/gateway.py`)

This wraps every LLM call. All four providers are OpenAI-compatible, so use one `openai.OpenAI(base_url=..., api_key=...)` client per (provider, key) and swap as needed.

### 6.1 Key rotation

For each provider keep a list of its keys and round-robin through them per call. DeepSeek: 2 keys. Mistral: 2 keys. OpenRouter: 2 keys. NanoGPT: 1 key.

### 6.2 Role-based fallback chain (`llm/routes.py`)

A single function `call_llm(role, messages, json=True)` resolves a **role** to an ordered list of `(provider, model)` attempts. On any failure (HTTP error, 429 rate limit, timeout, empty content, JSON failure that even the repair layer can't fix), move to the next entry. Apply exponential backoff (e.g. 1s, 2s, 4s) and rotate the key before switching provider.

```
ROUTES = {
  "mapper":             [deepseek(DEEPSEEK_MODEL), mistral(MISTRAL_MODEL), openrouter(OPENROUTER_MODEL)],
  "explainer":          [deepseek(DEEPSEEK_MODEL), mistral(MISTRAL_MODEL), nanogpt(NANOGPT_MODEL)],
  "json_judge":         [mistral(MISTRAL_MODEL), openrouter(OPENROUTER_MODEL), deepseek(DEEPSEEK_MODEL_PRO)],
  "faithfulness_judge": [deepseek(DEEPSEEK_MODEL_PRO), openrouter(OPENROUTER_MODEL), mistral(MISTRAL_MODEL)],
}
```

Rules: **DeepSeek is primary, Mistral is the fallback, NanoGPT free models for cheap bulk work, OpenRouter for the strong judge.** Use a *different* model family for a judge than the one that produced the output being judged (diversity matters).

### 6.3 JSON-safe calling (`llm/json_safe.py`) — required, do not skip

LLMs frequently return malformed JSON. Every JSON call passes through this 5-layer pipeline and only returns once it validates against the expected pydantic schema:

1. **Prompt for strict JSON.** Set `response_format={"type":"json_object"}` (DeepSeek, Mistral, OpenRouter, NanoGPT all support it). The prompt must contain the word "json", the exact schema, and one filled example.
2. **Direct parse** with `json.loads`.
3. **Cleanup parse.** Strip ```json fences, grab the first balanced `{...}` block, retry.
4. **Deterministic repair** with the `json-repair` library, then re-parse.
5. **Judge repair.** Send the raw broken text to the `json_judge` role: *"Convert the following into valid JSON matching this schema. Output JSON only."* Parse the judge's output.

After any layer succeeds, **validate against the pydantic schema**. If validation fails, run one more `json_judge` pass with the validation error included. If all layers fail, return a structured error object (never crash the pipeline) and mark that loan's explanation as `degraded`.

### 6.4 Security

All LLM calls happen in the backend. Frontend never holds a key. Log model id, provider, key index (not the key), latency, and finish reason for every call so the demo can show a transparent model trace.

---

## 7. Backend pipeline + Flask API

### 7.1 Pipeline stages (run by `orchestrator.py`, each emits a progress %)

`ingest → map → features → train+score → survival → explain → judge → recourse → fairness → assemble`

### 7.2 Endpoints

- `POST /api/upload` — multipart CSV. Saves file, profiles it. Returns `{upload_id, columns:[{name,dtype,sample[5],null_pct}], n_rows}`.
- `POST /api/map` — body `{upload_id}`. Runs the LLM mapper. Returns `{mapping:{canonical:source|null}, target, protected[], confidence, notes}`. The user may edit and resubmit.
- `POST /api/run` — body `{upload_id, mapping, target, protected[]}`. Starts the full pipeline as a background job. Returns `{job_id}`.
- `GET /api/status/{job_id}` — returns `{stage, percent, message, done, error?}`.
- `GET /api/results/{job_id}` — returns the **PortfolioResult** (Section 10).
- `GET /api/loan/{job_id}/{loan_id}` — returns the **LoanResult** (Section 10).
- `GET /api/models` — lists which provider/models resolved at startup (for the settings panel and demo).
- `GET /api/health`.

Enable CORS for the Vite dev origin. Reject non-CSV (allow `.csv`, optional `.xlsx`, `.parquet`). Enforce `MAX_UPLOAD_MB`.

---

## 8. ML engine

### 8.1 Mapping (`mapper.py`)

Give the LLM the column names, dtypes, null %, and 5 sample rows. It returns the canonical mapping JSON, the detected `target`, and detected `protected` fields. Validate: target must be roughly binary. Always let the user override in the UI before running.

### 8.2 Prediction model router (`model.py`)

- If rows ≤ TabPFN's limit **and** `tabpfn` is importable → use TabPFN (no training needed, strong on small data). Else → **LightGBM** (`LGBMClassifier`). LightGBM is the default and must always work.
- Split train/validation (stratified). If rows > `TRAIN_ROW_CAP`, sample for training but **score all rows** in batches.
- Drop protected fields and ids from features. One-hot / target-encode categoricals; impute missing.
- **Calibrate** probabilities (`CalibratedClassifierCV`, isotonic) so the PD is a real probability. Report AUC-ROC, PR-AUC, Brier/ECE on validation.
- Output a calibrated **PD** per loan.

### 8.3 Survival / 12-month curve (`survival.py`)

- If a time/observation column exists, fit a discrete-time hazard (or `lifelines` Cox) to produce monthly hazard `h(1..12)` and a cumulative PD curve per loan.
- If no time data, derive a 12-point curve by spreading the calibrated PD across months with a parametric (Weibull-shaped) hazard anchored on `term_months`. **Label this output `estimated`** in the response so the UI can mark it honestly.
- Also output the alert: estimated month of onset (first month the curve crosses a threshold) and the lead time.

### 8.4 Explanation → causal reason codes (`explain.py`)

- Compute SHAP values per loan (TreeExplainer for LightGBM; permutation/KernelSHAP for TabPFN).
- Map the top SHAP drivers onto a **fixed reason-code taxonomy** so every loan is described in the same vocabulary (this is the "common interpretation framework"):
  `LIQUIDITY_STRESS, LEVERAGE_HIGH, REVENUE_DECLINE, REPAYMENT_HISTORY_POOR, SECTOR_RISK, COLLATERAL_LOW, BEHAVIOUR_ANOMALY, TEXT_DISTRESS_SIGNAL, TENURE_RISK, OTHER`.
- Call the `explainer` LLM with the loan's top drivers (feature, value, SHAP sign/magnitude) and ask it to (a) assign reason codes from the fixed list only, (b) write a 2–4 sentence plain-English explanation, (c) propose a recommended action. Return strict JSON (Section 10).

### 8.5 Recourse (`recourse.py`)

Use `dice-ml` (or a greedy search if DiCE is unavailable) over **actionable features only** (`term_months`, `collateral_value`, working-capital/`income_or_turnover`) to find the smallest change that pushes PD below threshold. Return the action and the projected post-action PD.

### 8.6 Fairness (`fairness.py`)

Use `fairlearn` `MetricFrame` to compute demographic-parity difference and equalized-odds difference across each protected attribute. **Difference-aware:** report disparities but flag only those attributable to protected attributes (use SHAP attribution of any proxy features). Return a per-attribute pass/review flag plus the metric values.

---

## 9. The Multi-Model Judge Panel (`judges.py`) — a headline feature

Three judges, each a *different* model family from the producer:

1. **JSON-repair judge** — already wired into `json_safe.py` (Section 6.3). Repairs malformed JSON.
2. **Faithfulness judge** (anti-hallucination, the differentiator) — receives the loan's actual top SHAP drivers and the explainer's reason codes + prose. It must verify every claimed driver is supported by the SHAP evidence and the direction (increases/decreases risk) matches. If the explanation cites a driver not in the SHAP top-k, or flips a sign, it returns `faithful:false` with the offending items; the orchestrator then regenerates the explanation once. Mark the final explanation `faithful:true|false` in the output so the UI can badge it.
3. **Consensus judge** (optional, enable on a toggle) — generate the explanation with two different models, have a third judge pick or merge the clearer, more faithful one. Use for high-risk loans only to save cost.

Expose the judge verdicts in the API so the dashboard can show "explanation verified by faithfulness judge".

---

## 10. Output JSON schemas (define with pydantic in `schemas.py`)

**LoanResult**

```json
{
  "loan_id": "string",
  "pd": 0.72,
  "risk_band": "high|medium|low",
  "risk_curve": { "months": [1,2,...,12], "pd": [0.1,...], "estimated": true },
  "alert": { "flagged": true, "onset_month": 8, "lead_time_months": 7 },
  "reason_codes": [
    { "code": "LIQUIDITY_STRESS", "weight": 0.34, "direction": "increases_risk",
      "evidence": "DSCR 0.8 < 1.0", "feature": "dscr", "shap": -0.41 }
  ],
  "explanation": "Plain-English, 2-4 sentences.",
  "recommended_action": { "action": "Extend tenure 6 months + add working-capital line",
                          "expected_pd_after": 0.28, "rationale": "..." },
  "fairness": { "flag": "pass|review", "details": [ { "attribute": "region", "dp_diff": 0.03 } ] },
  "explanation_quality": { "faithful": true, "json_status": "ok|repaired|degraded",
                           "model_used": "deepseek:deepseek-v4-flash", "judge": "deepseek:deepseek-v4-pro" }
}
```

**PortfolioResult**

```json
{
  "job_id": "string",
  "model": { "type": "lightgbm|tabpfn", "auc": 0.88, "pr_auc": 0.61, "ece": 0.04, "n_loans": 50000 },
  "risk_distribution": { "high": 1200, "medium": 8800, "low": 40000 },
  "top_risk_loans": [ { "loan_id": "...", "pd": 0.93 } ],
  "fairness_summary": [ { "attribute": "gender", "flag": "pass", "eo_diff": 0.02 } ],
  "mapping_used": { "...": "..." },
  "warnings": [ "survival curve is estimated (no time column found)" ]
}
```

Every field the UI shows must come from these contracts. The LLM never outputs the PD — the ML model does. The LLM only writes reason codes, prose, and the action.

---

## 11. Frontend (React)

Design goal: a calm, trustworthy credit "war-room". Risk = saffron/amber, safe = teal, navy ink base. Reasoning and explanation are the hero of the UI, never hidden behind a click.

**Pages / flow**

1. **Upload** — drag-drop CSV; show detected columns, dtypes, sample, row count.
2. **Mapping** — show the AI's proposed canonical mapping with a confidence chip per field; user can change any mapping via dropdowns; confirm target and protected fields. "Run analysis" button.
3. **Processing** — live progress bar driven by `GET /status` (mapping → training → scoring → explaining → auditing), with the current stage message.
4. **Dashboard (portfolio)** — model metric cards (AUC, calibration, n loans); risk distribution chart; sortable table of loans by PD; fairness summary badges; warnings banner (e.g. "survival estimated").
5. **Loan detail** — the **RiskCurve** chart (12-month, with threshold line + onset marker, marked "estimated" when applicable); the PD and alert; **ReasonCodes** list (ranked, fixed-taxonomy chips with evidence); the **plain-English explanation** in a prominent card with a "verified by faithfulness judge" badge; the **RecourseCard** (before-PD → action → after-PD); the **FairnessBadge**; and a small **ModelTrace** showing which provider/model wrote the explanation and the JSON status (ok/repaired). This transparency is a strong demo and judging point.

**Components:** `RiskCurve` (recharts line + reference line), `ReasonCodes`, `RecourseCard`, `FairnessBadge`, `ModelTrace`, `MetricCard`, `RiskTable`, `MappingEditor`, `StageProgress`.

Add a small **Settings** drawer to switch the model per role and toggle the consensus judge.

---

## 12. Resilience & error handling (required)

- No single LLM failure may crash a job. Failed explanations degrade gracefully to a SHAP-only reason list with `json_status:"degraded"`.
- All LLM JSON goes through the 5-layer pipeline (6.3). Never trust raw model JSON.
- Large CSVs: profile with polars; sample for training; **score in batches**; paginate the loan table.
- Verify model IDs at startup via each provider's `GET /models`; log and self-correct if a configured ID is missing (DeepSeek rename on 2026-07-24).
- Per-call: timeout, retry with backoff, key rotation, then provider fallback.
- Validate every uploaded file type and size. Handle missing target column with a clear UI error.

---

## 13. Build order (ship an MVP fast, then deepen)

1. **M1 — skeleton:** Flask + React up; `/upload` profiles a CSV; React shows columns. (No ML yet.)
2. **M2 — gateway:** `llm/gateway.py` + `json_safe.py` + `/api/models`. Prove a JSON call works across all four providers with rotation + fallback.
3. **M3 — mapping:** `/api/map` maps SBA and German CSVs to the canonical schema; UI mapping editor.
4. **M4 — predict:** LightGBM train + calibrate + PD + portfolio metrics; dashboard table and metric cards.
5. **M5 — explain:** SHAP → reason codes → explainer LLM → loan detail with reasoning and recourse.
6. **M6 — judges:** faithfulness judge + degraded-mode badges + model trace.
7. **M7 — survival + fairness:** 12-month curve (estimated fallback) and fairness audit.
8. **M8 — polish:** TabPFN router, consensus toggle, settings, demo styling.

Each milestone must run end-to-end before starting the next.

---

## 14. Acceptance tests (use real Kaggle CSVs)

- **SBA** ("Should This Loan be Approved or Denied", ~899k rows): mapper finds `MIS_Status`/charge-off as target; trains on a 100k sample; scores all rows; AUC printed; dashboard loads; a high-PD loan shows reason codes + action + fairness flag.
- **German Credit** (1,000 rows, has sex/age): fairness audit produces a non-trivial disparity number across `gender`; faithfulness judge passes on at least 90% of explanations.
- **Malformed-JSON test:** force the explainer to emit broken JSON (e.g. trailing comma, code fences) and confirm the 5-layer pipeline + json_judge recover it without crashing.
- **Provider-failure test:** disable the DeepSeek keys and confirm the app falls back to Mistral automatically.

A run is "done" when: any of these CSVs uploads, maps, trains, scores, and renders per-loan results **with a faithful plain-English explanation and a recommended action**, with no unhandled exception.

---

## 15. Winning / demo touches (do these)

- **Faithfulness badge + model trace** on every explanation — shows judges the output is verified, not hallucinated. This is the trust story that wins.
- **Recourse before→after PD** card — the single most memorable visual; lead the demo with it.
- **Fixed reason-code taxonomy** — emphasise "same explanation format for every loan", which is exactly the problem statement's "common interpretation framework".
- **Honest labelling** — mark estimated survival curves as estimated; never overstate. Judges trust honest systems.
- **One-screen-per-loan** — curve + reason + action + fairness in a single view a credit officer could use on day one.

---

## 16. Non-negotiables

- The **ML model** outputs the probability of default. The **LLM** only writes reasons, prose, and actions. Never let the LLM invent a PD.
- Protected attributes (region, gender, community) are **never** model features — audit only.
- Secrets stay server-side. Frontend holds no keys.
- Every LLM JSON response is schema-validated; unrecoverable ones degrade, they do not crash.
- Keep the orchestrator deterministic and logged so any run is reproducible for the paper.
