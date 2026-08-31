"""
Orchestrator: runs every pipeline stage deterministically and emits progress.

ingest -> map -> features -> train+score -> survival -> explain -> judge ->
recourse -> fairness -> assemble

Design: PD + SHAP reason codes are computed for every (capped) loan cheaply.
The expensive LLM explanation + judge + recourse is precomputed only for the
top-K riskiest loans during the run, and produced lazily (and cached) for any
other loan on first access. This keeps a portfolio run fast and cost-bounded
while every loan the user opens still gets a faithful, verified explanation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import SETTINGS
from jobs import Job
from pipeline import fairness as fairness_mod
from pipeline.explain import ShapEngine, build_drivers, explain_loan
from pipeline.features import build_features
from pipeline.ingest import read_table
from pipeline.judges import consensus_explanation, faithfulness_check
from pipeline.model import risk_band, train_and_score
from pipeline.recourse import find_recourse
from pipeline.survival import HORIZON, alert_from_curve, build_curves
from schemas import (Alert, ExplanationQuality, LoanFairness, LoanResult,
                     ModelInfo, PortfolioResult, RecommendedAction, RiskCurve,
                     RiskDistribution, TopRiskLoan)


def run_pipeline(job: Job, upload_path: str, mapping: Dict[str, Optional[str]],
                 target: Optional[str], protected: List[str],
                 options: Optional[dict] = None):
    options = options or {}
    job.options = options

    def prog(stage, pct, msg):
        job.update(stage=stage, percent=pct, message=msg)

    # ---- ingest ----------------------------------------------------------
    prog("ingest", 4, "Reading dataset")
    df = read_table(upload_path, max_rows=SETTINGS.score_row_cap)
    df = df.reset_index(drop=True)
    n_total = len(df)
    warnings: List[str] = []
    if n_total >= SETTINGS.score_row_cap:
        warnings.append(f"Used the first {SETTINGS.score_row_cap:,} rows of your file for "
                        f"this run to keep it fast.")

    # ---- map (already chosen by user; validate) --------------------------
    prog("map", 12, "Validating column mapping")
    if not target or target not in df.columns:
        raise ValueError(f"target column '{target}' not found in dataset")

    # ---- features --------------------------------------------------------
    prog("features", 20, "Building feature matrix")
    bundle = build_features(df, mapping, target, protected, for_training=True)
    warnings += bundle.warnings
    if bundle.X.shape[1] == 0:
        raise ValueError("no usable features after mapping")

    # ---- stack the pre-trained global prior in as a feature --------------
    # The global model was trained offline on ~3.3M public loans in a shared
    # canonical vocabulary. Feeding its calibrated PD in as one extra column
    # lets this bank's book REFINE a prior instead of starting from zero, and
    # is a no-op when the bundle is absent or the upload maps too few fields.
    try:
        from pipeline import pretrained
        if pretrained.available():
            ident = {k: k for k in pretrained.APP_TO_GLOBAL}
            cov = pretrained.coverage(bundle.raw_features, ident)
            if cov >= 0.25:
                gp = pretrained.score(bundle.raw_features, ident)
                if gp is not None and len(gp) == len(bundle.X):
                    bundle.X["global_prior_pd"] = gp.astype("float32")
                    bundle.feature_names = list(bundle.X.columns)
                    warnings.append(
                        "Used SAARTHI's pre-trained global model (trained on public "
                        "credit datasets) as an extra input, so this run starts from "
                        "prior knowledge rather than from scratch.")
            else:
                warnings.append(
                    "Pre-trained global model skipped: too few standard fields were "
                    "mapped for it to contribute reliably.")
    except Exception as e:  # noqa: BLE001 - never let the prior break a run
        print(f"[orchestrator] global prior unavailable: {e}", flush=True)

    # ---- train + score ---------------------------------------------------
    prog("train", 30, "Training & calibrating model")
    artifacts = train_and_score(bundle, progress=prog)
    warnings += artifacts.warnings
    pd_all = artifacts.pd_all
    n = len(pd_all)

    # ---- survival curves -------------------------------------------------
    prog("survival", 78, "Computing 12-month risk curves")
    curves, estimated, swarn = build_curves(bundle, pd_all, mapping, df.iloc[:n])
    warnings += swarn
    alerts = [alert_from_curve(curves[i]) for i in range(n)]

    # ---- fairness audit --------------------------------------------------
    prog("fairness", 84, "Auditing fairness across protected attributes")
    fairness_summary = fairness_mod.audit(
        bundle.protected, pd_all, bundle.y, SETTINGS.band_high)

    # ---- assemble portfolio ---------------------------------------------
    prog("assemble", 88, "Assembling portfolio")
    bands = np.array([risk_band(p) for p in pd_all])
    dist = RiskDistribution(
        high=int((bands == "high").sum()),
        medium=int((bands == "medium").sum()),
        low=int((bands == "low").sum()))
    order = np.argsort(-pd_all)
    top_idx = order[:50]
    top_risk = [TopRiskLoan(loan_id=str(bundle.loan_ids.iloc[i]),
                            pd=round(float(pd_all[i]), 4)) for i in top_idx]

    fairness_api = [
        {"attribute": s["attribute"], "flag": s["flag"],
         "eo_diff": s["eo_diff"], "dp_diff": s["dp_diff"]}
        for s in fairness_summary
    ]
    portfolio = PortfolioResult(
        job_id=job.job_id,
        model=ModelInfo(type=artifacts.model_type, auc=artifacts.metrics["auc"],
                        pr_auc=artifacts.metrics["pr_auc"], ece=artifacts.metrics["ece"],
                        brier=artifacts.metrics.get("brier", 0.0), n_loans=n),
        risk_distribution=dist,
        top_risk_loans=top_risk,
        fairness_summary=fairness_api,
        mapping_used={k: v for k, v in mapping.items()},
        warnings=_dedupe(warnings),
    )

    # ---- stash artifacts for lazy per-loan explanation -------------------
    shap_engine = ShapEngine(artifacts, bundle.feature_names)
    id_to_idx = {str(lid): i for i, lid in enumerate(bundle.loan_ids.tolist())}
    job.artifacts = {
        "X": bundle.X.reset_index(drop=True),
        "raw_features": bundle.raw_features,
        "loan_ids": bundle.loan_ids,
        "pd_all": pd_all,
        "bands": bands,
        "curves": curves,
        "alerts": alerts,
        "estimated": estimated,
        "model_artifacts": artifacts,
        "shap_engine": shap_engine,
        "canonical_present": bundle.canonical_present,
        "protected": bundle.protected,
        "fairness_summary": fairness_summary,
        "id_to_idx": id_to_idx,
    }
    job.portfolio = portfolio.model_dump()

    # ---- precompute explanations for the top-K riskiest loans (parallel) -
    prog("explain", 92, "Explaining top-risk loans")
    k = min(SETTINGS.eager_explain_top_k, n)
    if k > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        with ThreadPoolExecutor(max_workers=SETTINGS.explain_workers) as pool:
            futs = {pool.submit(build_loan_result, job, int(idx)): int(idx)
                    for idx in order[:k]}
            for fut in as_completed(futs):
                try:
                    lr = fut.result()
                    job.loans[lr["loan_id"]] = lr
                except Exception:  # noqa: BLE001 - never let one loan fail the run
                    pass
                done += 1
                job.update(percent=92 + 7 * done / max(1, k),
                           message=f"Explaining top-risk loans ({done}/{k})")

    prog("assemble", 100, "Complete")


# ---------------------------------------------------------------------------
# per-loan result builder (used eagerly for top-K and lazily on demand)
# ---------------------------------------------------------------------------
def build_loan_result(job: Job, idx: int) -> dict:
    a = job.artifacts
    loan_id = str(a["loan_ids"].iloc[idx])
    pd_value = float(a["pd_all"][idx])
    band = a["bands"][idx]
    curve = a["curves"][idx]
    alert = a["alerts"][idx]

    X_row = a["X"].iloc[[idx]]
    raw_row = a["raw_features"].iloc[idx]

    # SHAP -> drivers
    shap_vals = a["shap_engine"].shap_for(X_row)
    if shap_vals is None:
        drivers = []
    else:
        drivers = build_drivers(np.asarray(shap_vals)[0], a["X"].columns.tolist(),
                                raw_row, a["canonical_present"])

    # recourse (grounded counterfactual) before explanation, so the LLM can cite it
    recourse_hint = None
    try:
        recourse_hint = find_recourse(a["model_artifacts"], X_row, raw_row,
                                      a["canonical_present"], pd_value)
    except Exception:  # noqa: BLE001
        recourse_hint = None

    # explanation
    if drivers:
        ex = explain_loan(drivers, pd_value, band, recourse_hint, loan_label=loan_id)
    else:
        ex = _no_driver_explanation(pd_value, band, recourse_hint)

    # faithfulness judge (+ one regeneration if it flags hallucination)
    driver_dicts = [{"feature": d.feature, "direction": d.direction,
                     "shap": d.shap, "value": d.value} for d in drivers]
    faithful = True
    judge_used = ""
    consensus_used = False
    if drivers and not ex.get("degraded"):
        verdict = faithfulness_check(driver_dicts, ex["reason_codes"], ex["explanation"])
        judge_used = verdict.get("judge", "")
        if not verdict["faithful"]:
            # regenerate once, then re-judge
            ex2 = explain_loan(drivers, pd_value, band, recourse_hint, loan_label=loan_id)
            verdict2 = faithfulness_check(driver_dicts, ex2["reason_codes"], ex2["explanation"])
            if verdict2["faithful"] or _fewer_issues(verdict2, verdict):
                ex = ex2
                verdict = verdict2
                judge_used = verdict2.get("judge", judge_used)
            faithful = bool(verdict["faithful"])
        else:
            faithful = True

        # optional consensus for high-risk loans
        if band == "high" and job.options.get("consensus"):
            try:
                ex_b = explain_loan(drivers, pd_value, band, recourse_hint, loan_label=loan_id)
                cons = consensus_explanation(driver_dicts, pd_value,
                                             ex["explanation"], ex_b["explanation"])
                if cons and cons.get("explanation"):
                    if cons["chosen"] == "b":
                        ex = ex_b
                    elif cons["chosen"] == "merged":
                        ex["explanation"] = cons["explanation"]
                    consensus_used = True
            except Exception:  # noqa: BLE001
                pass

    # per-loan fairness from the portfolio audit
    loan_groups = {}
    prot = a["protected"]
    if prot is not None and idx < len(prot):
        for col in prot.columns:
            loan_groups[col] = str(prot.iloc[idx][col])
    loan_fair = fairness_mod.per_loan_fairness(loan_groups, a["fairness_summary"])

    # assemble & validate
    result = LoanResult(
        loan_id=loan_id,
        pd=round(pd_value, 4),
        risk_band=band,
        risk_curve=RiskCurve(months=list(range(1, HORIZON + 1)),
                             pd=[round(float(x), 4) for x in curve],
                             estimated=bool(a["estimated"])),
        alert=Alert(**alert),
        reason_codes=ex["reason_codes"],
        explanation=ex["explanation"],
        recommended_action=ex["recommended_action"],
        fairness=LoanFairness(**loan_fair),
        explanation_quality=ExplanationQuality(
            faithful=faithful,
            json_status=ex["json_status"],
            model_used=ex["model_used"],
            judge=judge_used or "n/a",
            consensus=consensus_used),
    )
    return result.model_dump()


def _no_driver_explanation(pd_value, band, recourse_hint) -> dict:
    return {
        "reason_codes": [],
        "explanation": (f"This loan has a {band} default risk (PD {pd_value:.0%}). "
                        "SHAP attribution was unavailable, so no per-feature drivers "
                        "are shown."),
        "recommended_action": RecommendedAction(
            action=(recourse_hint or {}).get("action", "Review the loan manually."),
            expected_pd_after=float((recourse_hint or {}).get("expected_pd_after",
                                    round(pd_value * 0.7, 2))),
            rationale="Limited explainability for this record."),
        "json_status": "degraded", "model_used": "none", "judge_used": "",
        "degraded": True,
    }


def _fewer_issues(a: dict, b: dict) -> bool:
    na = len(a.get("unsupported_claims", [])) + len(a.get("sign_flips", []))
    nb = len(b.get("unsupported_claims", [])) + len(b.get("sign_flips", []))
    return na < nb


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
