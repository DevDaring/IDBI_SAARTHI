"""
Explanation engine: SHAP -> fixed reason-code taxonomy -> explainer LLM prose.

The ML model owns the numbers (PD, SHAP). The LLM owns only words: it picks a
code from the FIXED taxonomy for each ML-provided driver, writes the evidence
sentence, the 2-4 sentence explanation, and the recommended action's phrasing.
We inject the SHAP value, weight and direction deterministically so the LLM
cannot invent magnitudes or flip signs unnoticed (the faithfulness judge checks
the rest).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from config import SETTINGS
from llm.json_safe import call_json
from schemas import (REASON_CODES, ExplanationQuality, ReasonCodeItem,
                     RecommendedAction)

# ---------------------------------------------------------------------------
# canonical feature -> default reason code (heuristic candidate the LLM refines)
# ---------------------------------------------------------------------------
FEATURE_TO_REASON = {
    "dscr": "LIQUIDITY_STRESS",
    "income_or_turnover": "REVENUE_DECLINE",
    "loan_amount": "LEVERAGE_HIGH",
    "interest_rate": "LEVERAGE_HIGH",
    "prior_delinquencies": "REPAYMENT_HISTORY_POOR",
    "credit_score": "REPAYMENT_HISTORY_POOR",
    "sector": "SECTOR_RISK",
    "collateral_value": "COLLATERAL_LOW",
    "employment_length": "BEHAVIOUR_ANOMALY",
    "term_months": "TENURE_RISK",
    "time_observed": "TENURE_RISK",
    "text_purpose": "TEXT_DISTRESS_SIGNAL",
}


@dataclass
class Driver:
    feature: str          # canonical-ish feature name (e.g. 'dscr', 'sector_Retail')
    base: str             # canonical base ('dscr', 'sector', ...)
    value: object         # human-readable value
    shap: float           # signed SHAP (>0 increases default risk)
    weight: float         # normalised |shap| in [0,1]
    direction: str        # increases_risk | decreases_risk
    candidate_code: str   # heuristic reason code


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
class ShapEngine:
    """Lazily-built SHAP explainer bound to a fitted model + feature space."""

    def __init__(self, artifacts, feature_names: List[str]):
        self.artifacts = artifacts
        self.feature_names = feature_names
        self._explainer = None
        self._kind = None
        self._built = False
        self._lock = __import__("threading").Lock()

    def _ensure(self):
        if self._built:
            return
        with self._lock:
            if self._built:
                return
            self._build()
            self._built = True

    def _build(self):
        import shap
        raw = self.artifacts.raw_model
        try:
            if self.artifacts.model_type == "lightgbm":
                self._explainer = shap.TreeExplainer(raw)
                self._kind = "tree"
                return
        except Exception:
            pass
        # fallback: KernelSHAP over a small background (slow; used for TabPFN/prior)
        try:
            bg = self.artifacts.background
            f = lambda d: _proba(raw, pd.DataFrame(d, columns=self.feature_names))
            self._explainer = shap.KernelExplainer(f, bg.values[:50])
            self._kind = "kernel"
        except Exception:
            self._explainer = None
            self._kind = None

    def shap_for(self, X_rows: pd.DataFrame) -> Optional[np.ndarray]:
        self._ensure()
        if self._explainer is None:
            return None
        try:
            if self._kind == "tree":
                vals = self._explainer.shap_values(X_rows)
                if isinstance(vals, list):           # [class0, class1]
                    vals = vals[1] if len(vals) > 1 else vals[0]
                return np.asarray(vals)
            else:
                vals = self._explainer.shap_values(X_rows.values, nsamples=100, silent=True)
                if isinstance(vals, list):
                    vals = vals[1] if len(vals) > 1 else vals[0]
                return np.asarray(vals)
        except Exception:
            return None


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if getattr(p, "ndim", 1) == 2 else p
    return np.asarray(model.predict(X), dtype=float)


# ---------------------------------------------------------------------------
# driver extraction
# ---------------------------------------------------------------------------
def _canonical_base(feature: str, canonical_present: Dict[str, str]) -> str:
    for canon in canonical_present:
        if feature == canon or feature.startswith(canon + "_") or feature.startswith(canon + "__"):
            return canon
    return feature


def build_drivers(shap_row: np.ndarray, feature_names: List[str],
                  raw_feature_row: pd.Series, canonical_present: Dict[str, str],
                  k: int = None) -> List[Driver]:
    k = k or SETTINGS.top_k_drivers
    order = np.argsort(-np.abs(shap_row))
    max_abs = float(np.abs(shap_row).max()) or 1.0
    drivers: List[Driver] = []
    seen_bases = set()
    for idx in order:
        if len(drivers) >= k:
            break
        sv = float(shap_row[idx])
        if abs(sv) < 1e-6:
            continue
        fname = feature_names[idx]
        base = _canonical_base(fname, canonical_present)
        # human value: prefer the readable raw feature for the base
        if base in raw_feature_row.index:
            value = raw_feature_row[base]
        elif fname in raw_feature_row.index:
            value = raw_feature_row[fname]
        else:
            # one-hot column: value is the category encoded in the name
            value = fname.split("_", 1)[1] if "_" in fname else fname
        value = _fmt_value(value)
        direction = "increases_risk" if sv > 0 else "decreases_risk"
        cand = FEATURE_TO_REASON.get(base, "OTHER")
        if base.startswith("text") or "distress" in fname:
            cand = "TEXT_DISTRESS_SIGNAL"
        drivers.append(Driver(
            feature=base, base=base, value=value, shap=round(sv, 4),
            weight=round(abs(sv) / max_abs, 3), direction=direction,
            candidate_code=cand if cand in REASON_CODES else "OTHER",
        ))
        seen_bases.add(base)
    return drivers


def _fmt_value(v) -> str:
    try:
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "missing"
            return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"
    except Exception:
        pass
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


# ---------------------------------------------------------------------------
# explainer LLM
# ---------------------------------------------------------------------------
class _LLMReason(BaseModel):
    feature: str
    code: str
    evidence: str


class _LLMExplain(BaseModel):
    reason_codes: List[_LLMReason]
    explanation: str
    recommended_action: RecommendedAction


_SCHEMA_HINT = (
    '{"reason_codes": [{"feature": <one of the given features>, '
    '"code": <one of the fixed reason codes>, "evidence": "short fact e.g. DSCR 0.8 < 1.0"}], '
    '"explanation": "2-4 plain-English sentences for a credit officer", '
    '"recommended_action": {"action": "concrete step", "expected_pd_after": 0.0-1.0, "rationale": "why"}}'
)


def explain_loan(drivers: List[Driver], pd_value: float, risk_band: str,
                 recourse_hint: Optional[dict] = None,
                 loan_label: str = "") -> dict:
    """Run the explainer LLM and assemble grounded reason codes + prose.

    Returns dict with: reason_codes (List[ReasonCodeItem]), explanation,
    recommended_action (RecommendedAction), quality (json_status, model_used).
    Degrades to a SHAP-only reason list if the LLM/JSON fails.
    """
    drivers_payload = [{
        "feature": d.feature, "value": str(d.value), "shap": d.shap,
        "direction": d.direction, "magnitude": d.weight,
        "candidate_code": d.candidate_code,
    } for d in drivers]

    sys = ("You are a senior MSME credit risk analyst. You explain a model's "
           "default prediction using ONLY the SHAP drivers given. Strict rules: "
           "(1) never introduce a risk factor that is not one of the given drivers; "
           "(2) never contradict a driver's direction; (3) never output a "
           "probability of your own; (4) describe each driver as a CURRENT LEVEL or "
           "VALUE and its effect on risk — do NOT assert trends or changes over "
           "time (avoid words like 'declined', 'rising', 'dropped', 'increased') "
           "because you only see a snapshot, not history. Say 'low DSCR' not 'DSCR "
           "has fallen'; say 'turnover is low relative to debt' not 'revenue "
           "declined'. Output valid JSON only (the word json applies).")
    user = (
        f"Loan {loan_label}: model probability of default PD = {pd_value:.2f} "
        f"(risk band: {risk_band}).\n\n"
        f"SHAP drivers (signed shap > 0 means it INCREASES default risk):\n"
        f"{json.dumps(drivers_payload, indent=2)}\n\n"
        f"Allowed reason codes (use ONLY these): {REASON_CODES}\n\n"
        + (f"A counterfactual recourse search suggests: {json.dumps(recourse_hint)}.\n"
           if recourse_hint else "")
        + "Tasks:\n"
        "1. For each driver, assign the single best-fitting reason code from the "
        "allowed list (you may keep candidate_code or improve it).\n"
        "2. Write a 2-4 sentence plain-English explanation a credit officer can act "
        "on. Reference the actual driver values. Be concrete, not generic.\n"
        "3. Recommend ONE specific action to reduce the default risk, with a brief "
        "rationale; if a recourse hint is given, phrase it naturally and use its "
        "projected PD as expected_pd_after.\n\n"
        f"Respond in EXACTLY this JSON shape:\n{_SCHEMA_HINT}"
    )

    res = call_json("explainer", [{"role": "system", "content": sys},
                                  {"role": "user", "content": user}],
                    _LLMExplain, _SCHEMA_HINT, temperature=0.3, max_tokens=700)

    # degraded path: build reason codes straight from SHAP, generic prose
    if res.obj is None:
        return _degraded(drivers, pd_value, risk_band, recourse_hint,
                         res.model_used, res.status)

    out: _LLMExplain = res.obj  # type: ignore
    # merge LLM code+evidence onto deterministic shap/weight/direction
    by_feature = {d.feature: d for d in drivers}
    reason_codes: List[ReasonCodeItem] = []
    for r in out.reason_codes:
        d = by_feature.get(r.feature)
        if d is None:
            # LLM referenced an unknown feature; skip (faithfulness will catch)
            continue
        code = r.code.strip().upper()
        if code not in REASON_CODES:
            code = d.candidate_code
        reason_codes.append(ReasonCodeItem(
            code=code, weight=d.weight, direction=d.direction,
            evidence=r.evidence.strip()[:160] or f"{d.feature} = {d.value}",
            feature=d.feature, shap=d.shap))
    # ensure we kept all drivers (backfill any the LLM dropped)
    covered = {rc.feature for rc in reason_codes}
    for d in drivers:
        if d.feature not in covered:
            reason_codes.append(ReasonCodeItem(
                code=d.candidate_code, weight=d.weight, direction=d.direction,
                evidence=f"{d.feature} = {d.value}", feature=d.feature, shap=d.shap))
    reason_codes.sort(key=lambda r: -r.weight)

    action = out.recommended_action
    if recourse_hint:  # ground the projected PD on the actual recourse search
        action = RecommendedAction(
            action=action.action,
            expected_pd_after=float(recourse_hint.get("expected_pd_after",
                                                       action.expected_pd_after)),
            rationale=action.rationale)

    return {
        "reason_codes": reason_codes,
        "explanation": out.explanation.strip(),
        "recommended_action": action,
        "json_status": res.status,
        "model_used": res.model_used,
        "judge_used": res.judge_used,
        "degraded": False,
    }


def _degraded(drivers: List[Driver], pd_value: float, risk_band: str,
              recourse_hint: Optional[dict], model_used: str, status: str) -> dict:
    reason_codes = [ReasonCodeItem(
        code=d.candidate_code, weight=d.weight, direction=d.direction,
        evidence=f"{d.feature} = {d.value} ({'raises' if d.shap > 0 else 'lowers'} risk)",
        feature=d.feature, shap=d.shap) for d in drivers]
    top = reason_codes[0] if reason_codes else None
    expl = (f"This loan carries a {risk_band} default risk (PD {pd_value:.0%}). "
            + (f"The largest driver is {top.feature} ({top.evidence}). " if top else "")
            + "Explanation generated in degraded mode (LLM unavailable); drivers are "
              "from the model's SHAP attribution.")
    action = RecommendedAction(
        action=(recourse_hint or {}).get("action", "Review collateral and tenure; "
                "consider a working-capital line to ease repayment."),
        expected_pd_after=float((recourse_hint or {}).get("expected_pd_after",
                                round(max(0.02, pd_value * 0.6), 2))),
        rationale="Automated recourse estimate (LLM explanation unavailable).")
    return {
        "reason_codes": reason_codes, "explanation": expl,
        "recommended_action": action, "json_status": "degraded",
        "model_used": model_used or "none", "judge_used": "", "degraded": True,
    }
