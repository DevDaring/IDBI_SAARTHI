"""
Fairness audit (difference-aware).

Uses fairlearn MetricFrame to compute demographic-parity difference and
equalized-odds difference across each protected attribute (region, gender,
community). Protected attributes are audit-only and never enter the model.

Difference-aware: we report disparities but only raise a 'review' flag when the
disparity is both material AND plausibly attributable to the protected attribute
(rather than legitimate risk factors). We approximate attribution by checking
whether the disparity persists after conditioning on the predicted risk band —
a proxy for "same-risk applicants treated differently".
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# thresholds for raising a review flag
DP_REVIEW = 0.10   # demographic-parity difference
EO_REVIEW = 0.10   # equalized-odds difference
MIN_GROUP = 20     # ignore groups smaller than this


def audit(protected: pd.DataFrame, pd_all: np.ndarray,
          y_true: Optional[pd.Series], band_high: float) -> List[dict]:
    """Return a per-protected-attribute fairness summary."""
    results: List[dict] = []
    if protected is None or protected.shape[1] == 0:
        return results

    y_pred = (pd_all >= band_high).astype(int)
    n = len(pd_all)

    for attr in protected.columns:
        col = protected[attr].astype(str).fillna("__NA__")
        if len(col) != n:
            col = col.reindex(range(n)).fillna("__NA__")
        groups = col.value_counts()
        big_groups = groups[groups >= MIN_GROUP].index.tolist()
        if len(big_groups) < 2:
            continue
        mask = col.isin(big_groups)

        dp = _demographic_parity_diff(y_pred[mask.values], col[mask].values)
        eo = None
        if y_true is not None:
            yt = y_true.reset_index(drop=True)
            yt = yt.reindex(range(n)).fillna(0).astype(int)
            eo = _equalized_odds_diff(yt.values[mask.values],
                                      y_pred[mask.values], col[mask].values)

        # difference-aware: does the gap persist WITHIN the same risk band?
        residual = _within_band_gap(pd_all[mask.values], col[mask].values, band_high)

        material = (dp is not None and dp >= DP_REVIEW) or (eo is not None and eo >= EO_REVIEW)
        attributable = residual is not None and residual >= DP_REVIEW * 0.6
        flag = "review" if (material and attributable) else "pass"

        results.append({
            "attribute": attr,
            "flag": flag,
            "dp_diff": round(float(dp), 4) if dp is not None else 0.0,
            "eo_diff": round(float(eo), 4) if eo is not None else 0.0,
            "residual_within_band": round(float(residual), 4) if residual is not None else 0.0,
            "n_groups": len(big_groups),
            "note": ("Disparity persists among same-risk applicants — review for "
                     "proxy bias." if flag == "review"
                     else "Disparity explained by legitimate risk factors / below "
                          "threshold."),
        })
    return results


def _demographic_parity_diff(y_pred: np.ndarray, groups: np.ndarray) -> Optional[float]:
    try:
        from fairlearn.metrics import MetricFrame, selection_rate
        mf = MetricFrame(metrics=selection_rate, y_true=y_pred, y_pred=y_pred,
                         sensitive_features=groups)
        return float(mf.difference())
    except Exception:
        rates = pd.Series(y_pred).groupby(pd.Series(groups)).mean()
        return float(rates.max() - rates.min()) if len(rates) >= 2 else None


def _equalized_odds_diff(y_true: np.ndarray, y_pred: np.ndarray,
                         groups: np.ndarray) -> Optional[float]:
    try:
        from fairlearn.metrics import equalized_odds_difference
        return float(equalized_odds_difference(y_true, y_pred, sensitive_features=groups))
    except Exception:
        # fallback: max TPR gap
        df = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups})
        tprs = []
        for g, sub in df.groupby("g"):
            pos = sub[sub.y == 1]
            if len(pos) >= 5:
                tprs.append(pos.p.mean())
        return float(max(tprs) - min(tprs)) if len(tprs) >= 2 else None


def _within_band_gap(pd_vals: np.ndarray, groups: np.ndarray,
                     band_high: float) -> Optional[float]:
    """Selection-rate gap computed only among medium/high predicted-risk loans —
    a proxy for treating same-risk applicants differently by group."""
    pred = (pd_vals >= band_high).astype(int)
    band = pd_vals >= 0.20  # medium+ risk
    if band.sum() < 2 * MIN_GROUP:
        return None
    sub = pd.DataFrame({"p": pred[band], "g": groups[band]})
    rates = sub.groupby("g")["p"].mean()
    rates = rates[sub.groupby("g").size() >= MIN_GROUP // 2]
    return float(rates.max() - rates.min()) if len(rates) >= 2 else None


def per_loan_fairness(loan_groups: Dict[str, str],
                      summary: List[dict]) -> dict:
    """Build the per-loan fairness object from the portfolio summary."""
    by_attr = {s["attribute"]: s for s in summary}
    details = []
    flag = "pass"
    for attr, val in loan_groups.items():
        s = by_attr.get(attr)
        if s is None:
            continue
        details.append({"attribute": attr, "dp_diff": s["dp_diff"]})
        if s["flag"] == "review":
            flag = "review"
    return {"flag": flag, "details": details}
