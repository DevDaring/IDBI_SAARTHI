"""
Survival / 12-month risk curve.

Key design principle: the curve must be CONSISTENT with the headline PD. The
cumulative default probability at month 12 always equals the loan's calibrated
PD; survival analysis only shapes *when* that risk accrues across the 12 months.

Two shaping modes:
* If a usable time/observation column exists, fit a lifelines Cox model
  (duration = months observed, event = default) and use its baseline cumulative
  hazard to derive the monthly accrual SHAPE from real time-to-event data.
  Marked `estimated = False`.
* Otherwise, use a per-loan Weibull hazard shape anchored on `term_months`.
  Marked `estimated = True` so the UI labels it honestly.

In both modes curve_i[t] = PD_i * shape[t], shape[12] = 1.

Also derives the early-warning alert: the first month the cumulative curve
crosses the onset threshold, and the resulting lead time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import SETTINGS
from pipeline.features import FeatureBundle

HORIZON = 12


def build_curves(bundle: FeatureBundle, pd_all: np.ndarray,
                 mapping: Dict[str, Optional[str]],
                 df: pd.DataFrame) -> Tuple[np.ndarray, bool, List[str]]:
    """Return (curves[n, 12] cumulative PD, estimated_flag, warnings)."""
    warnings: List[str] = []
    n = len(pd_all)
    time_src = mapping.get("time_observed")
    term_src = mapping.get("term_months")

    # --- try a Cox-informed accrual shape from real time-to-event data ----
    if time_src and time_src in df.columns and bundle.y is not None:
        shape = _cox_shape(bundle, df, time_src)
        if shape is not None:
            curves = pd_all[:, None] * shape[None, :]
            return np.clip(curves, 0.0, 1.0), False, warnings
        warnings.append("Couldn’t fit a full time-to-default model on this data, so the "
                        "12-month risk curve is estimated (and labelled as such).")

    # --- parametric (estimated) per-loan shape ----------------------------
    terms = None
    if term_src and term_src in df.columns:
        terms = pd.to_numeric(df[term_src], errors="coerce").reset_index(drop=True)
        terms = terms.reindex(range(n)).ffill().bfill()
    curves = _parametric_curves(pd_all, terms.values if terms is not None else None)
    warnings.append("The 12-month risk curve is estimated — your data has no "
                    "month-by-month history column, so the timing is modelled and "
                    "clearly labelled ‘estimated’.")
    return curves, True, warnings


def _parametric_curves(pd_all: np.ndarray, terms: Optional[np.ndarray]) -> np.ndarray:
    """Per-loan Weibull accrual shape, scaled so curve[12] == PD."""
    n = len(pd_all)
    months = np.arange(1, HORIZON + 1)
    curves = np.zeros((n, HORIZON))
    for i in range(n):
        p = float(pd_all[i])
        term = 12.0
        if terms is not None and i < len(terms) and not np.isnan(terms[i]) and terms[i] > 0:
            term = float(terms[i])
        # longer-tenure loans accrue hazard a little later (larger shape k)
        k = float(np.clip(1.3 + term / 60.0, 1.3, 2.6))
        lam = HORIZON / 1.6
        raw = 1 - np.exp(-((months / lam) ** k))      # increasing 0..1
        shape = raw / raw[-1] if raw[-1] > 1e-9 else np.linspace(0, 1, HORIZON)
        curves[i] = np.clip(p * shape, 0.0, 1.0)
    return curves


def _cox_shape(bundle: FeatureBundle, df: pd.DataFrame, time_src: str) -> Optional[np.ndarray]:
    """Population monthly accrual SHAPE in [0,1] (shape[12]=1) from a Cox fit.

    Returns None on failure. The shape captures WHEN defaults occur over the
    first 12 months; per-loan level is applied separately (= PD).
    """
    try:
        from lifelines import CoxPHFitter
    except Exception:
        return None
    try:
        durations = pd.to_numeric(df[time_src], errors="coerce").reset_index(drop=True)
        y = bundle.y.reset_index(drop=True)
        m = min(len(durations), len(y), len(bundle.X))
        durations = durations.iloc[:m].fillna(durations.median()).clip(lower=0.1)
        event = y.iloc[:m].astype(int)

        X = bundle.X.iloc[:m].copy()
        num = X.select_dtypes(include=[np.number])
        cols = num.var().sort_values(ascending=False).head(10).index.tolist()
        if not cols:
            return None
        cph_df = X[cols].copy()
        cph_df = (cph_df - cph_df.mean()) / (cph_df.std().replace(0, 1))
        cph_df["__duration__"] = durations.values
        cph_df["__event__"] = event.values

        cph = CoxPHFitter(penalizer=0.5)
        cph.fit(cph_df, duration_col="__duration__", event_col="__event__",
                show_progress=False)

        # baseline cumulative hazard at months 1..12 -> baseline cumulative prob
        times = list(range(1, HORIZON + 1))
        bch = cph.baseline_cumulative_hazard_
        col0 = bch.columns[0]
        # interpolate baseline cumulative hazard onto integer months
        h = np.interp(times, bch.index.values, bch[col0].values)
        cum = 1 - np.exp(-h)                  # baseline cumulative default prob
        if cum[-1] <= 1e-9:
            return None
        shape = cum / cum[-1]                 # normalise so month 12 == 1
        shape = np.maximum.accumulate(shape)  # enforce monotone
        return np.clip(shape, 0.0, 1.0)
    except Exception:
        return None


def alert_from_curve(curve: np.ndarray) -> dict:
    """First month the cumulative PD crosses the onset threshold + lead time."""
    thr = SETTINGS.onset_threshold
    months = np.arange(1, HORIZON + 1)
    crossed = np.where(curve >= thr)[0]
    if len(crossed) == 0:
        return {"flagged": False, "onset_month": None, "lead_time_months": None}
    onset = int(months[crossed[0]])
    return {"flagged": True, "onset_month": onset, "lead_time_months": onset}
