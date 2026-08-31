"""
Pre-trained global model: SAARTHI arrives knowing something.

Before this module the app could only train on whatever the user uploaded,
which meant (a) a file with no outcome column could not be scored at all, and
(b) every demo was a cold start. Here we load a model trained offline on a
pooled corpus of public credit datasets (SBA, Lending Club, Home Credit, GMSC,
Taiwan, German, Berka) expressed in a shared canonical vocabulary, and use it
two ways:

  * NO LABEL in the upload  -> score directly from the global model.
  * LABEL present           -> train the per-book model as before, and feed the
                               global model's PD in as an extra feature
                               (stacking), so the bank's own data refines a
                               prior instead of starting from zero.

The bundle is produced by training/train_global.py and is optional: if it is
absent the app degrades to exactly its previous behaviour.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# canonical vocabulary of the trained global model
CANON_NUM = [
    "can_loan_amount", "can_term_months", "can_interest_rate", "can_income",
    "can_dti", "can_credit_score_n", "can_age", "can_emp_length",
    "can_delinq_count", "can_open_accounts", "can_utilization",
    "can_n_employees", "can_loan_to_income", "can_installment",
]
CANON_CAT = ["can_sector"]

# app canonical field -> global-model canonical field
APP_TO_GLOBAL = {
    "loan_amount": "can_loan_amount",
    "term_months": "can_term_months",
    "interest_rate": "can_interest_rate",
    "income_or_turnover": "can_income",
    "credit_score": "can_credit_score_n",
    "sector": "can_sector",
    "prior_delinquencies": "can_delinq_count",
    "employment_length": "can_emp_length",
}

MODEL_PATH = os.environ.get(
    "SAARTHI_GLOBAL_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "models", "global_canonical.joblib"))

_LOCK = threading.Lock()
_BUNDLE = None
_TRIED = False


def _num(s) -> pd.Series:
    if isinstance(s, pd.Series) and pd.api.types.is_numeric_dtype(s):
        return s.astype("float32")
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(r"[,$%\s]", "", regex=True),
        errors="coerce").astype("float32")


def available() -> bool:
    return load_global() is not None


def load_global():
    """Load (once) the pooled global model bundle, or None if unavailable."""
    global _BUNDLE, _TRIED
    if _BUNDLE is not None or _TRIED:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None or _TRIED:
            return _BUNDLE
        _TRIED = True
        try:
            import joblib
            if not os.path.exists(MODEL_PATH):
                return None
            _BUNDLE = joblib.load(MODEL_PATH)
            print(f"[pretrained] loaded global model: "
                  f"{len(_BUNDLE.get('features', []))} features, "
                  f"members={list(_BUNDLE.get('members', {}))}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[pretrained] could not load global model: {e}", flush=True)
            _BUNDLE = None
    return _BUNDLE


def to_canonical(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Project an uploaded frame onto the global model's canonical vocabulary."""
    out = pd.DataFrame(index=df.index)
    for c in CANON_NUM:
        out[c] = np.nan
    out["can_sector"] = None

    for app_field, glob_field in APP_TO_GLOBAL.items():
        src = mapping.get(app_field)
        if not src or src not in df.columns:
            continue
        if glob_field == "can_sector":
            out[glob_field] = df[src].astype(str)
        elif glob_field == "can_credit_score_n":
            v = _num(df[src])
            # normalise common bureau ranges (FICO 300-850, CIBIL 300-900) to [0,1]
            hi = float(np.nanmax(v.values)) if v.notna().any() else 1.0
            out[glob_field] = (v / 900.0).clip(0, 1) if hi > 1.5 else v.clip(0, 1)
        else:
            out[glob_field] = _num(df[src])

    # engineered ratios the global model expects
    if out["can_loan_amount"].notna().any() and out["can_income"].notna().any():
        out["can_loan_to_income"] = (out["can_loan_amount"]
                                     / out["can_income"].replace(0, np.nan))
    if out["can_loan_amount"].notna().any() and out["can_term_months"].notna().any():
        out["can_installment"] = (out["can_loan_amount"]
                                  / out["can_term_months"].replace(0, np.nan))
    out["can_sector"] = out["can_sector"].astype("category")
    return out


def score(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Optional[np.ndarray]:
    """Calibrated PD per row from the pre-trained global model, or None."""
    b = load_global()
    if b is None:
        return None
    try:
        X = to_canonical(df, mapping)
        X = X.reindex(columns=b["features"])
        for c in b.get("cat_cols", []):
            if c in X.columns:
                X[c] = X[c].astype("category")
        raw = []
        for m in b["members"].values():
            name = type(m).__name__
            if name.startswith("CatBoost"):
                Xc = X.copy()
                for c in b.get("cat_cols", []):
                    if c in Xc.columns:
                        Xc[c] = Xc[c].astype(str).fillna("__NA__")
                raw.append(m.predict_proba(Xc)[:, 1])
            else:
                raw.append(m.predict_proba(X)[:, 1])
        p = np.mean(raw, axis=0)
        return np.clip(b["calibrator"].predict(p), 1e-4, 1 - 1e-4)
    except Exception as e:  # noqa: BLE001
        print(f"[pretrained] scoring failed: {e}", flush=True)
        return None


def coverage(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> float:
    """Fraction of canonical fields the upload actually populates (0-1)."""
    X = to_canonical(df, mapping)
    cols = CANON_NUM + CANON_CAT
    return float(sum(X[c].notna().any() for c in cols) / len(cols))


def info() -> dict:
    b = load_global()
    if b is None:
        return {"available": False}
    m = b.get("metrics", {}).get("calibrated_test", {})
    return {
        "available": True,
        "features": len(b.get("features", [])),
        "members": list(b.get("members", {})),
        "trained_on": b.get("pool_datasets", []),
        "test_auc": m.get("auc"),
        "test_ece": m.get("ece"),
        "test_n": m.get("n"),
    }
