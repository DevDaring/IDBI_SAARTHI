"""
Prediction model router: TabPFN (best-effort, small data) or LightGBM (default),
with isotonic probability calibration so the PD is a real probability.

Outputs a calibrated PD per loan plus validation metrics (AUC-ROC, PR-AUC,
Brier, ECE). The raw tree model is kept for SHAP (TreeExplainer) in explain.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

try:  # sklearn >= 1.6 replaced cv="prefit" with FrozenEstimator
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except Exception:  # noqa: BLE001
    _HAS_FROZEN = False

from config import SETTINGS
from pipeline.features import FeatureBundle

TABPFN_MAX_ROWS = 1100
TABPFN_MAX_FEATURES = 100


@dataclass
class ModelArtifacts:
    model_type: str                       # "lightgbm" | "tabpfn"
    pd_all: np.ndarray                    # calibrated PD per row (all scored rows)
    metrics: dict                         # auc, pr_auc, brier, ece, n_loans
    raw_model: object = None              # uncalibrated estimator (for SHAP)
    calibrated: object = None             # calibrated classifier
    feature_names: List[str] = field(default_factory=list)
    background: Optional[pd.DataFrame] = None  # SHAP background sample
    warnings: List[str] = field(default_factory=list)


def _expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(prob, bins[1:-1])
    ece = 0.0
    n = len(prob)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def _fit_lightgbm(X_tr, y_tr):
    import lightgbm as lgb
    pos = max(1, int(y_tr.sum()))
    neg = max(1, int((1 - y_tr).sum()))
    spw = neg / pos
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=48,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_lambda=2.0, min_child_samples=30, scale_pos_weight=spw,
        random_state=SETTINGS.random_seed, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


def _try_tabpfn(X_tr, y_tr):
    try:
        from tabpfn import TabPFNClassifier  # type: ignore
    except Exception:
        return None
    try:
        clf = TabPFNClassifier(device="cpu")
        clf.fit(X_tr.values, y_tr.values)
        return clf
    except Exception:
        return None


def train_and_score(bundle: FeatureBundle,
                    progress=lambda *a, **k: None) -> ModelArtifacts:
    X = bundle.X
    y = bundle.y
    warnings: List[str] = list(bundle.warnings)
    n = len(X)

    if y is None:
        raise ValueError("target is required for training")
    if y.nunique() < 2:
        # degenerate: only one class — cannot train a discriminative model
        warnings.append("Every loan in this file has the same outcome, so the model "
                        "can’t tell defaults from repayments — check the target column.")
        prior = float(y.mean())
        return ModelArtifacts(
            model_type="prior", pd_all=np.full(n, prior),
            metrics={"auc": 0.5, "pr_auc": prior, "brier": prior * (1 - prior),
                     "ece": 0.0, "n_loans": n},
            feature_names=bundle.feature_names, warnings=warnings,
        )

    # ---- training sample (cap) + scoring set (all rows) ------------------
    score_X = X
    train_idx = np.arange(n)
    if n > SETTINGS.train_row_cap:
        warnings.append(f"Large dataset: trained on a {SETTINGS.train_row_cap:,}-loan "
                        f"sample and scored all {n:,} loans (keeps training fast without "
                        f"losing accuracy).")
        rng = np.random.RandomState(SETTINGS.random_seed)
        # stratified subsample
        train_idx = _stratified_sample(y.values, SETTINGS.train_row_cap, rng)
    Xs = X.iloc[train_idx]
    ys = y.iloc[train_idx]

    # stratified train/val split
    progress("train", 35, "Splitting train / validation")
    X_tr, X_val, y_tr, y_val = train_test_split(
        Xs, ys, test_size=0.25, stratify=ys, random_state=SETTINGS.random_seed)

    # ---- model router ----------------------------------------------------
    model_type = "lightgbm"
    raw = None
    if n <= TABPFN_MAX_ROWS and X.shape[1] <= TABPFN_MAX_FEATURES:
        progress("train", 42, "Trying TabPFN foundation model")
        raw = _try_tabpfn(X_tr, y_tr)
        if raw is not None:
            model_type = "tabpfn"
    if raw is None:
        progress("train", 45, "Training LightGBM gradient boosting")
        raw = _fit_lightgbm(X_tr, y_tr)
        model_type = "lightgbm"

    # ---- calibrate (isotonic, prefit on validation) ----------------------
    progress("train", 60, "Calibrating probabilities (isotonic)")
    try:
        if _HAS_FROZEN:
            calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method="isotonic")
        else:
            calibrated = CalibratedClassifierCV(raw, method="isotonic", cv="prefit")
        calibrated.fit(X_val, y_val)
        val_prob = calibrated.predict_proba(X_val)[:, 1]
    except Exception:  # noqa: BLE001
        warnings.append("Probability calibration was skipped for this run; "
                        "scores use the model's raw probabilities.")
        calibrated = None
        val_prob = _proba(raw, X_val)

    # ---- metrics on validation ------------------------------------------
    metrics = _metrics(y_val.values, val_prob, n)

    # ---- score ALL rows (batched) ---------------------------------------
    progress("train", 72, "Scoring all loans")
    pd_all = _score_all(calibrated or raw, score_X, calibrated is not None)

    # SHAP background sample (small, for KernelSHAP fallback / TabPFN)
    bg = Xs.sample(min(100, len(Xs)), random_state=SETTINGS.random_seed)

    return ModelArtifacts(
        model_type=model_type, pd_all=pd_all, metrics=metrics,
        raw_model=raw, calibrated=calibrated, feature_names=bundle.feature_names,
        background=bg, warnings=warnings,
    )


def _metrics(y_true: np.ndarray, prob: np.ndarray, n_loans: int) -> dict:
    try:
        auc = float(roc_auc_score(y_true, prob))
    except Exception:
        auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, prob))
    except Exception:
        pr_auc = float(y_true.mean())
    try:
        brier = float(brier_score_loss(y_true, prob))
    except Exception:
        brier = 0.25
    ece = _expected_calibration_error(y_true, prob)
    return {"auc": round(auc, 4), "pr_auc": round(pr_auc, 4),
            "brier": round(brier, 4), "ece": round(ece, 4), "n_loans": int(n_loans)}


def _proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    return np.asarray(model.predict(X), dtype=float)


def _score_all(model, X, is_calibrated: bool, batch: int = 50000) -> np.ndarray:
    out = np.empty(len(X), dtype=float)
    for start in range(0, len(X), batch):
        chunk = X.iloc[start:start + batch]
        out[start:start + len(chunk)] = _proba(model, chunk)
    return np.clip(out, 1e-4, 1 - 1e-4)


def _stratified_sample(y: np.ndarray, cap: int, rng) -> np.ndarray:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    frac = cap / len(y)
    n_pos = max(1, int(len(pos_idx) * frac))
    n_neg = max(1, cap - n_pos)
    pos_s = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    neg_s = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    idx = np.concatenate([pos_s, neg_s])
    rng.shuffle(idx)
    return idx


def risk_band(pd_value: float) -> str:
    if pd_value >= SETTINGS.band_high:
        return "high"
    if pd_value >= SETTINGS.band_medium:
        return "medium"
    return "low"
