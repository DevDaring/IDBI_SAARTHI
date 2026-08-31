"""
SAARTHI global model trainer.

Fixes the methodological flaw in the live app (calibration fitted and scored on
the same fold) by using a strict THREE-way split everywhere:

    fit (60%)  ->  trains the boosters
    cal (15%)  ->  fits the isotonic calibrator ONLY
    test(25%)  ->  never seen by either; all reported metrics come from here

Produces
--------
* specialist models  - one per corpus, full native features, headline AUCs
* pooled global model - canonical vocabulary across corpora, ships with the app
* leave-one-dataset-out transfer table
* calibration metrics that are actually honest (ECE/Brier on untouched test)

Artifacts land in  models/  as joblib bundles + a metrics.json report.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import adapters  # noqa: E402

SEED = 20260502
OUT = os.environ.get("SAARTHI_MODELS",
                     "/home/Debz/Hackathon/IDBI_Hackathon/saarthi/models")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def ece(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    tot, n = 0.0, len(p)
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def evaluate(y, p, prefix="") -> Dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    out = {
        f"{prefix}auc": float(roc_auc_score(y, p)),
        f"{prefix}pr_auc": float(average_precision_score(y, p)),
        f"{prefix}brier": float(brier_score_loss(y, p)),
        f"{prefix}ece": ece(y, p),
        f"{prefix}gini": float(2 * roc_auc_score(y, p) - 1),
        f"{prefix}n": int(len(y)),
        f"{prefix}base_rate": float(y.mean()),
    }
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# model zoo
# ---------------------------------------------------------------------------
def fit_lgbm(X, y, cat_cols, n_estimators=1200, lr=0.03, seed=SEED):
    import lightgbm as lgb
    spw = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))
    m = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=lr, num_leaves=64,
        min_child_samples=40, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=3.0, scale_pos_weight=spw,
        random_state=seed, n_jobs=-1, verbose=-1, max_bin=255,
    )
    m.fit(X, y, categorical_feature=cat_cols or "auto")
    return m


def fit_xgb(X, y, seed=SEED):
    try:
        import xgboost as xgb
    except ImportError:
        return None
    spw = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))
    m = xgb.XGBClassifier(
        n_estimators=900, learning_rate=0.04, max_depth=7,
        subsample=0.85, colsample_bytree=0.8, reg_lambda=3.0,
        scale_pos_weight=spw, random_state=seed, n_jobs=-1,
        tree_method="hist", enable_categorical=True, eval_metric="auc",
    )
    m.fit(X, y, verbose=False)
    return m


def fit_cat(X, y, cat_cols, seed=SEED):
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        return None
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str).fillna("__NA__")
    m = CatBoostClassifier(
        iterations=900, learning_rate=0.05, depth=7, l2_leaf_reg=4.0,
        random_seed=seed, verbose=0, auto_class_weights="Balanced",
        allow_writing_files=False,
    )
    m.fit(Pool(Xc, y, cat_features=cat_cols))
    return m


def predict(model, X, cat_cols) -> np.ndarray:
    name = type(model).__name__
    if name.startswith("CatBoost"):
        Xc = X.copy()
        for c in cat_cols:
            Xc[c] = Xc[c].astype(str).fillna("__NA__")
        return model.predict_proba(Xc)[:, 1]
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# the 3-way-split trainer
# ---------------------------------------------------------------------------
def train_three_way(X: pd.DataFrame, y: pd.Series, tag: str,
                    use_xgb=True, use_cat=True, seed=SEED) -> Dict:
    """fit / calibrate / test with no leakage between the three roles."""
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    strat = y if y.nunique() > 1 and y.value_counts().min() >= 3 else None

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=seed, stratify=strat)
    strat2 = y_tmp if y_tmp.nunique() > 1 and y_tmp.value_counts().min() >= 3 else None
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.625, random_state=seed, stratify=strat2)

    t0 = time.time()
    members, raw_cal, raw_te = {}, {}, {}

    m = fit_lgbm(X_tr, y_tr, cat_cols, seed=seed)
    members["lgbm"] = m
    raw_cal["lgbm"] = predict(m, X_cal, cat_cols)
    raw_te["lgbm"] = predict(m, X_te, cat_cols)

    if use_xgb:
        m = fit_xgb(X_tr, y_tr, seed=seed)
        if m is not None:
            members["xgb"] = m
            raw_cal["xgb"] = predict(m, X_cal, cat_cols)
            raw_te["xgb"] = predict(m, X_te, cat_cols)
    if use_cat:
        m = fit_cat(X_tr, y_tr, cat_cols, seed=seed)
        if m is not None:
            members["cat"] = m
            raw_cal["cat"] = predict(m, X_cal, cat_cols)
            raw_te["cat"] = predict(m, X_te, cat_cols)

    # ensemble = simple mean of members (robust, no extra fitting data needed)
    ens_cal = np.mean([raw_cal[k] for k in members], axis=0)
    ens_te = np.mean([raw_te[k] for k in members], axis=0)

    # isotonic calibrator fitted ONLY on the calibration fold
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(ens_cal, y_cal.values)
    cal_te = iso.predict(ens_te)

    metrics = {
        "uncalibrated_test": evaluate(y_te.values, ens_te),
        "calibrated_test": evaluate(y_te.values, cal_te),
        "per_member_test": {k: evaluate(y_te.values, raw_te[k]) for k in members},
        "split": {"fit": len(X_tr), "cal": len(X_cal), "test": len(X_te)},
        "train_seconds": round(time.time() - t0, 1),
        "members": list(members),
    }
    return {"tag": tag, "members": members, "calibrator": iso,
            "cat_cols": cat_cols, "features": list(X.columns),
            "metrics": metrics}


def bundle_predict(bundle, X: pd.DataFrame) -> np.ndarray:
    """Score new rows with a trained bundle (aligns columns, applies isotonic)."""
    X = X.reindex(columns=bundle["features"])
    for c in bundle["cat_cols"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    raw = np.mean([predict(m, X, bundle["cat_cols"])
                   for m in bundle["members"].values()], axis=0)
    return bundle["calibrator"].predict(raw)


# ---------------------------------------------------------------------------
# leave-one-dataset-out transfer
# ---------------------------------------------------------------------------
def leave_one_out(pool: Dict, seed=SEED) -> Dict:
    X, y, ds = pool["X"], pool["y"], pool["dataset"]
    results = {}
    for held in sorted(ds.unique()):
        tr = ds != held
        te = ds == held
        if y[tr].nunique() < 2 or y[te].nunique() < 2 or te.sum() < 50:
            continue
        cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
        Xtr, ytr = X[tr], y[tr]
        # small internal calibration slice from the training corpora
        Xf, Xc, yf, yc = train_test_split(
            Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
        m = fit_lgbm(Xf, yf, cat_cols, n_estimators=700, seed=seed)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(predict(m, Xc, cat_cols), yc.values)
        p = iso.predict(predict(m, X[te], cat_cols))
        results[held] = evaluate(y[te].values, p)
        print(f"    LODO  hold-out={held:14} AUC={results[held]['auc']:.4f} "
              f"n={results[held]['n']:,}", flush=True)
    return results


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--skip-specialists", action="store_true")
    ap.add_argument("--skip-pool", action="store_true")
    ap.add_argument("--skip-lodo", action="store_true")
    ap.add_argument("--cap", type=int, default=300_000)
    ap.add_argument("--amex-customers", type=int, default=120_000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import joblib
    os.makedirs(args.out, exist_ok=True)
    report = {"seed": SEED, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "specialists": {}, "pooled": {}, "lodo": {}}

    names = args.datasets or list(adapters.LOADERS)

    # ---- specialists --------------------------------------------------
    if not args.skip_specialists:
        for n in names:
            print(f"\n=== specialist: {n} ===", flush=True)
            try:
                kw = {"max_customers": args.amex_customers} if n == "amex" else {}
                c = adapters.load(n, **kw)
                print(f"    {c.summary()}", flush=True)
                b = train_three_way(c.native, c.y, tag=n)
                m = b["metrics"]
                print(f"    AUC(test)={m['calibrated_test']['auc']:.4f}  "
                      f"PR={m['calibrated_test']['pr_auc']:.4f}  "
                      f"ECE={m['calibrated_test']['ece']:.4f}  "
                      f"({m['train_seconds']}s)", flush=True)
                joblib.dump(b, f"{args.out}/specialist_{n}.joblib", compress=3)
                report["specialists"][n] = m
            except Exception as e:
                print(f"    FAILED {type(e).__name__}: {e}", flush=True)
                report["specialists"][n] = {"error": f"{type(e).__name__}: {e}"}
            with open(f"{args.out}/metrics.json", "w") as fh:
                json.dump(report, fh, indent=2)

    # ---- pooled global -------------------------------------------------
    if not args.skip_pool:
        print("\n=== pooled global (canonical vocabulary) ===", flush=True)
        poolnames = [n for n in adapters.POOLABLE if (not args.datasets or n in names)]
        pool = adapters.build_pool(poolnames, cap_per_dataset=args.cap)
        b = train_three_way(pool["X"], pool["y"], tag="global")
        m = b["metrics"]
        print(f"    GLOBAL AUC(test)={m['calibrated_test']['auc']:.4f}  "
              f"ECE={m['calibrated_test']['ece']:.4f}  "
              f"n={len(pool['y']):,}", flush=True)
        b["pool_datasets"] = poolnames
        b["canonical"] = adapters.CANON_ALL
        joblib.dump(b, f"{args.out}/global_canonical.joblib", compress=3)
        report["pooled"] = m
        report["pooled"]["datasets"] = poolnames

        if not args.skip_lodo:
            print("\n=== leave-one-dataset-out transfer ===", flush=True)
            report["lodo"] = leave_one_out(pool)

    with open(f"{args.out}/metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
