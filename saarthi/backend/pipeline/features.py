"""
Feature engineering: turn a raw uploaded DataFrame + canonical mapping into a
model-ready matrix.

Hard rules
----------
* Protected attributes (region, gender, community) and ids are NEVER features.
  They are split out for the fairness audit only.
* The binary `target` is robustly coerced to {0,1} with 1 = default/charge-off,
  and the detected class mapping is reported so the UI can be honest about it.
* Categoricals are one-hot encoded with a top-N cap (rare levels -> __OTHER__);
  numerics are median-imputed. The fitted preprocessor scores every row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.canonical import CANONICAL, PROTECTED_FIELDS

# words that indicate the DEFAULT / bad outcome (class 1)
_DEFAULT_WORDS = {
    "default", "defaulted", "chgoff", "charge off", "charged off", "chargeoff",
    "bad", "yes", "y", "true", "1", "late", "delinquent", "npa", "overdue",
    "writeoff", "write off", "loss",
}
_GOOD_WORDS = {
    "paid", "p i f", "pif", "fully paid", "good", "no", "n", "false", "0",
    "current", "clp", "ontime", "on time", "closed", "settled",
}

MAX_CATEGORY_LEVELS = 25
MAX_FEATURE_COLUMNS = 400


@dataclass
class FeatureBundle:
    X: pd.DataFrame                       # numeric design matrix (all rows)
    y: Optional[pd.Series]                # binary target (None if not training)
    feature_names: List[str]
    loan_ids: pd.Series
    protected: pd.DataFrame               # protected attrs, audit-only
    raw_features: pd.DataFrame            # human-readable feature values for evidence
    canonical_present: Dict[str, str]     # canonical_field -> source column actually used
    target_mapping: Dict[str, int] = field(default_factory=dict)  # raw value -> 0/1
    warnings: List[str] = field(default_factory=list)
    # encoder metadata for scoring/recourse
    numeric_cols: List[str] = field(default_factory=list)
    categorical_levels: Dict[str, List[str]] = field(default_factory=dict)
    medians: Dict[str, float] = field(default_factory=dict)


def binarize_target(s: pd.Series) -> Tuple[pd.Series, Dict[str, int], List[str]]:
    """Coerce an arbitrary binary-ish column to {0,1} with 1 = default."""
    warnings: List[str] = []
    vals = pd.Series(s.dropna().unique())
    raw_to_bin: Dict[str, int] = {}

    def classify(v) -> Optional[int]:
        t = str(v).strip().lower()
        if t in _DEFAULT_WORDS:
            return 1
        if t in _GOOD_WORDS:
            return 0
        # substring checks
        if any(w in t for w in ("chgoff", "charge", "default", "delinq", "npa", "writeoff", "loss")):
            return 1
        if any(w in t for w in ("paid", "pif", "good", "current", "closed", "settled")):
            return 0
        return None

    # 1) keyword classification
    classified = {str(v): classify(v) for v in vals}
    if all(c is not None for c in classified.values()) and len(set(classified.values())) == 2:
        raw_to_bin = {k: int(v) for k, v in classified.items()}  # type: ignore
        defaults = ", ".join(k for k, v in raw_to_bin.items() if v == 1)
        repaid = ", ".join(k for k, v in raw_to_bin.items() if v == 0)
        warnings.append(f"Read the default column automatically — “{defaults}” means "
                        f"defaulted and “{repaid}” means repaid.")
        return s.astype(str).map(raw_to_bin).astype("float"), raw_to_bin, warnings

    # 2) numeric handling
    numeric = pd.to_numeric(s, errors="coerce")
    nun = numeric.dropna().nunique()
    if nun == 2:
        uniq = sorted(numeric.dropna().unique())
        if set(uniq) == {0, 1}:
            raw_to_bin = {"0": 0, "1": 1}
            return numeric.astype("float"), raw_to_bin, warnings
        if set(uniq) == {1, 2}:
            # common convention (e.g. German credit): 1 = good, 2 = bad
            raw_to_bin = {"1": 0, "2": 1}
            warnings.append("Read the default column: value “2” = defaulted, “1” = repaid.")
            return numeric.map({1: 0, 2: 1}).astype("float"), raw_to_bin, warnings
        # generic two-value numeric: higher value = default
        hi = max(uniq)
        mapped = (numeric == hi).astype("float")
        raw_to_bin = {str(uniq[0]): 0, str(hi): 1}
        warnings.append(f"Read the default column: value “{hi}” treated as defaulted.")
        return mapped, raw_to_bin, warnings

    # 3) fallback: minority class = default
    counts = s.astype(str).value_counts()
    if len(counts) >= 2:
        minority = counts.index[-1]
        raw_to_bin = {str(v): (1 if str(v) == str(minority) else 0) for v in vals}
        warnings.append(f"The default column wasn’t clearly yes/no, so the rarer "
                        f"value (“{minority}”) was treated as ‘defaulted’ — please "
                        f"double-check the target mapping if that looks wrong.")
        return s.astype(str).map(raw_to_bin).astype("float"), raw_to_bin, warnings

    warnings.append("could not binarize target; all zeros")
    return pd.Series(np.zeros(len(s)), index=s.index), {}, warnings


def _clean_numeric(s: pd.Series) -> pd.Series:
    """Coerce a possibly-dirty numeric column (strip $, commas, %)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float")
    cleaned = s.astype(str).str.replace(r"[,$%\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def build_features(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    target_col: Optional[str],
    protected_cols: List[str],
    for_training: bool = True,
) -> FeatureBundle:
    warnings: List[str] = []
    n = len(df)

    # ---- loan ids --------------------------------------------------------
    id_src = mapping.get("loan_id")
    if id_src and id_src in df.columns:
        loan_ids = df[id_src].astype(str)
        # ensure uniqueness
        if loan_ids.duplicated().any():
            loan_ids = pd.Series([f"{v}__{i}" for i, v in enumerate(loan_ids)], index=df.index)
    else:
        loan_ids = pd.Series([f"L{i:06d}" for i in range(n)], index=df.index)

    # ---- target ----------------------------------------------------------
    y = None
    target_mapping: Dict[str, int] = {}
    if for_training and target_col and target_col in df.columns:
        y, target_mapping, twarn = binarize_target(df[target_col])
        warnings += twarn
        # drop rows with unknown target
        if y.isna().any():
            warnings.append(f"{int(y.isna().sum()):,} rows were skipped because their "
                            f"default value was blank or unreadable.")

    # ---- determine feature source columns (exclude id/target/protected) --
    protected_present = [c for c in protected_cols if c in df.columns]
    # also exclude any canonical protected mapping
    for pf in PROTECTED_FIELDS:
        src = mapping.get(pf)
        if src and src in df.columns and src not in protected_present:
            protected_present.append(src)
    exclude = set(protected_present)
    if id_src:
        exclude.add(id_src)
    if target_col:
        exclude.add(target_col)

    # which canonical FEATURE fields are present
    canonical_present: Dict[str, str] = {}
    feature_src_cols: List[str] = []
    for canon, (_, role) in CANONICAL.items():
        if role not in ("feature", "text", "time"):
            continue
        src = mapping.get(canon)
        if src and src in df.columns and src not in exclude:
            canonical_present[canon] = src
            feature_src_cols.append(src)

    # If the mapping is sparse, fall back to using all non-excluded columns so
    # the model still has signal (keeps the app robust on unmapped datasets).
    if len(feature_src_cols) < 2:
        warnings.append("Only a few columns matched the standard schema, so all other "
                        "(non-protected) columns were also used as model inputs.")
        for c in df.columns:
            if c not in exclude and c not in feature_src_cols:
                feature_src_cols.append(c)
                canonical_present.setdefault(c, c)

    # ---- build raw feature frame (human readable) ------------------------
    # map source column -> canonical name for readability where possible
    src_to_canon = {v: k for k, v in canonical_present.items()}
    raw = pd.DataFrame(index=df.index)
    numeric_cols: List[str] = []
    categorical_levels: Dict[str, List[str]] = {}
    medians: Dict[str, float] = {}

    text_src = mapping.get("text_purpose")

    design_parts: List[pd.DataFrame] = []
    for col in feature_src_cols:
        s = df[col]
        canon_name = src_to_canon.get(col, col)
        raw[canon_name] = s

        # text purpose -> simple distress-signal numeric feature
        if col == text_src:
            feat = _text_distress_score(s)
            design_parts.append(feat.to_frame(f"{canon_name}__distress"))
            numeric_cols.append(f"{canon_name}__distress")
            continue

        # decide numeric vs categorical
        num = _clean_numeric(s)
        is_numeric = num.notna().mean() > 0.6 and s.nunique(dropna=True) > 10
        # but keep known-numeric canonical fields numeric
        if canon_name in ("loan_amount", "term_months", "interest_rate",
                          "income_or_turnover", "dscr", "credit_score",
                          "collateral_value", "prior_delinquencies",
                          "employment_length", "time_observed"):
            is_numeric = num.notna().mean() > 0.3

        if is_numeric:
            med = float(np.nanmedian(num)) if num.notna().any() else 0.0
            medians[canon_name] = med
            filled = num.fillna(med)
            design_parts.append(filled.to_frame(canon_name))
            numeric_cols.append(canon_name)
        else:
            cat = s.astype(str).fillna("__NA__")
            top = cat.value_counts().head(MAX_CATEGORY_LEVELS).index.tolist()
            categorical_levels[canon_name] = top
            cat = cat.where(cat.isin(top), "__OTHER__")
            dummies = pd.get_dummies(cat, prefix=canon_name, dtype=float)
            design_parts.append(dummies)

    X = pd.concat(design_parts, axis=1) if design_parts else pd.DataFrame(index=df.index)
    # cap feature width (keep most-varying columns)
    if X.shape[1] > MAX_FEATURE_COLUMNS:
        variances = X.var().sort_values(ascending=False)
        keep = variances.head(MAX_FEATURE_COLUMNS).index.tolist()
        X = X[keep]
        warnings.append(f"Kept the {MAX_FEATURE_COLUMNS} most informative input columns "
                        f"to stay fast.")
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X.columns = [re.sub(r"[^0-9A-Za-z_]", "_", str(c)) for c in X.columns]

    # align y with X (drop unmappable-target rows from BOTH)
    if y is not None:
        mask = y.notna()
        if not mask.all():
            X = X[mask.values]
            raw = raw[mask.values]
            loan_ids = loan_ids[mask.values]
            protected_frame_idx = df.index[mask.values]
            y = y[mask].astype(int)
        else:
            protected_frame_idx = df.index
            y = y.astype(int)
    else:
        protected_frame_idx = df.index

    protected_frame = df.loc[protected_frame_idx, protected_present].copy() if protected_present else pd.DataFrame(index=protected_frame_idx)
    # rename protected columns to canonical names where known
    ren = {}
    for pf in PROTECTED_FIELDS:
        src = mapping.get(pf)
        if src in protected_frame.columns:
            ren[src] = pf
    protected_frame = protected_frame.rename(columns=ren)

    return FeatureBundle(
        X=X, y=y, feature_names=list(X.columns), loan_ids=loan_ids.reset_index(drop=True),
        protected=protected_frame.reset_index(drop=True),
        raw_features=raw.reset_index(drop=True),
        canonical_present=canonical_present, target_mapping=target_mapping,
        warnings=warnings, numeric_cols=numeric_cols,
        categorical_levels=categorical_levels, medians=medians,
    )


# ---------------------------------------------------------------------------
# simple lexical distress score for the text_purpose column
# ---------------------------------------------------------------------------
_DISTRESS_TERMS = [
    "loss", "debt", "overdue", "default", "struggle", "struggling", "emergency",
    "urgent", "shortfall", "delay", "delayed", "cash crunch", "closure", "closed",
    "covid", "pandemic", "flood", "drought", "lawsuit", "dispute", "refinance",
    "consolidate", "consolidation", "medical", "repay debt", "pay off",
]


def _text_distress_score(s: pd.Series) -> pd.Series:
    text = s.astype(str).str.lower().fillna("")
    score = pd.Series(0.0, index=s.index)
    for term in _DISTRESS_TERMS:
        score += text.str.contains(re.escape(term), regex=True).astype(float)
    return score
