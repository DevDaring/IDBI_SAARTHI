"""
Ingest: read an uploaded tabular file and profile its columns.

Uses polars for fast CSV reading where possible, falling back to pandas. Profiles
each column with dtype, 5 non-null samples and null %. Supports .csv, .xlsx,
.parquet.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

ALLOWED_EXT = {".csv", ".xlsx", ".xls", ".parquet"}


def read_table(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read a file into a pandas DataFrame, using polars for big CSVs."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:  # csv (default)
        try:
            import polars as pl
            # polars handles messy/large CSVs fast; infer schema generously
            lf = pl.scan_csv(path, infer_schema_length=10000,
                             ignore_errors=True, try_parse_dates=True)
            if max_rows:
                lf = lf.limit(max_rows)
            df = lf.collect().to_pandas()
        except Exception:
            df = pd.read_csv(path, low_memory=False,
                             nrows=max_rows, on_bad_lines="skip")
    # normalise column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _json_safe(v):
    """Coerce a sample cell to a JSON-serialisable scalar."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else round(f, 4)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    s = str(v)
    return s if len(s) <= 80 else s[:77] + "..."


def profile_columns(df: pd.DataFrame, n_samples: int = 5) -> List[dict]:
    """Return per-column profile: name, dtype, sample[5], null_pct."""
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        null_pct = float(s.isna().mean() * 100.0) if n else 0.0
        dtype = _friendly_dtype(s)
        samples = [_json_safe(v) for v in s.dropna().head(n_samples).tolist()]
        # pad to n_samples for a stable UI grid
        while len(samples) < n_samples:
            samples.append(None)
        cols.append({
            "name": str(c),
            "dtype": dtype,
            "sample": samples,
            "null_pct": round(null_pct, 1),
            "n_unique": int(s.nunique(dropna=True)),
        })
    return cols


def _friendly_dtype(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    # object: decide categorical vs text vs numeric-as-string
    nun = s.nunique(dropna=True)
    if nun <= max(20, int(0.05 * max(1, len(s)))):
        return "categorical"
    # try numeric coercion
    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.notna().mean() > 0.8:
        return "numeric_text"
    return "text"


def profile_file(path: str, filename: str) -> dict:
    df = read_table(path, max_rows=5000)   # profiling sample only
    # n_rows of the full file (cheap line count for csv)
    n_rows = _count_rows(path) or len(df)
    return {
        "filename": filename,
        "n_rows": int(n_rows),
        "columns": profile_columns(df),
    }


def _count_rows(path: str) -> Optional[int]:
    ext = os.path.splitext(path)[1].lower()
    if ext != ".csv":
        return None
    try:
        with open(path, "rb") as f:
            count = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
        return max(0, count - 1)   # minus header
    except Exception:  # noqa: BLE001
        return None
