"""
Transaction-as-language: turn raw transaction records into token sequences.

Each borrower/account becomes an ordered sequence of "events". Every event is
described by a handful of categorical tokens plus a signed amount, exactly the
shape a sequence encoder (CoLES / TabBERT) consumes:

    client_id : str
    t         : int    relative event index
    amt_bucket: int    signed log-magnitude bucket  (the "word")
    kind      : int    transaction type / operation
    channel   : int    merchant category / k_symbol / chip usage
    dt_bucket : int    days since previous event, bucketed
    amount    : float  standardised signed amount

Three corpora are supported:
  tabformer - 24.4M synthetic card transactions   (no default label; PRETRAIN)
  berka     - 1.05M real bank transactions        (HAS default label; FINETUNE)
  amex      - 5.5M monthly statement rows         (HAS default label; FINETUNE)

Output: one parquet per corpus in Dataset/sequences/, plus a labels parquet
where a label exists.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

DATA = os.environ.get("SAARTHI_DATA", "/home/Debz/Hackathon/IDBI_Hackathon/Dataset")
OUT = f"{DATA}/sequences"
os.makedirs(OUT, exist_ok=True)

MAX_LEN = 128          # keep the most recent N events per client
N_AMT_BUCKETS = 32


def amount_bucket(a: np.ndarray) -> np.ndarray:
    """Signed log-magnitude bucket: the core 'word' of the transaction language."""
    a = np.asarray(a, dtype="float64")
    sign = np.sign(a)
    mag = np.log1p(np.abs(a))
    hi = np.nanpercentile(mag[np.isfinite(mag)], 99.5) or 1.0
    b = np.clip((mag / max(hi, 1e-6)) * (N_AMT_BUCKETS // 2 - 1), 0, N_AMT_BUCKETS // 2 - 1)
    out = (sign * b).astype("int16") + (N_AMT_BUCKETS // 2)
    return np.clip(out, 0, N_AMT_BUCKETS - 1).astype("int16")


def dt_bucket(days: np.ndarray) -> np.ndarray:
    edges = np.array([0, 1, 2, 3, 7, 14, 30, 60, 120, 365])
    return np.digitize(np.nan_to_num(days, nan=0.0), edges).astype("int16")


def _codes(s: pd.Series) -> np.ndarray:
    return s.astype("category").cat.codes.astype("int16").values


def _tail(df: pd.DataFrame, key: str, max_len: int = MAX_LEN) -> pd.DataFrame:
    """Keep the last `max_len` events per client (most recent behaviour matters)."""
    return df.groupby(key, sort=False, group_keys=False).tail(max_len)


def _standardise(g: pd.DataFrame) -> pd.DataFrame:
    m, s = g["amount"].mean(), g["amount"].std()
    g["amount"] = ((g["amount"] - m) / (s if s and np.isfinite(s) else 1.0)).astype("float32")
    return g


# ---------------------------------------------------------------------------
def build_tabformer(max_users: Optional[int] = None,
                    max_rows: Optional[int] = 12_000_000,
                    chunk: bool = True) -> str:
    """24.4M synthetic card transactions -> pretraining corpus (no labels).

    TabFormer is DEEP not WIDE (~2k users, thousands of txns each). Contrastive
    learning needs many distinct entities, so each user's history is split into
    consecutive MAX_LEN windows, every window becoming its own pseudo-client.
    """
    f = f"{DATA}/tabformer/card_transaction.v1.csv"
    use = ["User", "Card", "Year", "Month", "Day", "Amount", "Use Chip",
           "Merchant State", "MCC", "Errors?"]
    print(f"  reading {f} ...", flush=True)
    df = pd.read_csv(f, usecols=use, nrows=max_rows, low_memory=False)
    df["client_id"] = ("TF_" + df["User"].astype(str) + "_" + df["Card"].astype(str))
    if max_users:
        keep = df["client_id"].drop_duplicates().head(max_users)
        df = df[df["client_id"].isin(set(keep))]

    df["amount"] = pd.to_numeric(
        df["Amount"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    df = df.dropna(subset=["amount"])
    date = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day), errors="coerce")
    df["ts"] = (date - pd.Timestamp("1990-01-01")).dt.days.astype("float32")
    df = df.sort_values(["client_id", "ts"], kind="stable")
    if chunk:
        pos = df.groupby("client_id", sort=False).cumcount()
        df["client_id"] = df["client_id"] + "_w" + (pos // MAX_LEN).astype(str)
        # drop stub windows too short to split into two contrastive views
        sizes = df.groupby("client_id")["amount"].transform("size")
        df = df[sizes >= 16]
    else:
        df = _tail(df, "client_id")

    out = pd.DataFrame({
        "client_id": df["client_id"].values,
        "amount": df["amount"].values.astype("float32"),
        "amt_bucket": amount_bucket(df["amount"].values),
        "kind": _codes(df["Use Chip"].fillna("NA")),
        "channel": _codes(df["MCC"].astype(str)),
        "err": _codes(df["Errors?"].fillna("none").astype(str)),
        "ts": df["ts"].values,
    })
    out["dt"] = out.groupby("client_id")["ts"].diff().fillna(0).values
    out["dt_bucket"] = dt_bucket(out["dt"].values)
    out = out.groupby("client_id", group_keys=False).apply(_standardise)
    p = f"{OUT}/tabformer_seq.parquet"
    out.drop(columns=["ts", "dt"]).to_parquet(p, index=False)
    print(f"  tabformer: {out.client_id.nunique():,} clients, {len(out):,} events -> {p}",
          flush=True)
    return p


def build_berka(preloan_only: bool = False, suffix: str = "") -> str:
    """Real bank transactions WITH loan default labels - the end-to-end proof.

    `preloan_only=True` keeps only transactions strictly before the loan date.
    Required for any LABELLED evaluation: 71% of an account's transactions come
    after origination and encode the repayment behaviour that defines the label.
    The full-history version is still fine for unlabelled contrastive
    pre-training, where no label is ever consulted.
    """
    trans = pd.read_csv(f"{DATA}/berka/trans.csv", sep=";", low_memory=False)
    loan = pd.read_csv(f"{DATA}/berka/loan.csv", sep=";")

    if preloan_only:
        trans = (trans.merge(loan[["account_id", "date"]]
                             .rename(columns={"date": "__loan_date"}),
                             on="account_id", how="inner")
                 .query("date < __loan_date")
                 .drop(columns="__loan_date"))

    trans["ts"] = pd.to_numeric(trans["date"], errors="coerce")
    trans = trans.sort_values(["account_id", "ts"], kind="stable")
    trans["client_id"] = "BK_" + trans["account_id"].astype(str)
    trans["amount_signed"] = np.where(
        trans["type"].astype(str).str.startswith("V"),
        -trans["amount"].abs(), trans["amount"].abs())
    trans = _tail(trans, "client_id")

    out = pd.DataFrame({
        "client_id": trans["client_id"].values,
        "amount": trans["amount_signed"].values.astype("float32"),
        "amt_bucket": amount_bucket(trans["amount_signed"].values),
        "kind": _codes(trans["type"].fillna("NA").astype(str)),
        "channel": _codes(trans["k_symbol"].fillna("none").astype(str)),
        "err": _codes(trans["operation"].fillna("none").astype(str)),
        "ts": trans["ts"].values,
    })
    out["dt"] = out.groupby("client_id")["ts"].diff().fillna(0).values
    out["dt_bucket"] = dt_bucket(out["dt"].values)
    out = out.groupby("client_id", group_keys=False).apply(_standardise)
    p = f"{OUT}/berka{suffix}_seq.parquet"
    out.drop(columns=["ts", "dt"]).to_parquet(p, index=False)

    lab = pd.DataFrame({
        "client_id": "BK_" + loan["account_id"].astype(str),
        "target": loan["status"].isin(["B", "D"]).astype(int),
    })
    lab.to_parquet(f"{OUT}/berka{suffix}_labels.parquet", index=False)
    print(f"  berka: {out.client_id.nunique():,} clients, {len(out):,} events, "
          f"{len(lab):,} labels (rate {lab.target.mean():.4f}) -> {p}", flush=True)
    return p


def build_amex(max_customers: int = 60_000) -> str:
    """Monthly statement panel -> sequences. Uses the strongest numeric columns."""
    import pyarrow.parquet as pq
    lab = pd.read_csv(f"{DATA}/amex/train_labels.csv").head(max_customers)
    keep = set(lab["customer_ID"])

    pf = pq.ParquetFile(f"{DATA}/amex/train.parquet")
    cols = ["customer_ID", "S_2", "P_2", "B_1", "D_39", "B_2", "R_1", "S_3", "D_41"]
    have = [c for c in cols if c in pf.schema_arrow.names]
    parts = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=have).to_pandas()
        t = t[t["customer_ID"].isin(keep)]
        if len(t):
            parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    del parts

    df["ts"] = (pd.to_datetime(df["S_2"], errors="coerce")
                - pd.Timestamp("2017-01-01")).dt.days.astype("float32")
    df = df.sort_values(["customer_ID", "ts"], kind="stable")
    df["client_id"] = "AX_" + df["customer_ID"].astype(str).str[:16]
    amt = pd.to_numeric(df.get("P_2"), errors="coerce").fillna(0).values

    out = pd.DataFrame({
        "client_id": df["client_id"].values,
        "amount": amt.astype("float32"),
        "amt_bucket": amount_bucket(amt),
        "kind": pd.cut(pd.to_numeric(df.get("B_1"), errors="coerce").fillna(0),
                       bins=16, labels=False, duplicates="drop").fillna(0).astype("int16").values,
        "channel": pd.cut(pd.to_numeric(df.get("D_39"), errors="coerce").fillna(0),
                          bins=16, labels=False, duplicates="drop").fillna(0).astype("int16").values,
        "err": pd.cut(pd.to_numeric(df.get("R_1"), errors="coerce").fillna(0),
                      bins=8, labels=False, duplicates="drop").fillna(0).astype("int16").values,
        "ts": df["ts"].values,
    })
    out["dt"] = out.groupby("client_id")["ts"].diff().fillna(0).values
    out["dt_bucket"] = dt_bucket(out["dt"].values)
    out = out.groupby("client_id", group_keys=False).apply(_standardise)
    p = f"{OUT}/amex_seq.parquet"
    out.drop(columns=["ts", "dt"]).to_parquet(p, index=False)

    lab2 = pd.DataFrame({
        "client_id": "AX_" + lab["customer_ID"].astype(str).str[:16],
        "target": lab["target"].astype(int),
    }).drop_duplicates("client_id")
    lab2.to_parquet(f"{OUT}/amex_labels.parquet", index=False)
    print(f"  amex: {out.client_id.nunique():,} clients, {len(out):,} events -> {p}",
          flush=True)
    return p


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["berka", "amex", "tabformer"]
    for w in which:
        print(f"=== {w} ===", flush=True)
        try:
            {"tabformer": build_tabformer, "berka": build_berka,
             "amex": build_amex}[w]()
        except Exception as e:
            import traceback
            print(f"  FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
