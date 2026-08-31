"""
Dataset adapters: every public corpus -> (native frame, canonical frame, y).

Two feature spaces are produced per dataset:

* NATIVE    - the dataset's own full feature set. Used to train per-dataset
              "specialist" models whose AUC is comparable to published
              benchmarks (that is the headline table).
* CANONICAL - a shared ~14-field credit vocabulary that every dataset can be
              projected onto. Used to train the POOLED global model and to run
              leave-one-dataset-out transfer tests (that is the "will it move
              to IDBI's book" evidence).

Target convention everywhere: 1 = default / charge-off / bad, 0 = repaid.
Protected attributes (state, gender, ...) are returned separately and never
enter either feature space.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATA = os.environ.get(
    "SAARTHI_DATA", "/home/Debz/Hackathon/IDBI_Hackathon/Dataset")

# ---------------------------------------------------------------------------
# canonical credit vocabulary (shared across corpora)
# ---------------------------------------------------------------------------
CANON_NUM = [
    "can_loan_amount",      # principal / credit amount
    "can_term_months",      # tenure
    "can_interest_rate",    # rate %
    "can_income",           # annual income / turnover
    "can_dti",              # debt-to-income or debt ratio
    "can_credit_score_n",   # bureau score normalised to [0,1]
    "can_age",              # borrower / principal age (years)
    "can_emp_length",       # employment or business stability (years)
    "can_delinq_count",     # count of past late payments
    "can_open_accounts",    # open credit lines
    "can_utilization",      # revolving utilisation %
    "can_n_employees",      # firm size proxy
    "can_loan_to_income",   # engineered ratio
    "can_installment",      # periodic payment
]
CANON_CAT = ["can_sector"]
CANON_ALL = CANON_NUM + CANON_CAT


@dataclass
class Corpus:
    name: str
    native: pd.DataFrame                 # full native features (no target)
    canonical: pd.DataFrame              # CANON_ALL columns (NaN where absent)
    y: pd.Series                         # int 0/1
    protected: pd.DataFrame = field(default_factory=pd.DataFrame)
    time_key: Optional[pd.Series] = None  # for temporal validation
    notes: str = ""

    def __len__(self):
        return len(self.y)

    def summary(self) -> str:
        return (f"{self.name:16} n={len(self.y):>9,}  "
                f"native={self.native.shape[1]:>4}  "
                f"default_rate={self.y.mean():.4f}  {self.notes}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _num(s) -> pd.Series:
    """Coerce a possibly dirty column to float."""
    if isinstance(s, pd.Series) and pd.api.types.is_numeric_dtype(s):
        return s.astype("float32")
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(r"[,$%\s]", "", regex=True),
        errors="coerce").astype("float32")


def _blank_canon(index) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    for c in CANON_NUM:
        df[c] = np.nan
    for c in CANON_CAT:
        df[c] = pd.Series([None] * len(index), index=index, dtype=object)
    return df


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numerics to float32 and object columns to pandas category."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("category")
        elif pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype("float32")
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("float32")
    return out


# ---------------------------------------------------------------------------
# 1. SBA National  - the MSME anchor
# ---------------------------------------------------------------------------
def load_sba(nrows: Optional[int] = None) -> Corpus:
    f = f"{DATA}/sba/SBAnational.csv"
    df = pd.read_csv(f, low_memory=False, nrows=nrows)
    df = df[df["MIS_Status"].isin(["P I F", "CHGOFF"])].reset_index(drop=True)
    y = (df["MIS_Status"] == "CHGOFF").astype(int)

    money = ["DisbursementGross", "BalanceGross", "GrAppv", "SBA_Appv", "ChgOffPrinGr"]
    for c in money:
        df[c] = _num(df[c])

    nat = pd.DataFrame(index=df.index)
    nat["term"] = _num(df["Term"])
    nat["no_emp"] = _num(df["NoEmp"])
    nat["new_exist"] = _num(df["NewExist"])
    nat["create_job"] = _num(df["CreateJob"])
    nat["retained_job"] = _num(df["RetainedJob"])
    nat["franchise"] = (_num(df["FranchiseCode"]) > 1).astype("float32")
    nat["urban_rural"] = _num(df["UrbanRural"])
    nat["rev_line_cr"] = df["RevLineCr"].astype(str).str.upper().str[:1]
    nat["low_doc"] = df["LowDoc"].astype(str).str.upper().str[:1]
    nat["disbursement"] = df["DisbursementGross"]
    nat["gr_appv"] = df["GrAppv"]
    nat["sba_appv"] = df["SBA_Appv"]
    nat["sba_portion"] = (df["SBA_Appv"] / df["GrAppv"].replace(0, np.nan)).astype("float32")
    nat["naics2"] = df["NAICS"].astype(str).str[:2]
    nat["bank_state_same"] = (df["BankState"].astype(str) == df["State"].astype(str)).astype("float32")
    ap = pd.to_datetime(df["ApprovalDate"], errors="coerce", format="mixed")
    nat["approval_year"] = ap.dt.year.astype("float32")

    can = _blank_canon(df.index)
    can["can_loan_amount"] = df["GrAppv"]
    can["can_term_months"] = nat["term"]
    can["can_n_employees"] = nat["no_emp"]
    can["can_sector"] = nat["naics2"]
    can["can_emp_length"] = np.where(nat["new_exist"] == 1.0, 5.0, 0.5)
    can["can_installment"] = df["GrAppv"] / nat["term"].replace(0, np.nan)

    prot = pd.DataFrame({"region": df["State"].astype(str)}, index=df.index)
    return Corpus("sba", _finish(nat), _finish(can), y, prot,
                  time_key=nat["approval_year"],
                  notes="US small-business loans; MSME anchor")


# ---------------------------------------------------------------------------
# 2. Lending Club
# ---------------------------------------------------------------------------
def load_lending_club(nrows: Optional[int] = None) -> Corpus:
    f = f"{DATA}/lending_club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
    use = ["loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
           "emp_length", "home_ownership", "annual_inc", "verification_status",
           "issue_d", "loan_status", "purpose", "dti", "delinq_2yrs",
           "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
           "pub_rec", "revol_bal", "revol_util", "total_acc", "addr_state",
           "application_type", "mort_acc", "pub_rec_bankruptcies"]
    df = pd.read_csv(f, low_memory=False, usecols=use, nrows=nrows)

    bad = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off",
           "Late (31-120 days)"]
    good = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
    df = df[df["loan_status"].isin(bad + good)].reset_index(drop=True)
    y = df["loan_status"].isin(bad).astype(int)

    nat = pd.DataFrame(index=df.index)
    nat["loan_amnt"] = _num(df["loan_amnt"])
    nat["term"] = _num(df["term"].astype(str).str.extract(r"(\d+)")[0])
    nat["int_rate"] = _num(df["int_rate"])
    nat["installment"] = _num(df["installment"])
    nat["grade"] = df["grade"].astype(str)
    nat["sub_grade"] = df["sub_grade"].astype(str)
    nat["emp_length"] = _num(df["emp_length"].astype(str).str.extract(r"(\d+)")[0])
    nat["home_ownership"] = df["home_ownership"].astype(str)
    nat["annual_inc"] = _num(df["annual_inc"])
    nat["verification_status"] = df["verification_status"].astype(str)
    nat["purpose"] = df["purpose"].astype(str)
    nat["dti"] = _num(df["dti"])
    nat["delinq_2yrs"] = _num(df["delinq_2yrs"])
    nat["fico"] = (_num(df["fico_range_low"]) + _num(df["fico_range_high"])) / 2
    nat["inq_last_6mths"] = _num(df["inq_last_6mths"])
    nat["open_acc"] = _num(df["open_acc"])
    nat["pub_rec"] = _num(df["pub_rec"])
    nat["revol_bal"] = _num(df["revol_bal"])
    nat["revol_util"] = _num(df["revol_util"])
    nat["total_acc"] = _num(df["total_acc"])
    nat["application_type"] = df["application_type"].astype(str)
    nat["mort_acc"] = _num(df["mort_acc"])
    nat["pub_rec_bankruptcies"] = _num(df["pub_rec_bankruptcies"])

    issue = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    can = _blank_canon(df.index)
    can["can_loan_amount"] = nat["loan_amnt"]
    can["can_term_months"] = nat["term"]
    can["can_interest_rate"] = nat["int_rate"]
    can["can_income"] = nat["annual_inc"]
    can["can_dti"] = nat["dti"]
    can["can_credit_score_n"] = ((nat["fico"] - 300) / 550).clip(0, 1)
    can["can_emp_length"] = nat["emp_length"]
    can["can_delinq_count"] = nat["delinq_2yrs"]
    can["can_open_accounts"] = nat["open_acc"]
    can["can_utilization"] = nat["revol_util"]
    can["can_installment"] = nat["installment"]
    can["can_sector"] = nat["purpose"]
    can["can_loan_to_income"] = nat["loan_amnt"] / nat["annual_inc"].replace(0, np.nan)

    prot = pd.DataFrame({"region": df["addr_state"].astype(str)}, index=df.index)
    return Corpus("lending_club", _finish(nat), _finish(can), y, prot,
                  time_key=issue.dt.year.astype("float32"),
                  notes="2007-2018 consumer loans; temporal split available")


# ---------------------------------------------------------------------------
# 3. Home Credit 2018 (application table + light bureau aggregates)
# ---------------------------------------------------------------------------
def load_home_credit(nrows: Optional[int] = None, with_bureau: bool = True) -> Corpus:
    f = f"{DATA}/home_credit/application_train.csv"
    df = pd.read_csv(f, nrows=nrows)
    y = df["TARGET"].astype(int)
    ids = df["SK_ID_CURR"]

    drop = ["TARGET", "SK_ID_CURR"]
    prot_cols = [c for c in ["CODE_GENDER"] if c in df.columns]
    nat = df.drop(columns=drop + prot_cols, errors="ignore")

    # a few strong engineered ratios used by every top Kaggle solution
    nat["ratio_credit_income"] = _num(df["AMT_CREDIT"]) / _num(df["AMT_INCOME_TOTAL"]).replace(0, np.nan)
    nat["ratio_annuity_income"] = _num(df["AMT_ANNUITY"]) / _num(df["AMT_INCOME_TOTAL"]).replace(0, np.nan)
    nat["credit_term"] = _num(df["AMT_CREDIT"]) / _num(df["AMT_ANNUITY"]).replace(0, np.nan)
    nat["days_employed_pct"] = _num(df["DAYS_EMPLOYED"]) / _num(df["DAYS_BIRTH"]).replace(0, np.nan)

    if with_bureau:
        bf = f"{DATA}/home_credit/bureau.csv"
        if os.path.exists(bf):
            b = pd.read_csv(bf, usecols=["SK_ID_CURR", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
                                         "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                                         "CREDIT_ACTIVE"])
            agg = b.groupby("SK_ID_CURR").agg(
                bureau_n=("DAYS_CREDIT", "size"),
                bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
                bureau_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
                bureau_sum=("AMT_CREDIT_SUM", "sum"),
                bureau_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
            )
            active = (b[b.CREDIT_ACTIVE == "Active"].groupby("SK_ID_CURR").size()
                      .rename("bureau_active_n"))
            agg = agg.join(active)
            nat = nat.join(agg, on=ids.name if ids.name in nat.columns else None) \
                if ids.name in nat.columns else nat.join(agg.reindex(ids.values).reset_index(drop=True))

    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    ext_mean = _num(df[ext].mean(axis=1)) if ext else pd.Series(np.nan, index=df.index)

    can = _blank_canon(df.index)
    can["can_loan_amount"] = _num(df["AMT_CREDIT"])
    can["can_income"] = _num(df["AMT_INCOME_TOTAL"])
    can["can_installment"] = _num(df["AMT_ANNUITY"])
    can["can_age"] = (-_num(df["DAYS_BIRTH"]) / 365.25)
    can["can_emp_length"] = (-_num(df["DAYS_EMPLOYED"]) / 365.25).clip(0, 60)
    can["can_credit_score_n"] = ext_mean.clip(0, 1)
    can["can_term_months"] = (_num(df["AMT_CREDIT"]) / _num(df["AMT_ANNUITY"]).replace(0, np.nan))
    can["can_loan_to_income"] = nat["ratio_credit_income"]
    can["can_sector"] = df.get("NAME_INCOME_TYPE", pd.Series(None, index=df.index)).astype(str)
    if "bureau_overdue_max" in nat.columns:
        can["can_delinq_count"] = (_num(nat["bureau_overdue_max"]) > 0).astype("float32")
        can["can_open_accounts"] = _num(nat.get("bureau_active_n"))

    prot = pd.DataFrame({"gender": df["CODE_GENDER"].astype(str)}, index=df.index) \
        if "CODE_GENDER" in df.columns else pd.DataFrame(index=df.index)
    return Corpus("home_credit", _finish(nat), _finish(can), y, prot,
                  notes="consumer credit + bureau aggregates")


# ---------------------------------------------------------------------------
# 4. Give Me Some Credit
# ---------------------------------------------------------------------------
def load_gmsc(nrows: Optional[int] = None) -> Corpus:
    df = pd.read_csv(f"{DATA}/gmsc/cs-training.csv", nrows=nrows)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.dropna(subset=["SeriousDlqin2yrs"]).reset_index(drop=True)
    y = df["SeriousDlqin2yrs"].astype(int)
    nat = df.drop(columns=["SeriousDlqin2yrs"])

    can = _blank_canon(df.index)
    can["can_age"] = _num(df["age"])
    can["can_income"] = _num(df["MonthlyIncome"]) * 12
    can["can_dti"] = _num(df["DebtRatio"])
    can["can_utilization"] = _num(df["RevolvingUtilizationOfUnsecuredLines"]) * 100
    can["can_open_accounts"] = _num(df["NumberOfOpenCreditLinesAndLoans"])
    can["can_delinq_count"] = (_num(df["NumberOfTime30-59DaysPastDueNotWorse"])
                               + _num(df["NumberOfTimes90DaysLate"])
                               + _num(df["NumberOfTime60-89DaysPastDueNotWorse"]))
    return Corpus("gmsc", _finish(nat), _finish(can), y,
                  notes="90+ DPD within 2 years")


# ---------------------------------------------------------------------------
# 5. Taiwan credit-card default (UCI 350)
# ---------------------------------------------------------------------------
TAIWAN_NAMES = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
                + [f"PAY_{i}" for i in range(0, 6)]
                + [f"BILL_AMT{i}" for i in range(1, 7)]
                + [f"PAY_AMT{i}" for i in range(1, 7)])


def load_taiwan(nrows: Optional[int] = None) -> Corpus:
    df = pd.read_csv(f"{DATA}/taiwan/taiwan.csv", nrows=nrows)
    df.columns = TAIWAN_NAMES + ["target"]
    y = df["target"].astype(int)
    nat = df.drop(columns=["target", "SEX"])
    pay_cols = [f"PAY_{i}" for i in range(0, 6)]
    nat["pay_max"] = df[pay_cols].max(axis=1)
    nat["pay_sum_late"] = (df[pay_cols] > 0).sum(axis=1)
    nat["util_ratio"] = _num(df["BILL_AMT1"]) / _num(df["LIMIT_BAL"]).replace(0, np.nan)
    nat["pay_ratio"] = _num(df["PAY_AMT1"]) / _num(df["BILL_AMT1"]).replace(0, np.nan)

    can = _blank_canon(df.index)
    can["can_loan_amount"] = _num(df["LIMIT_BAL"])
    can["can_age"] = _num(df["AGE"])
    can["can_delinq_count"] = nat["pay_sum_late"]
    can["can_utilization"] = (nat["util_ratio"] * 100).clip(0, 500)
    can["can_installment"] = _num(df["PAY_AMT1"])
    prot = pd.DataFrame({"gender": df["SEX"].map({1: "male", 2: "female"}).astype(str)},
                        index=df.index)
    return Corpus("taiwan", _finish(nat), _finish(can), y, prot,
                  notes="6-month repayment panel")


# ---------------------------------------------------------------------------
# 6. German credit (UCI 144)
# ---------------------------------------------------------------------------
def load_german(nrows: Optional[int] = None) -> Corpus:
    df = pd.read_csv(f"{DATA}/german/german.csv", nrows=nrows)
    y = (df["class"] == 2).astype(int) if df["class"].max() == 2 else df["class"].astype(int)
    nat = df.drop(columns=["class"])
    can = _blank_canon(df.index)
    can["can_term_months"] = _num(df["Attribute2"])
    can["can_loan_amount"] = _num(df["Attribute5"])
    can["can_age"] = _num(df["Attribute13"])
    can["can_open_accounts"] = _num(df["Attribute16"])
    can["can_installment"] = _num(df["Attribute8"])
    return Corpus("german", _finish(nat), _finish(can), y,
                  notes="UCI benchmark, 1000 rows")


# ---------------------------------------------------------------------------
# 7. Berka - loans WITH transaction sequences (sequence->default validation)
# ---------------------------------------------------------------------------
def load_berka(nrows: Optional[int] = None) -> Corpus:
    loan = pd.read_csv(f"{DATA}/berka/loan.csv", sep=";")
    acct = pd.read_csv(f"{DATA}/berka/account.csv", sep=";")
    trans = pd.read_csv(f"{DATA}/berka/trans.csv", sep=";", low_memory=False)
    # A finished-ok, C running-ok => 0 ; B finished-unpaid, D running-debt => 1
    y = loan["status"].isin(["B", "D"]).astype(int)

    tagg = trans.groupby("account_id").agg(
        txn_n=("amount", "size"),
        txn_amt_mean=("amount", "mean"),
        txn_amt_std=("amount", "std"),
        txn_amt_max=("amount", "max"),
        bal_mean=("balance", "mean"),
        bal_min=("balance", "min"),
        bal_std=("balance", "std"),
    )
    withdr = (trans[trans["type"].astype(str).str.startswith("V")]
              .groupby("account_id")["amount"].sum().rename("withdrawal_sum"))
    credit = (trans[trans["type"].astype(str).str.startswith("P")]
              .groupby("account_id")["amount"].sum().rename("credit_sum"))
    tagg = tagg.join(withdr).join(credit)
    tagg["cred_debit_ratio"] = tagg["credit_sum"] / tagg["withdrawal_sum"].replace(0, np.nan)
    neg = (trans.assign(neg=(trans["balance"] < 0).astype(int))
           .groupby("account_id")["neg"].mean().rename("neg_balance_frac"))
    tagg = tagg.join(neg)

    m = loan.merge(acct, on="account_id", how="left", suffixes=("", "_acct")) \
            .merge(tagg, on="account_id", how="left")

    nat = pd.DataFrame(index=m.index)
    nat["amount"] = _num(m["amount"])
    nat["duration"] = _num(m["duration"])
    nat["payments"] = _num(m["payments"])
    nat["frequency"] = m["frequency"].astype(str)
    for c in tagg.columns:
        nat[c] = _num(m[c])
    nat["payment_to_bal"] = nat["payments"] / nat["bal_mean"].replace(0, np.nan)

    can = _blank_canon(m.index)
    can["can_loan_amount"] = nat["amount"]
    can["can_term_months"] = nat["duration"]
    can["can_installment"] = nat["payments"]
    can["can_income"] = nat["credit_sum"]
    can["can_utilization"] = (nat["neg_balance_frac"] * 100)
    return Corpus("berka", _finish(nat), _finish(can), y,
                  notes="PKDD'99; has real txn sequences")


# ---------------------------------------------------------------------------
# 8. Amex - specialist only (anonymised features), aggregated per customer
# ---------------------------------------------------------------------------
def load_amex(nrows: Optional[int] = None, max_customers: Optional[int] = None) -> Corpus:
    import pyarrow.parquet as pq
    lab = pd.read_csv(f"{DATA}/amex/train_labels.csv")
    if max_customers:
        lab = lab.head(max_customers)
    keep = set(lab["customer_ID"])

    pf = pq.ParquetFile(f"{DATA}/amex/train.parquet")
    parts = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i).to_pandas()
        if max_customers:
            t = t[t["customer_ID"].isin(keep)]
        if len(t):
            parts.append(t)
        if nrows and sum(len(p) for p in parts) > nrows:
            break
    raw = pd.concat(parts, ignore_index=True)
    del parts

    feat = [c for c in raw.columns if c not in ("customer_ID", "S_2")]
    num = [c for c in feat if pd.api.types.is_numeric_dtype(raw[c])]
    g = raw.groupby("customer_ID")
    agg = g[num].agg(["mean", "std", "min", "max", "last"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_statements"] = g.size()

    lab = lab.set_index("customer_ID")
    agg = agg.join(lab, how="inner")
    y = agg.pop("target").astype(int).reset_index(drop=True)
    agg = agg.reset_index(drop=True)

    can = _blank_canon(agg.index)  # anonymised -> no canonical mapping
    return Corpus("amex", _finish(agg), _finish(can), y,
                  notes="anonymised statement panel; specialist only")


# ---------------------------------------------------------------------------
# 9. Home Credit 2024 stability - specialist only (base + light aggregates)
# ---------------------------------------------------------------------------
def load_hc2024(nrows: Optional[int] = None) -> Corpus:
    import glob
    base = pd.read_parquet(f"{DATA}/home_credit_2024/train/train_base.parquet")
    if nrows:
        base = base.head(nrows)
    y = base["target"].astype(int)

    nat = pd.DataFrame(index=base.index)
    nat["month"] = _num(base["MONTH"])
    nat["week_num"] = _num(base["WEEK_NUM"])

    # join the depth-0 static tables (one row per case_id)
    for pat in ["train_static_cb_0", "train_static_0"]:
        for f in sorted(glob.glob(f"{DATA}/home_credit_2024/train/{pat}*.parquet")):
            t = pd.read_parquet(f)
            t = t[t["case_id"].isin(set(base["case_id"]))]
            num = [c for c in t.columns
                   if c != "case_id" and pd.api.types.is_numeric_dtype(t[c])]
            if not num:
                continue
            t = t[["case_id"] + num].drop_duplicates("case_id").set_index("case_id")
            nat = nat.join(t.reindex(base["case_id"].values).reset_index(drop=True),
                           rsuffix=f"_{os.path.basename(f)[:12]}")

    nat = nat.loc[:, ~nat.columns.duplicated()]
    can = _blank_canon(base.index)
    return Corpus("hc2024", _finish(nat), _finish(can), y,
                  time_key=_num(base["WEEK_NUM"]),
                  notes="2024 stability comp; WEEK_NUM for drift metric")


# ---------------------------------------------------------------------------
LOADERS = {
    "sba": load_sba,
    "lending_club": load_lending_club,
    "home_credit": load_home_credit,
    "gmsc": load_gmsc,
    "taiwan": load_taiwan,
    "german": load_german,
    "berka": load_berka,
    "amex": load_amex,
    "hc2024": load_hc2024,
}
# corpora that project meaningfully onto the canonical vocabulary
POOLABLE = ["sba", "lending_club", "home_credit", "gmsc", "taiwan", "german", "berka"]


def load(name: str, **kw) -> Corpus:
    return LOADERS[name](**kw)


def build_pool(names: List[str] = None, cap_per_dataset: int = 300_000,
               seed: int = 20260502) -> Dict:
    """Concatenate canonical frames from several corpora into one training set."""
    names = names or POOLABLE
    Xs, ys, tags = [], [], []
    rng = np.random.RandomState(seed)
    for n in names:
        c = load(n)
        X, y = c.canonical, c.y
        if len(X) > cap_per_dataset:
            idx = rng.choice(len(X), cap_per_dataset, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        Xs.append(X)
        ys.append(y)
        tags.append(pd.Series([n] * len(X)))
        print(f"  pooled {n:14} +{len(X):>8,} rows  (rate {y.mean():.4f})", flush=True)
    X = pd.concat(Xs, ignore_index=True)
    for c in CANON_CAT:
        X[c] = X[c].astype("category")
    return {"X": X, "y": pd.concat(ys, ignore_index=True).reset_index(drop=True),
            "dataset": pd.concat(tags, ignore_index=True).reset_index(drop=True)}


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or list(LOADERS)
    for n in which:
        try:
            c = load(n)
            print(c.summary(), flush=True)
        except Exception as e:
            print(f"{n:16} FAILED: {type(e).__name__}: {e}", flush=True)
