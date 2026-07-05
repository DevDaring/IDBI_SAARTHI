"""
Generate realistic synthetic demo datasets so SAARTHI runs end-to-end offline.

Two files (deliberately DIFFERENT column names to exercise the LLM mapper):
  data/msme_demo.csv          ~3000 rows, MSME loan book with protected attrs
  data/credit_applicants.csv  ~1000 rows, German-credit-style naming

The target has genuine signal (DSCR, credit score, delinquencies, leverage,
sector) so the model gets a strong AUC and SHAP has real drivers. A mild,
sector-mediated correlation with protected attributes gives the fairness audit
something non-trivial to report.
"""
import os
import numpy as np
import pandas as pd

SEED = 20260502
rng = np.random.RandomState(SEED)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

SECTORS = ["Manufacturing", "Retail Trade", "Textiles", "Food Processing",
           "Construction", "IT Services", "Logistics", "Agriculture"]
SECTOR_RISK = {"Manufacturing": 0.0, "Retail Trade": 0.3, "Textiles": 0.6,
               "Food Processing": 0.1, "Construction": 0.7, "IT Services": -0.4,
               "Logistics": 0.2, "Agriculture": 0.5}
STATES = ["Maharashtra", "Gujarat", "Tamil Nadu", "Karnataka", "West Bengal",
          "Uttar Pradesh", "Punjab", "Kerala"]
PURPOSES = [
    "Purchase of new machinery to expand production capacity",
    "Working capital for raw material procurement",
    "Refinance existing high-interest debt and consolidate dues",
    "Open a second retail outlet in the district",
    "Bridge a seasonal cash crunch ahead of festival demand",
    "Upgrade delivery fleet and cold-storage units",
    "Emergency funds after delayed receivables from buyers",
    "Expand textile weaving unit with additional looms",
]


def make_msme(n=3000):
    sector = rng.choice(SECTORS, size=n, p=_norm([3, 4, 2, 2, 2, 2, 2, 1]))
    # gender mildly concentrated by sector (creates an auditable disparity)
    gender = []
    for s in sector:
        p_f = 0.55 if s in ("Textiles", "Food Processing") else 0.28
        gender.append("Female" if rng.rand() < p_f else "Male")
    gender = np.array(gender)
    state = rng.choice(STATES, size=n)

    loan_amount = np.round(rng.lognormal(mean=13.2, sigma=0.7, size=n), -3)
    term_months = rng.choice([12, 24, 36, 48, 60, 84], size=n,
                             p=_norm([2, 3, 4, 3, 2, 1]))
    interest_rate = np.round(rng.normal(13.5, 2.5, n).clip(8, 24), 2)
    turnover = np.round(loan_amount * rng.uniform(1.5, 6.0, n), -3)
    dscr = np.round(rng.gamma(shape=4.0, scale=0.35, size=n).clip(0.3, 3.5), 2)
    credit_score = rng.normal(690, 70, n).clip(450, 900).astype(int)
    collateral = np.round(loan_amount * rng.uniform(0.2, 1.8, n), -3)
    prior_delinq = rng.poisson(0.6, n).clip(0, 8)
    emp_years = np.round(rng.gamma(2.2, 2.5, n).clip(0, 30), 1)
    vintage = rng.randint(1, 60, n)
    purpose = rng.choice(PURPOSES, size=n)

    sect_risk = np.array([SECTOR_RISK[s] for s in sector])
    distress = np.array([1.0 if ("debt" in p or "crunch" in p or "Emergency" in p
                                 or "delayed" in p.lower()) else 0.0 for p in purpose])

    # latent default propensity (genuine signal)
    z = (
        -1.7
        - 1.7 * (dscr - 1.2)
        - 0.013 * (credit_score - 690)
        + 0.55 * prior_delinq
        + 1.0 * sect_risk
        + 0.0000012 * (loan_amount - collateral)
        + 0.08 * (interest_rate - 13.5)
        - 0.05 * emp_years
        + 0.6 * distress
        + 0.25 * (gender == "Female").astype(float)   # mild proxy effect for audit
        + rng.normal(0, 0.55, n)
    )
    p_default = 1 / (1 + np.exp(-z))
    defaulted = (rng.rand(n) < p_default).astype(int)

    df = pd.DataFrame({
        "LoanID": [f"MSME{100000 + i}" for i in range(n)],
        "DisbursedAmount": loan_amount,
        "TermMonths": term_months,
        "InterestRate": interest_rate,
        "AnnualTurnover": turnover,
        "DSCR": dscr,
        "BureauScore": credit_score,
        "Sector": sector,
        "CollateralValue": collateral,
        "PastDelinquencies": prior_delinq,
        "YearsInBusiness": emp_years,
        "LoanPurpose": purpose,
        "State": state,
        "ProprietorGender": gender,
        "MonthsOnBook": vintage,
        "Defaulted": defaulted,
    })
    return df


def make_credit_german_style(n=1000):
    """Different naming + 1/2 target to test the mapper + binarizer."""
    age = rng.randint(19, 72, n)
    sex = rng.choice(["male", "female"], size=n, p=[0.69, 0.31])
    duration = rng.choice([6, 12, 18, 24, 36, 48], size=n)
    amount = np.round(rng.lognormal(8.0, 0.6, n), 0)
    purpose = rng.choice(["car", "furniture", "business", "education",
                          "repairs", "radio/tv"], size=n)
    job_years = rng.gamma(2.0, 2.0, n).clip(0, 25).round(1)
    existing_credits = rng.poisson(0.5, n).clip(0, 4)
    savings = rng.choice(["<100", "100-500", "500-1000", ">1000", "unknown"], size=n)
    install_rate = rng.choice([1, 2, 3, 4], size=n)

    z = (
        -0.8
        + 0.030 * (duration - 18)
        + 0.00035 * (amount - 3000)
        + 0.7 * existing_credits
        + 0.4 * install_rate
        - 0.08 * job_years
        + 0.3 * (sex == "female").astype(float)
        + rng.normal(0, 0.62, n)
    )
    bad = (rng.rand(n) < 1 / (1 + np.exp(-z))).astype(int)
    # German convention: 1 = good, 2 = bad
    target = np.where(bad == 1, 2, 1)

    df = pd.DataFrame({
        "applicant_age": age,
        "personal_sex": sex,
        "credit_duration_months": duration,
        "credit_amount": amount,
        "purpose": purpose,
        "employment_years": job_years,
        "num_existing_credits": existing_credits,
        "savings_status": savings,
        "installment_rate_pct": install_rate,
        "credit_risk": target,    # 1=good, 2=bad
    })
    return df


def _norm(w):
    w = np.array(w, dtype=float)
    return w / w.sum()


if __name__ == "__main__":
    msme = make_msme()
    msme.to_csv(os.path.join(OUT, "msme_demo.csv"), index=False)
    print(f"msme_demo.csv: {msme.shape}, default rate = {msme.Defaulted.mean():.1%}")

    ger = make_credit_german_style()
    ger.to_csv(os.path.join(OUT, "credit_applicants.csv"), index=False)
    print(f"credit_applicants.csv: {ger.shape}, "
          f"bad rate = {(ger.credit_risk == 2).mean():.1%}")
