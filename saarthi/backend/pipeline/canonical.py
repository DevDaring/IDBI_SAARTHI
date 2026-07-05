"""Canonical loan schema definitions shared by the mapper and feature builder."""
from __future__ import annotations

# canonical field -> (meaning, role)  where role in {feature, protected, target, id, text, time}
CANONICAL = {
    "loan_id":            ("unique id (generate if absent)", "id"),
    "loan_amount":        ("principal", "feature"),
    "term_months":        ("loan tenure in months", "feature"),
    "interest_rate":      ("interest rate", "feature"),
    "income_or_turnover": ("borrower income / firm turnover", "feature"),
    "dscr":               ("debt service coverage ratio", "feature"),
    "credit_score":       ("bureau credit score", "feature"),
    "sector":             ("industry / MSME category", "feature"),
    "collateral_value":   ("security / collateral value", "feature"),
    "prior_delinquencies": ("count of past late payments", "feature"),
    "employment_length":  ("employment / business stability (years)", "feature"),
    "text_purpose":       ("loan purpose / description text", "text"),
    "region":             ("state / district", "protected"),
    "gender":             ("proprietor gender", "protected"),
    "community":          ("caste / community", "protected"),
    "time_observed":      ("months observed / vintage", "time"),
    "target":             ("default / charge-off label (binary)", "target"),
}

FEATURE_FIELDS = [k for k, (_, r) in CANONICAL.items() if r in ("feature", "text", "time")]
PROTECTED_FIELDS = [k for k, (_, r) in CANONICAL.items() if r == "protected"]
ACTIONABLE_FIELDS = ["term_months", "collateral_value", "income_or_turnover", "interest_rate"]

# keyword hints used by the deterministic fallback mapper
HINTS = {
    "loan_id": ["loan_id", "id", "loanid", "account", "acct", "applicationid", "loannr", "loannumber"],
    "loan_amount": ["amount", "loan_amnt", "principal", "disbursed", "grossapproval", "sba_appv", "funded", "credit_amount"],
    "term_months": ["term", "tenure", "duration", "months", "maturity"],
    "interest_rate": ["int_rate", "interest", "rate", "apr"],
    "income_or_turnover": ["income", "turnover", "revenue", "salary", "annual_inc", "sales", "ebitda"],
    "dscr": ["dscr", "debt_service", "coverage", "dti"],
    "credit_score": ["credit_score", "fico", "cibil", "bureau", "score"],
    "sector": ["sector", "industry", "naics", "purpose_category", "business_type", "occupation"],
    "collateral_value": ["collateral", "security", "asset_value", "property", "guarantee"],
    "prior_delinquencies": ["delinq", "late", "default_count", "past_due", "overdue", "dpd"],
    "employment_length": ["emp_length", "employment", "job_years", "tenure_emp", "years_in_business"],
    "text_purpose": ["purpose", "description", "desc", "title", "memo", "notes", "remarks"],
    "region": ["state", "region", "district", "city", "zip", "location", "province"],
    "gender": ["gender", "sex"],
    "community": ["community", "caste", "religion", "ethnic", "race"],
    "time_observed": ["vintage", "observed", "age_months", "months_on_book", "tenure_obs", "issue_d"],
    "target": ["default", "target", "label", "charge_off", "chargeoff", "mis_status",
               "is_default", "credit_risk", "risk_flag", "default_flag", "outcome",
               "good_bad", "loan_status", "class", "bad"],
}
