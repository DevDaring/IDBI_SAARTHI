"""
Pydantic models for every JSON contract in SAARTHI.

These are the single source of truth: the LLM JSON layers validate against them,
the Flask API serialises them, and the React types mirror them.

Critical invariant: the LLM NEVER produces the PD. The ML model produces `pd`
and the SHAP-derived drivers; the LLM only writes reason codes (from a fixed
taxonomy), prose, and the recommended action.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Fixed reason-code taxonomy — the "common interpretation framework".
# Every loan is described in exactly this vocabulary so explanations are
# comparable across the whole portfolio.
# ---------------------------------------------------------------------------
REASON_CODES = [
    "LIQUIDITY_STRESS",
    "LEVERAGE_HIGH",
    "REVENUE_DECLINE",
    "REPAYMENT_HISTORY_POOR",
    "SECTOR_RISK",
    "COLLATERAL_LOW",
    "BEHAVIOUR_ANOMALY",
    "TEXT_DISTRESS_SIGNAL",
    "TENURE_RISK",
    "OTHER",
]
ReasonCode = Literal[
    "LIQUIDITY_STRESS", "LEVERAGE_HIGH", "REVENUE_DECLINE",
    "REPAYMENT_HISTORY_POOR", "SECTOR_RISK", "COLLATERAL_LOW",
    "BEHAVIOUR_ANOMALY", "TEXT_DISTRESS_SIGNAL", "TENURE_RISK", "OTHER",
]
Direction = Literal["increases_risk", "decreases_risk"]


# ---------------------------------------------------------------------------
# Mapping (mapper LLM output)
# ---------------------------------------------------------------------------
class MappingResult(BaseModel):
    mapping: Dict[str, Optional[str]] = Field(
        ..., description="canonical_field -> source_column (or null)")
    target: Optional[str] = None
    protected: List[str] = Field(default_factory=list)
    confidence: Dict[str, float] = Field(default_factory=dict)
    notes: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reason code (built by explain.py; LLM picks code + writes evidence)
# ---------------------------------------------------------------------------
class ReasonCodeItem(BaseModel):
    code: ReasonCode
    weight: float = Field(ge=0.0, le=1.0)
    direction: Direction
    evidence: str
    feature: str
    shap: float

    @field_validator("code", mode="before")
    @classmethod
    def _upper(cls, v):
        if isinstance(v, str):
            v = v.strip().upper()
            return v if v in REASON_CODES else "OTHER"
        return v


# ---------------------------------------------------------------------------
# Explainer LLM output (the part the LLM is allowed to author)
# ---------------------------------------------------------------------------
class ExplainerOutput(BaseModel):
    """What the explainer LLM returns. PD is injected separately by the model."""
    reason_codes: List[ReasonCodeItem]
    explanation: str
    recommended_action: "RecommendedAction"


class RecommendedAction(BaseModel):
    action: str
    expected_pd_after: float = Field(ge=0.0, le=1.0)
    rationale: str


# ---------------------------------------------------------------------------
# Judge outputs
# ---------------------------------------------------------------------------
class FaithfulnessVerdict(BaseModel):
    faithful: bool
    unsupported_claims: List[str] = Field(default_factory=list)
    sign_flips: List[str] = Field(default_factory=list)
    notes: str = ""


class ConsensusVerdict(BaseModel):
    chosen: Literal["a", "b", "merged"]
    explanation: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Final per-loan & portfolio results (API contract, Section 10)
# ---------------------------------------------------------------------------
class RiskCurve(BaseModel):
    months: List[int]
    pd: List[float]
    estimated: bool


class Alert(BaseModel):
    flagged: bool
    onset_month: Optional[int] = None
    lead_time_months: Optional[int] = None


class FairnessDetail(BaseModel):
    attribute: str
    dp_diff: float


class LoanFairness(BaseModel):
    flag: Literal["pass", "review"]
    details: List[FairnessDetail] = Field(default_factory=list)


class ExplanationQuality(BaseModel):
    faithful: bool
    json_status: Literal["ok", "repaired", "degraded"]
    model_used: str
    judge: str
    consensus: bool = False


class LoanResult(BaseModel):
    loan_id: str
    pd: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["high", "medium", "low"]
    risk_curve: RiskCurve
    alert: Alert
    reason_codes: List[ReasonCodeItem]
    explanation: str
    recommended_action: RecommendedAction
    fairness: LoanFairness
    explanation_quality: ExplanationQuality


class ModelInfo(BaseModel):
    type: str
    auc: float
    pr_auc: float
    ece: float
    n_loans: int
    brier: float = 0.0


class RiskDistribution(BaseModel):
    high: int
    medium: int
    low: int


class TopRiskLoan(BaseModel):
    loan_id: str
    pd: float


class FairnessSummaryItem(BaseModel):
    attribute: str
    flag: Literal["pass", "review"]
    eo_diff: float
    dp_diff: float = 0.0


class PortfolioResult(BaseModel):
    job_id: str
    model: ModelInfo
    risk_distribution: RiskDistribution
    top_risk_loans: List[TopRiskLoan]
    fairness_summary: List[FairnessSummaryItem]
    mapping_used: Dict[str, Optional[str]]
    warnings: List[str] = Field(default_factory=list)


ExplainerOutput.model_rebuild()
