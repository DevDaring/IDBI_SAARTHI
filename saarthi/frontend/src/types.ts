// ────────────────────────────────────────────────────────────────────────────
// SAARTHI API contract types
// ────────────────────────────────────────────────────────────────────────────

export type RiskBand = 'high' | 'medium' | 'low'
export type FairnessFlag = 'pass' | 'review'
export type ReasonDirection = 'increases_risk' | 'decreases_risk'
export type JsonStatus = 'ok' | 'repaired' | 'degraded'

// Reason-code fixed taxonomy
export type ReasonCodeKey =
  | 'LIQUIDITY_STRESS'
  | 'LEVERAGE_HIGH'
  | 'REVENUE_DECLINE'
  | 'REPAYMENT_HISTORY_POOR'
  | 'SECTOR_RISK'
  | 'COLLATERAL_LOW'
  | 'BEHAVIOUR_ANOMALY'
  | 'TEXT_DISTRESS_SIGNAL'
  | 'TENURE_RISK'
  | 'OTHER'

// ── Upload ──────────────────────────────────────────────────────────────────
export interface UploadColumn {
  name: string
  dtype: string
  sample: (string | number | null)[]
  null_pct: number
}

export interface UploadResult {
  upload_id: string
  columns: UploadColumn[]
  n_rows: number
  filename: string
}

// ── Mapping ─────────────────────────────────────────────────────────────────
export interface MapResult {
  mapping: Record<string, string | null>
  target: string | null
  protected: string[]
  confidence: Record<string, number>
  notes: Record<string, string>
}

// Request body for /api/run
export interface RunRequest {
  upload_id: string
  mapping: Record<string, string | null>
  target: string | null
  protected: string[]
  consensus?: boolean
}

export interface RunResult {
  job_id: string
}

// ── Status ──────────────────────────────────────────────────────────────────
export interface StatusResult {
  stage: string
  percent: number
  message: string
  done: boolean
  error?: string
}

// ── Portfolio (dashboard) ───────────────────────────────────────────────────
export interface PortfolioModel {
  type: string
  auc: number
  pr_auc: number
  ece: number
  n_loans: number
}

export interface RiskDistribution {
  high: number
  medium: number
  low: number
}

export interface TopRiskLoan {
  loan_id: string
  pd: number
  risk_band?: RiskBand
}

export interface FairnessSummaryItem {
  attribute: string
  flag: FairnessFlag
  eo_diff: number
}

export interface PortfolioResult {
  job_id: string
  model: PortfolioModel
  risk_distribution: RiskDistribution
  top_risk_loans: TopRiskLoan[]
  fairness_summary: FairnessSummaryItem[]
  mapping_used: Record<string, string | null>
  warnings: string[]
}

// ── Loan detail ─────────────────────────────────────────────────────────────
export interface RiskCurveData {
  months: number[]
  pd: number[]
  estimated: boolean
}

export interface AlertData {
  flagged: boolean
  onset_month: number | null
  lead_time_months: number | null
}

export interface ReasonCode {
  code: string
  weight: number
  direction: ReasonDirection
  evidence: string
  feature: string
  shap: number
}

export interface RecommendedAction {
  action: string
  expected_pd_after: number
  rationale: string
}

export interface FairnessDetail {
  attribute: string
  dp_diff: number
}

export interface FairnessData {
  flag: FairnessFlag
  details: FairnessDetail[]
}

export interface ExplanationQuality {
  faithful: boolean
  json_status: JsonStatus
  model_used: string
  judge: string
}

export interface LoanResult {
  loan_id: string
  pd: number
  risk_band: RiskBand
  risk_curve: RiskCurveData
  alert: AlertData
  reason_codes: ReasonCode[]
  explanation: string
  recommended_action: RecommendedAction
  fairness: FairnessData
  explanation_quality: ExplanationQuality
}

// ── Models / settings ───────────────────────────────────────────────────────
export interface ProviderModel {
  provider: string
  model: string
}

export interface ProviderInfo {
  name: string
  ok: boolean
  models: string[]
}

export interface ModelsResult {
  routes: Record<string, ProviderModel[]>
  providers: ProviderInfo[]
}

export interface HealthResult {
  status: 'ok'
}
