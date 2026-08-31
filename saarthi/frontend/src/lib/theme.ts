import type { RiskBand, ReasonCodeKey } from '../types'

// ── Risk band styling ────────────────────────────────────────────────────────
export interface BandStyle {
  label: string
  text: string
  bg: string
  border: string
  dot: string
  hex: string
}

export const BAND_STYLES: Record<RiskBand, BandStyle> = {
  high: {
    label: 'High risk',
    text: 'text-danger',
    bg: 'bg-danger/10',
    border: 'border-danger/30',
    dot: 'bg-danger',
    hex: '#DC2626',
  },
  medium: {
    label: 'Medium risk',
    text: 'text-risk-dark',
    bg: 'bg-risk/10',
    border: 'border-risk/40',
    dot: 'bg-risk',
    hex: '#F59E0B',
  },
  low: {
    label: 'Low risk',
    text: 'text-safe',
    bg: 'bg-safe/10',
    border: 'border-safe/30',
    dot: 'bg-safe',
    hex: '#0D9488',
  },
}

export function bandStyle(band: RiskBand | undefined): BandStyle {
  return BAND_STYLES[band ?? 'low'] ?? BAND_STYLES.low
}

/** Derive a band from a PD when none is provided. */
export function bandFromPd(pd: number): RiskBand {
  if (pd >= 0.5) return 'high'
  if (pd >= 0.2) return 'medium'
  return 'low'
}

// ── Reason-code taxonomy ─────────────────────────────────────────────────────
export interface ReasonMeta {
  label: string
  icon: string
  /** tailwind text color for accenting the chip label */
  accent: string
  hint: string
}

export const REASON_TAXONOMY: Record<ReasonCodeKey, ReasonMeta> = {
  LIQUIDITY_STRESS: {
    label: 'Liquidity stress',
    icon: '💧',
    accent: 'text-sky-600',
    hint: 'Cash buffers are thinning relative to obligations.',
  },
  LEVERAGE_HIGH: {
    label: 'High leverage',
    icon: '⚖️',
    accent: 'text-rose-600',
    hint: 'Debt load is elevated versus equity / income.',
  },
  REVENUE_DECLINE: {
    label: 'Revenue decline',
    icon: '📉',
    accent: 'text-orange-600',
    hint: 'Top-line is shrinking over recent periods.',
  },
  REPAYMENT_HISTORY_POOR: {
    label: 'Poor repayment history',
    icon: '🕗',
    accent: 'text-red-600',
    hint: 'Past delinquencies or missed instalments.',
  },
  SECTOR_RISK: {
    label: 'Sector risk',
    icon: '🏭',
    accent: 'text-amber-600',
    hint: 'Borrower operates in a stressed sector.',
  },
  COLLATERAL_LOW: {
    label: 'Low collateral',
    icon: '🛡️',
    accent: 'text-fuchsia-600',
    hint: 'Security cover is thin relative to exposure.',
  },
  BEHAVIOUR_ANOMALY: {
    label: 'Behaviour anomaly',
    icon: '📟',
    accent: 'text-purple-600',
    hint: 'Unusual transaction or account patterns detected.',
  },
  TEXT_DISTRESS_SIGNAL: {
    label: 'Text distress signal',
    icon: '📝',
    accent: 'text-pink-600',
    hint: 'Distress language found in notes / filings.',
  },
  TENURE_RISK: {
    label: 'Tenure risk',
    icon: '📅',
    accent: 'text-indigo-600',
    hint: 'Loan vintage / remaining tenure raises risk.',
  },
  OTHER: {
    label: 'Other factor',
    icon: '•',
    accent: 'text-slate-600',
    hint: 'Additional contributing factor.',
  },
}

export function reasonMeta(code: string): ReasonMeta {
  const key = code as ReasonCodeKey
  return REASON_TAXONOMY[key] ?? REASON_TAXONOMY.OTHER
}

// ── Pipeline stages (for processing screen) ──────────────────────────────────
export interface StageMeta {
  key: string
  label: string
  icon: string
  /** plain-language "what happens here", shown as a hover overlay */
  desc: string
}

export const PIPELINE_STAGES: StageMeta[] = [
  {
    key: 'ingest',
    label: 'Ingest',
    icon: '📥',
    desc: 'Reads your CSV, counts the rows, and profiles every column — its type, how many values are missing, and a few sample values.',
  },
  {
    key: 'map',
    label: 'Map',
    icon: '🗺️',
    desc: 'An AI matches your column names to SAARTHI’s fixed schema (e.g. DisbursedAmount → loan_amount) and flags sensitive fields like gender as protected (audit-only).',
  },
  {
    key: 'features',
    label: 'Features',
    icon: '🧬',
    desc: 'Turns raw columns into model-ready numbers: encodes categories, fills gaps, and removes IDs and protected attributes so they can never influence the score.',
  },
  {
    key: 'train',
    label: 'Train',
    icon: '🤖',
    desc: 'A gradient-boosting model (LightGBM) learns the patterns that separated repaid loans from defaults, then is calibrated so “70%” really means a 70% chance.',
  },
  {
    key: 'survival',
    label: 'Survival',
    icon: '⏳',
    desc: 'Spreads each loan’s risk across the next 12 months and finds the month it crosses the alert line — so trouble is visible early.',
  },
  {
    key: 'explain',
    label: 'Explain',
    icon: '💡',
    desc: 'SHAP measures how much each factor pushed the risk up or down, then an AI writes 2–3 plain-English sentences using only those real drivers.',
  },
  {
    key: 'judge',
    label: 'Judge',
    icon: '⚖️',
    desc: 'A second, different AI checks the explanation against the model’s evidence. If it invents a reason or flips a direction, it’s rewritten — then earns a ✓ Verified badge.',
  },
  {
    key: 'recourse',
    label: 'Recourse',
    icon: '🎯',
    desc: 'Searches for the smallest realistic change (extend tenure, add collateral, add working capital) that would push the loan back below the safe line, and shows before → after.',
  },
  {
    key: 'fairness',
    label: 'Fairness',
    icon: '🤝',
    desc: 'Checks whether any protected group is treated worse at the same risk level, and flags it for review only when the gap isn’t explained by legitimate risk factors.',
  },
  {
    key: 'assemble',
    label: 'Assemble',
    icon: '📦',
    desc: 'Packages everything — scores, risk curves, reasons, actions and fairness flags — into the portfolio dashboard and per-loan views.',
  },
]

// ── Formatting helpers ───────────────────────────────────────────────────────
export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}
