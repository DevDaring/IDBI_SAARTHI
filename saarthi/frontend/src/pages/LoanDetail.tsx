import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getLoan, errorMessage } from '../api/client'
import type { LoanResult } from '../types'
import { bandStyle, pct } from '../lib/theme'
import { Loading, ErrorCard, Section } from '../components/ui'
import RiskCurve from '../components/RiskCurve'
import ReasonCodes from '../components/ReasonCodes'
import RecourseCard from '../components/RecourseCard'
import FairnessBadge from '../components/FairnessBadge'
import ModelTrace from '../components/ModelTrace'

function FaithBadge({ faithful }: { faithful: boolean }) {
  return faithful ? (
    <span className="chip bg-safe/15 text-safe">
      ✓ Verified by faithfulness judge
    </span>
  ) : (
    <span className="chip bg-risk/15 text-risk-dark">⚠ Unverified</span>
  )
}

export default function LoanDetail() {
  const { jobId = '', loanId = '' } = useParams()
  const [loan, setLoan] = useState<LoanResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  function load() {
    setLoading(true)
    setError(null)
    getLoan(jobId, loanId)
      .then(setLoan)
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, loanId])

  const backLink = `/dashboard/${encodeURIComponent(jobId)}`

  if (loading) return <Loading label="Loading loan reasoning…" />
  if (error)
    return (
      <div>
        <Link to={backLink} className="btn-ghost mb-4">
          ← Back to dashboard
        </Link>
        <ErrorCard title="Could not load loan" message={error} onRetry={load} />
      </div>
    )
  if (!loan) return null

  const style = bandStyle(loan.risk_band)
  const quality = loan.explanation_quality

  return (
    <div className="animate-fade-in space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-3">
        <Link to={backLink} className="text-sm font-semibold text-risk-dark">
          ← Back to dashboard
        </Link>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div
              className={`flex h-20 min-w-[5.5rem] flex-col items-center justify-center gap-0.5 rounded-2xl border-2 px-3 ${style.border} ${style.bg}`}
            >
              <span
                className={`whitespace-nowrap text-xl font-extrabold leading-none tabular-nums ${style.text}`}
              >
                {pct(loan.pd, 1)}
              </span>
              <span className="text-[10px] font-bold uppercase text-ink-mute">
                PD
              </span>
            </div>
            <div>
              <h1 className="font-mono text-xl font-extrabold text-ink">
                {loan.loan_id}
              </h1>
              <span className={`chip mt-1 ${style.bg} ${style.text}`}>
                <span className={`h-2 w-2 rounded-full ${style.dot}`} />
                {style.label}
              </span>
            </div>
          </div>

          {loan.alert?.flagged && (
            <div className="rounded-xl border border-danger/30 bg-danger/5 px-4 py-2 text-sm">
              <div className="font-semibold text-danger">⚑ Early-warning alert</div>
              <div className="text-xs text-ink-mute">
                Onset at month {loan.alert.onset_month ?? '—'} ·{' '}
                {loan.alert.lead_time_months ?? '—'} months lead-time
              </div>
            </div>
          )}
        </div>
      </div>

      {/* HERO: Explanation */}
      <Section
        icon="💡"
        title="What SAARTHI sees"
        right={<FaithBadge faithful={!!quality?.faithful} />}
        className="border-risk/30 bg-gradient-to-br from-white to-risk/5"
      >
        <p className="text-base leading-relaxed text-ink-soft sm:text-lg">
          {loan.explanation || 'No explanation was produced for this loan.'}
        </p>
      </Section>

      {/* Recourse — the memorable visual */}
      <Section icon="🎯" title="Recommended recourse">
        <RecourseCard before={loan.pd} action={loan.recommended_action} />
      </Section>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Risk curve */}
        <Section
          icon="📉"
          title="12-month risk curve"
          subtitle="Cumulative probability of default over the horizon"
        >
          <RiskCurve curve={loan.risk_curve} alert={loan.alert} />
        </Section>

        {/* Reason codes */}
        <Section
          icon="🧩"
          title="Reason codes"
          subtitle="Ranked drivers behind this score"
        >
          <ReasonCodes codes={loan.reason_codes} />
        </Section>
      </div>

      {/* Fairness */}
      <Section
        icon="🤝"
        title="Fairness check"
        right={
          <span
            className={`chip ${
              loan.fairness?.flag === 'pass'
                ? 'bg-safe/15 text-safe'
                : 'bg-risk/15 text-risk-dark'
            }`}
          >
            {loan.fairness?.flag === 'pass' ? '✓ Pass' : '⚑ Review'}
          </span>
        }
      >
        {loan.fairness?.details && loan.fairness.details.length > 0 ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {loan.fairness.details.map((d) => (
              <FairnessBadge
                key={d.attribute}
                attribute={d.attribute}
                flag={loan.fairness.flag}
                metric={d.dp_diff}
                metricLabel="DP diff"
              />
            ))}
          </div>
        ) : (
          <p className="py-4 text-center text-sm text-ink-mute">
            No protected attributes were evaluated for this loan.
          </p>
        )}
      </Section>

      {/* Model trace footer */}
      <ModelTrace quality={quality} />
    </div>
  )
}
