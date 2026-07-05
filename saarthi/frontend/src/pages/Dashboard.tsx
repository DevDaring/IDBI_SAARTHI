import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from 'recharts'
import { getResults, errorMessage } from '../api/client'
import type { PortfolioResult, RiskBand } from '../types'
import { BAND_STYLES, pct, num } from '../lib/theme'
import { Loading, ErrorCard, Section } from '../components/ui'
import MetricCard from '../components/MetricCard'
import FairnessBadge from '../components/FairnessBadge'
import RiskTable from '../components/RiskTable'

function RiskDonut({
  dist,
}: {
  dist: PortfolioResult['risk_distribution']
}) {
  const data = (['high', 'medium', 'low'] as RiskBand[]).map((band) => ({
    name: BAND_STYLES[band].label,
    band,
    value: dist?.[band] ?? 0,
  }))
  const total = data.reduce((s, d) => s + d.value, 0)

  if (total === 0) {
    return (
      <p className="py-10 text-center text-sm text-ink-mute">
        No risk distribution available.
      </p>
    )
  }

  return (
    <div className="relative h-64">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            innerRadius={62}
            outerRadius={92}
            paddingAngle={2}
            stroke="none"
          >
            {data.map((d) => (
              <Cell key={d.band} fill={BAND_STYLES[d.band].hex} />
            ))}
          </Pie>
          <Tooltip
            formatter={(v) => [`${Number(v)} loans`, '']}
            contentStyle={{
              borderRadius: 12,
              border: '1px solid #e2e8f0',
              fontSize: 12,
            }}
          />
          <Legend
            verticalAlign="bottom"
            iconType="circle"
            wrapperStyle={{ fontSize: 12 }}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center pb-8">
        <span className="text-2xl font-extrabold text-ink">{total}</span>
        <span className="text-[11px] uppercase tracking-wide text-ink-mute">
          loans
        </span>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { jobId = '' } = useParams()
  const [data, setData] = useState<PortfolioResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  function load() {
    setLoading(true)
    setError(null)
    getResults(jobId)
      .then(setData)
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  if (loading) return <Loading label="Loading portfolio results…" />
  if (error)
    return (
      <ErrorCard
        title="Could not load results"
        message={error}
        onRetry={load}
      />
    )
  if (!data) return null

  const m = data.model
  const warnings = data.warnings ?? []

  return (
    <div className="animate-fade-in space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-ink">
            Portfolio war-room
          </h1>
          <p className="mt-1 text-sm text-ink-mute">
            {(m?.n_loans ?? 0).toLocaleString()} loans scored ·{' '}
            <span className="font-mono">{m?.type ?? 'model'}</span>
          </p>
        </div>
      </div>

      {/* Notes about this run */}
      {warnings.length > 0 && (
        <div className="rounded-2xl border border-risk/30 bg-risk/5 p-4">
          <p className="flex items-center gap-2 text-sm font-semibold text-ink">
            <span aria-hidden>ℹ️</span> A few notes about this run
          </p>
          <ul className="mt-2 space-y-1.5">
            {warnings.map((w, i) => (
              <li key={i} className="flex gap-2 text-sm leading-snug text-ink-mute">
                <span className="mt-[2px] text-risk-dark">•</span>
                <span>{w}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Metric cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <MetricCard
          label="AUC"
          value={num(m?.auc, 3)}
          icon="📈"
          accent="text-safe"
        />
        <MetricCard
          label="PR-AUC"
          value={num(m?.pr_auc, 3)}
          icon="🎯"
          accent="text-safe"
        />
        <MetricCard
          label="Calibration (ECE)"
          value={num(m?.ece, 3)}
          sub="lower is better"
          icon="🎚️"
          accent="text-risk-dark"
        />
        <MetricCard
          label="Loans"
          value={(m?.n_loans ?? 0).toLocaleString()}
          icon="🏦"
        />
        <MetricCard
          label="Model"
          value={m?.type ?? '—'}
          icon="🤖"
          accent="text-ink"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Distribution */}
        <Section
          title="Risk distribution"
          subtitle="High / medium / low bands"
          icon="🍩"
          className="lg:col-span-1"
        >
          <RiskDonut dist={data.risk_distribution} />
        </Section>

        {/* Fairness */}
        <Section
          title="Fairness audit"
          subtitle="Equalised-odds per protected attribute"
          icon="🤝"
          className="lg:col-span-2"
        >
          {data.fairness_summary && data.fairness_summary.length > 0 ? (
            <div className="grid gap-3 sm:grid-cols-2">
              {data.fairness_summary.map((f) => (
                <FairnessBadge
                  key={f.attribute}
                  attribute={f.attribute}
                  flag={f.flag}
                  metric={f.eo_diff}
                  metricLabel="EO diff"
                />
              ))}
            </div>
          ) : (
            <p className="py-6 text-center text-sm text-ink-mute">
              No protected attributes were audited.
            </p>
          )}
        </Section>
      </div>

      {/* Risk table */}
      <Section
        title="Loans by probability of default"
        subtitle={`${(data.top_risk_loans ?? []).length} loans · click a row for full reasoning & recourse`}
        icon="📋"
      >
        <RiskTable loans={data.top_risk_loans ?? []} jobId={jobId} />
      </Section>

      {/* Mapping used (collapsible-ish footer) */}
      {data.mapping_used && Object.keys(data.mapping_used).length > 0 && (
        <details className="card p-4 text-sm">
          <summary className="cursor-pointer font-semibold text-ink-mute">
            Mapping used for this run
          </summary>
          <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(data.mapping_used).map(([field, col]) => (
              <div
                key={field}
                className="flex items-center justify-between rounded-lg border border-slate-200 px-3 py-1.5"
              >
                <span className="font-mono text-xs font-semibold text-ink">
                  {field}
                </span>
                <span className="text-xs text-ink-mute">{col ?? '(none)'}</span>
              </div>
            ))}
          </div>
        </details>
      )}

      <p className="text-center text-xs text-ink-mute">
        Portfolio ECE {pct(m?.ece, 1)} · scored at job{' '}
        <span className="font-mono">{jobId}</span>
      </p>
    </div>
  )
}
