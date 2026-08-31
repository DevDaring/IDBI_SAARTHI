import type { ExplanationQuality, JsonStatus } from '../types'

const STATUS_STYLE: Record<JsonStatus, { label: string; cls: string }> = {
  ok: { label: 'JSON ok', cls: 'bg-safe/15 text-safe' },
  repaired: { label: 'JSON repaired', cls: 'bg-risk/15 text-risk-dark' },
  degraded: { label: 'JSON degraded', cls: 'bg-danger/15 text-danger' },
}

export default function ModelTrace({ quality }: { quality?: ExplanationQuality }) {
  if (!quality) return null
  const status = STATUS_STYLE[quality.json_status] ?? STATUS_STYLE.degraded
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-ink-mute">
      <span className="font-semibold uppercase tracking-wide text-ink-mute/80">
        Model trace
      </span>
      <span>
        Writer&nbsp;
        <span className="font-mono font-medium text-ink">
          {quality.model_used || '—'}
        </span>
      </span>
      <span>
        Judge&nbsp;
        <span className="font-mono font-medium text-ink">
          {quality.judge || '—'}
        </span>
      </span>
      <span className={`chip ${status.cls}`}>{status.label}</span>
    </div>
  )
}
