import type { ReasonCode } from '../types'
import { reasonMeta, num } from '../lib/theme'

function DirectionTag({ direction }: { direction: ReasonCode['direction'] }) {
  const increases = direction === 'increases_risk'
  return (
    <span
      className={`chip ${
        increases ? 'bg-danger/10 text-danger' : 'bg-safe/10 text-safe'
      }`}
    >
      {increases ? '▲ Increases risk' : '▼ Decreases risk'}
    </span>
  )
}

export default function ReasonCodes({ codes }: { codes?: ReasonCode[] }) {
  const list = (codes ?? [])
    .slice()
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))

  if (list.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-ink-mute">
        No reason codes were produced for this loan.
      </p>
    )
  }

  const maxWeight = Math.max(...list.map((c) => Math.abs(c.weight)), 1e-6)

  return (
    <ul className="space-y-3">
      {list.map((code, i) => {
        const meta = reasonMeta(code.code)
        const increases = code.direction === 'increases_risk'
        const widthPct = Math.max(
          4,
          Math.round((Math.abs(code.weight) / maxWeight) * 100),
        )
        return (
          <li
            key={`${code.code}-${i}`}
            className="animate-fade-in rounded-xl border border-slate-200 p-3.5"
            style={{ animationDelay: `${i * 40}ms` }}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="text-lg leading-none">{meta.icon}</span>
                <div>
                  <div className="text-sm font-bold text-ink">{meta.label}</div>
                  <div className="font-mono text-[10px] uppercase text-ink-mute">
                    {code.code}
                  </div>
                </div>
              </div>
              <DirectionTag direction={code.direction} />
            </div>

            {/* Weight bar */}
            <div className="mt-3">
              <div className="mb-1 flex items-center justify-between text-[11px] text-ink-mute">
                <span>Contribution</span>
                <span className="tabular-nums">
                  weight {num(code.weight, 3)} · shap {num(code.shap, 3)}
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    increases ? 'bg-danger' : 'bg-safe'
                  }`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
            </div>

            {code.evidence && (
              <p className="mt-2.5 text-sm text-ink-soft">
                <span className="font-semibold text-ink-mute">Evidence: </span>
                {code.evidence}
              </p>
            )}
            {code.feature && (
              <p className="mt-1 text-xs text-ink-mute">
                Feature:{' '}
                <span className="font-mono text-ink-soft">{code.feature}</span>
              </p>
            )}
          </li>
        )
      })}
    </ul>
  )
}
