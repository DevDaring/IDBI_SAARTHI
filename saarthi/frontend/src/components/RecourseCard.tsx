import type { RecommendedAction } from '../types'
import { pct } from '../lib/theme'

function PdPill({
  value,
  tone,
  caption,
}: {
  value: number
  tone: 'before' | 'after'
  caption: string
}) {
  const before = tone === 'before'
  return (
    <div className="flex flex-col items-center">
      <div
        className={`flex h-24 w-24 flex-col items-center justify-center gap-0.5 rounded-2xl border-2 px-2 sm:h-28 sm:w-28 ${
          before
            ? 'border-danger/40 bg-danger/5 text-danger'
            : 'border-safe/40 bg-safe/5 text-safe'
        }`}
      >
        <span className="whitespace-nowrap text-xl font-extrabold leading-none tabular-nums sm:text-2xl">
          {pct(value, 1)}
        </span>
        <span className="text-[10px] font-semibold uppercase tracking-wide opacity-70">
          PD
        </span>
      </div>
      <span className="mt-2 text-xs font-medium text-ink-mute">{caption}</span>
    </div>
  )
}

export default function RecourseCard({
  before,
  action,
}: {
  before: number
  action?: RecommendedAction
}) {
  if (!action) {
    return (
      <p className="py-6 text-center text-sm text-ink-mute">
        No recourse recommendation available.
      </p>
    )
  }

  const after = action.expected_pd_after
  const delta = before - after
  const improved = delta > 0

  return (
    <div className="animate-fade-in">
      <div className="rounded-2xl bg-gradient-to-br from-ink to-ink-soft p-5 text-white sm:p-6">
        <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-risk">
          🎯 The one move that fixes it
        </div>

        <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
          <PdPill value={before} tone="before" caption="Today" />

          {/* Action arrow */}
          <div className="flex flex-1 flex-col items-center px-2">
            <div className="rounded-xl bg-white/10 px-4 py-3 text-center backdrop-blur">
              <div className="text-sm font-bold text-white">
                {action.action}
              </div>
            </div>
            <div className="mt-3 flex w-full items-center text-white/40">
              <span className="h-px flex-1 bg-white/30" />
              <svg
                viewBox="0 0 24 24"
                className="mx-1 h-5 w-5 text-risk"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M5 12h14M13 6l6 6-6 6" />
              </svg>
            </div>
            {improved && (
              <span className="mt-2 chip bg-safe/20 text-safe">
                ↓ {pct(delta, 1)} reduction
              </span>
            )}
          </div>

          <PdPill value={after} tone="after" caption="After action" />
        </div>
      </div>

      {action.rationale && (
        <p className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3.5 text-sm text-ink-soft">
          <span className="font-semibold text-ink-mute">Why this works: </span>
          {action.rationale}
        </p>
      )}
    </div>
  )
}
