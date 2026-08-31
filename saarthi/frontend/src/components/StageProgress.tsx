import { PIPELINE_STAGES } from '../lib/theme'

export default function StageProgress({
  currentStage,
  percent,
  done,
  error,
}: {
  currentStage: string
  percent: number
  done: boolean
  error?: string
}) {
  const activeIdx = PIPELINE_STAGES.findIndex((s) => s.key === currentStage)
  const clampedPct = Math.max(0, Math.min(100, Math.round(percent)))

  return (
    <div>
      {/* Progress bar */}
      <div className="mb-6">
        <div className="mb-1.5 flex items-center justify-between text-xs font-medium text-ink-mute">
          <span>{done ? 'Complete' : error ? 'Halted' : 'Working…'}</span>
          <span className="tabular-nums">{clampedPct}%</span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${
              error
                ? 'bg-danger'
                : done
                  ? 'bg-safe'
                  : 'bg-gradient-to-r from-risk to-risk-dark'
            }`}
            style={{ width: `${clampedPct}%` }}
          />
        </div>
      </div>

      {/* Stage rail */}
      <ol className="grid grid-cols-2 gap-2 sm:grid-cols-5">
        {PIPELINE_STAGES.map((stage, i) => {
          const isDone = done || (activeIdx >= 0 && i < activeIdx)
          const isActive = !done && i === activeIdx
          const isErrored = error && isActive
          return (
            <li
              key={stage.key}
              tabIndex={0}
              className={`group relative flex cursor-help items-center gap-2 rounded-xl border px-3 py-2.5 outline-none transition-colors ${
                isErrored
                  ? 'border-danger/40 bg-danger/5'
                  : isActive
                    ? 'border-risk/50 bg-risk/5 shadow-glow'
                    : isDone
                      ? 'border-safe/30 bg-safe/5'
                      : 'border-slate-200 bg-white'
              }`}
            >
              {/* hover / focus overlay explaining this stage */}
              <div className="pointer-events-none absolute bottom-full left-1/2 z-40 mb-2 w-60 max-w-[78vw] -translate-x-1/2 translate-y-1 rounded-xl border border-slate-200 bg-white p-3 text-left opacity-0 shadow-xl transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:translate-y-0 group-focus-within:opacity-100">
                <div className="flex items-center gap-2">
                  <span className="text-base">{stage.icon}</span>
                  <span className="text-xs font-bold text-ink">
                    {i + 1}. {stage.label}
                  </span>
                  {isActive && (
                    <span className="ml-auto text-[10px] font-semibold text-risk-dark">
                      running…
                    </span>
                  )}
                  {isDone && (
                    <span className="ml-auto text-[10px] font-semibold text-safe">
                      done ✓
                    </span>
                  )}
                </div>
                <p className="mt-1.5 text-xs leading-snug text-ink-mute">
                  {stage.desc}
                </p>
                <span className="absolute left-1/2 top-full h-2.5 w-2.5 -translate-x-1/2 -translate-y-[6px] rotate-45 border-b border-r border-slate-200 bg-white" />
              </div>

              <span
                className={`flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full text-sm ${
                  isErrored
                    ? 'bg-danger text-white'
                    : isActive
                      ? 'bg-risk text-ink animate-pulse-soft'
                      : isDone
                        ? 'bg-safe text-white'
                        : 'bg-slate-100 text-ink-mute'
                }`}
              >
                {isDone ? '✓' : isErrored ? '✕' : stage.icon}
              </span>
              <span
                className={`truncate text-xs font-semibold ${
                  isActive ? 'text-ink' : isDone ? 'text-safe' : 'text-ink-mute'
                }`}
              >
                {stage.label}
              </span>
            </li>
          )
        })}
      </ol>
      <p className="mt-3 text-center text-[11px] text-ink-mute/70">
        💡 Hover any step to see what it does
      </p>
    </div>
  )
}
