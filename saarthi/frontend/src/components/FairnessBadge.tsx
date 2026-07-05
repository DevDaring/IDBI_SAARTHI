import type { FairnessFlag } from '../types'
import { pct } from '../lib/theme'

export interface FairnessBadgeProps {
  attribute: string
  flag: FairnessFlag
  /** disparity metric value (eo_diff or dp_diff) */
  metric?: number
  metricLabel?: string
}

export default function FairnessBadge({
  attribute,
  flag,
  metric,
  metricLabel = 'Δ',
}: FairnessBadgeProps) {
  const pass = flag === 'pass'
  return (
    <div
      className={`flex items-center justify-between gap-3 rounded-xl border px-3 py-2 ${
        pass
          ? 'border-safe/30 bg-safe/5'
          : 'border-risk/40 bg-risk/5'
      }`}
    >
      <div className="min-w-0">
        <div className="truncate text-sm font-semibold text-ink">
          {attribute}
        </div>
        {metric !== undefined && (
          <div className="text-xs text-ink-mute">
            {metricLabel} {pct(Math.abs(metric), 1)}
          </div>
        )}
      </div>
      <span
        className={`chip flex-shrink-0 ${
          pass ? 'bg-safe/15 text-safe' : 'bg-risk/15 text-risk-dark'
        }`}
      >
        {pass ? '✓ Pass' : '⚑ Review'}
      </span>
    </div>
  )
}
