import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts'
import type { RiskCurveData, AlertData } from '../types'
import { pct } from '../lib/theme'

interface Point {
  month: number
  pd: number
}

// matches the backend onset threshold (config.SETTINGS.onset_threshold = 0.20):
// the first month the cumulative-PD curve crosses this line is the alert onset.
const THRESHOLD = 0.2

export default function RiskCurve({
  curve,
  alert,
}: {
  curve?: RiskCurveData
  alert?: AlertData
}) {
  const months = curve?.months ?? []
  const pds = curve?.pd ?? []
  const data: Point[] = months.map((m, i) => ({
    month: m,
    pd: typeof pds[i] === 'number' ? pds[i] : 0,
  }))

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-ink-mute">
        No survival curve available.
      </div>
    )
  }

  const onset = alert?.onset_month ?? null

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {curve?.estimated && (
          <span className="chip bg-risk/15 text-risk-dark">⚠ Estimated</span>
        )}
        {alert?.flagged && onset !== null && (
          <span className="chip bg-danger/10 text-danger">
            Onset · month {onset}
          </span>
        )}
        {alert?.lead_time_months !== null &&
          alert?.lead_time_months !== undefined && (
            <span className="chip bg-safe/10 text-safe">
              {alert.lead_time_months} mo lead-time
            </span>
          )}
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 8, right: 12, left: 0, bottom: 4 }}
          >
            <defs>
              <linearGradient id="pdLine" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#0D9488" />
                <stop offset="60%" stopColor="#F59E0B" />
                <stop offset="100%" stopColor="#DC2626" />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="month"
              tick={{ fontSize: 11, fill: '#64748b' }}
              label={{
                value: 'Month',
                position: 'insideBottom',
                offset: -2,
                fontSize: 11,
                fill: '#94a3b8',
              }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              tick={{ fontSize: 11, fill: '#64748b' }}
              width={42}
            />
            <Tooltip
              formatter={(v) => [pct(Number(v), 1), 'Cumulative PD']}
              labelFormatter={(l) => `Month ${l}`}
              contentStyle={{
                borderRadius: 12,
                border: '1px solid #e2e8f0',
                fontSize: 12,
              }}
            />
            <ReferenceLine
              y={THRESHOLD}
              stroke="#DC2626"
              strokeDasharray="4 4"
              label={{
                value: `Threshold ${pct(THRESHOLD, 0)}`,
                position: 'right',
                fontSize: 10,
                fill: '#DC2626',
              }}
            />
            {onset !== null && (
              <ReferenceLine
                x={onset}
                stroke="#D97706"
                strokeWidth={2}
                label={{
                  value: 'Onset',
                  position: 'top',
                  fontSize: 10,
                  fill: '#D97706',
                }}
              />
            )}
            <Line
              type="monotone"
              dataKey="pd"
              stroke="url(#pdLine)"
              strokeWidth={3}
              dot={{ r: 3, fill: '#F59E0B' }}
              activeDot={{ r: 5 }}
              isAnimationActive
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
