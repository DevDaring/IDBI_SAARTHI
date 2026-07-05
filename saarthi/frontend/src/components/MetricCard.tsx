export default function MetricCard({
  label,
  value,
  sub,
  icon,
  accent = 'text-ink',
}: {
  label: string
  value: string
  sub?: string
  icon?: string
  accent?: string
}) {
  return (
    <div className="card animate-fade-in flex flex-col justify-between p-4 sm:p-5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink-mute">
          {label}
        </span>
        {icon && <span className="text-lg leading-none">{icon}</span>}
      </div>
      <div className="mt-2">
        <div className={`text-2xl font-extrabold tabular-nums ${accent}`}>
          {value}
        </div>
        {sub && <div className="mt-0.5 text-xs text-ink-mute">{sub}</div>}
      </div>
    </div>
  )
}
