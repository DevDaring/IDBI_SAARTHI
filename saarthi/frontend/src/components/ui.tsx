import type { ReactNode } from 'react'

// ── Spinner ──────────────────────────────────────────────────────────────────
export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <span
      className="inline-block animate-spin rounded-full border-2 border-current border-t-transparent"
      style={{ width: size, height: size }}
      role="status"
      aria-label="Loading"
    />
  )
}

// ── Full-area loading state ───────────────────────────────────────────────────
export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-20 text-ink-mute">
      <span className="text-risk">
        <Spinner size={32} />
      </span>
      <p className="text-sm font-medium">{label}</p>
    </div>
  )
}

// ── Friendly error card ───────────────────────────────────────────────────────
export function ErrorCard({
  message,
  onRetry,
  title = 'Something went wrong',
}: {
  message: string
  onRetry?: () => void
  title?: string
}) {
  return (
    <div className="card animate-fade-in mx-auto max-w-lg p-8 text-center">
      <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-danger/10 text-2xl">
        ⚠️
      </div>
      <h3 className="text-lg font-bold text-ink">{title}</h3>
      <p className="mt-2 text-sm text-ink-mute">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="btn-accent mt-5">
          Try again
        </button>
      )}
    </div>
  )
}

// ── Generic card section ──────────────────────────────────────────────────────
export function Section({
  title,
  subtitle,
  icon,
  right,
  children,
  className = '',
}: {
  title?: string
  subtitle?: string
  icon?: ReactNode
  right?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`card p-5 sm:p-6 ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div className="flex items-start gap-2.5">
            {icon && <span className="text-xl leading-none">{icon}</span>}
            <div>
              {title && (
                <h2 className="text-base font-bold text-ink">{title}</h2>
              )}
              {subtitle && (
                <p className="mt-0.5 text-xs text-ink-mute">{subtitle}</p>
              )}
            </div>
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

// ── Skeleton block ────────────────────────────────────────────────────────────
export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`skeleton rounded-lg ${className}`} />
}
