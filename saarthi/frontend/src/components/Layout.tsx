import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import SettingsDrawer from './SettingsDrawer'

function Wheel() {
  return (
    <span className="relative inline-flex h-9 w-9 items-center justify-center rounded-xl bg-ink">
      <svg viewBox="0 0 64 64" className="h-6 w-6">
        <g stroke="#F59E0B" strokeWidth="3" strokeLinecap="round" fill="none">
          <circle cx="32" cy="32" r="18" />
          <circle cx="32" cy="32" r="5" fill="#F59E0B" stroke="none" />
          <line x1="32" y1="14" x2="32" y2="50" />
          <line x1="14" y1="32" x2="50" y2="32" />
          <line x1="19.3" y1="19.3" x2="44.7" y2="44.7" />
          <line x1="44.7" y1="19.3" x2="19.3" y2="44.7" />
        </g>
      </svg>
    </span>
  )
}

export default function Layout({ children }: { children: React.ReactNode }) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const location = useLocation()
  const onHome = location.pathname === '/'

  return (
    <div className="flex min-h-full flex-col bg-slate-50">
      <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          <Link to="/" className="flex items-center gap-3">
            <Wheel />
            <span className="leading-tight">
              <span className="flex items-center gap-2">
                <span className="text-lg font-extrabold tracking-tight text-ink">
                  SAARTHI
                </span>
                <span className="devanagari text-base text-risk-dark">
                  सारथी
                </span>
              </span>
              <span className="block text-[11px] font-medium text-ink-mute">
                Credit War-Room · MSME default early-warning
              </span>
            </span>
          </Link>

          <div className="flex items-center gap-2">
            <Link
              to="/how-it-works"
              className="btn-ghost hidden px-3 sm:inline-flex"
              title="How SAARTHI works"
            >
              <span className="text-base">💡</span>
              <span className="hidden sm:inline">How it works</span>
            </Link>
            {!onHome && (
              <Link to="/" className="btn-ghost hidden sm:inline-flex">
                ＋ New analysis
              </Link>
            )}
            <button
              onClick={() => setSettingsOpen(true)}
              className="btn-ghost px-3"
              aria-label="Open settings"
              title="Settings"
            >
              <span className="text-base">⚙️</span>
              <span className="hidden sm:inline">Settings</span>
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
        {children}
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-4 text-center text-xs text-ink-mute sm:px-6">
          SAARTHI · IDBI Innovate 2026 ·{' '}
          <span className="text-ink-mute/70">
            Predict who defaults, see it 12 months early, and show the one move
            that fixes it.
          </span>
        </div>
      </footer>

      <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  )
}
