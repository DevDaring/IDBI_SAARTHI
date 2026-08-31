import { useEffect, useState } from 'react'
import { getModels, errorMessage } from '../api/client'
import type { ModelsResult } from '../types'
import { getConsensus, setConsensus } from '../lib/settings'
import { Spinner } from './ui'

export default function SettingsDrawer({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const [models, setModels] = useState<ModelsResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [consensus, setConsensusState] = useState<boolean>(getConsensus())

  useEffect(() => {
    if (!open || models) return
    let alive = true
    setLoading(true)
    setError(null)
    getModels()
      .then((m) => {
        if (alive) setModels(m)
      })
      .catch((e) => {
        if (alive) setError(errorMessage(e))
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
  }, [open, models])

  function toggleConsensus() {
    const next = !consensus
    setConsensusState(next)
    setConsensus(next)
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 z-40 bg-ink/40 backdrop-blur-sm transition-opacity duration-300 ${
          open ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
        onClick={onClose}
        aria-hidden
      />
      {/* Panel */}
      <aside
        className={`fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col bg-white shadow-2xl transition-transform duration-300 ${
          open ? 'translate-x-0' : 'translate-x-full'
        }`}
        role="dialog"
        aria-label="Settings"
      >
        <header className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">⚙️</span>
            <h2 className="text-base font-bold text-ink">Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-ink-mute hover:bg-slate-100"
            aria-label="Close settings"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto px-6 py-5">
          {/* Consensus toggle */}
          <div className="rounded-xl border border-slate-200 p-4">
            <label className="flex cursor-pointer items-start justify-between gap-4">
              <span>
                <span className="block text-sm font-semibold text-ink">
                  Consensus judge for high-risk loans
                </span>
                <span className="mt-1 block text-xs text-ink-mute">
                  Use multiple judge models to cross-check explanations on the
                  riskiest loans. Slower, but more robust.
                </span>
              </span>
              <button
                type="button"
                role="switch"
                aria-checked={consensus}
                onClick={toggleConsensus}
                className={`relative mt-0.5 h-6 w-11 flex-shrink-0 rounded-full transition-colors ${
                  consensus ? 'bg-safe' : 'bg-slate-300'
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${
                    consensus ? 'translate-x-5' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </label>
          </div>

          {/* Resolved models */}
          <div>
            <h3 className="mb-2 text-xs font-bold uppercase tracking-wide text-ink-mute">
              Resolved models
            </h3>

            {loading && (
              <div className="flex items-center gap-2 py-6 text-sm text-ink-mute">
                <span className="text-risk">
                  <Spinner size={18} />
                </span>
                Loading model routes…
              </div>
            )}

            {error && (
              <div className="rounded-lg bg-danger/10 px-3 py-2 text-xs text-danger">
                {error}
              </div>
            )}

            {models && (
              <div className="space-y-4">
                <div className="space-y-2">
                  {Object.entries(models.routes ?? {}).map(([role, list]) => (
                    <div
                      key={role}
                      className="rounded-lg border border-slate-200 px-3 py-2"
                    >
                      <div className="text-xs font-bold capitalize text-ink">
                        {role}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-1.5">
                        {(list ?? []).map((m, i) => (
                          <span
                            key={`${m.provider}-${m.model}-${i}`}
                            className="chip bg-slate-100 text-ink-mute"
                          >
                            {m.provider}:{m.model}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div>
                  <h3 className="mb-2 text-xs font-bold uppercase tracking-wide text-ink-mute">
                    Providers
                  </h3>
                  <div className="space-y-1.5">
                    {(models.providers ?? []).map((p) => (
                      <div
                        key={p.name}
                        className="flex items-center justify-between rounded-lg border border-slate-200 px-3 py-2 text-sm"
                      >
                        <span className="flex items-center gap-2 font-medium text-ink">
                          <span
                            className={`h-2 w-2 rounded-full ${
                              p.ok ? 'bg-safe' : 'bg-danger'
                            }`}
                          />
                          {p.name}
                        </span>
                        <span className="text-xs text-ink-mute">
                          {(p.models ?? []).length} models
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="border-t border-slate-200 px-6 py-3 text-center text-xs text-ink-mute">
          SAARTHI · settings are stored locally in your browser
        </footer>
      </aside>
    </>
  )
}
