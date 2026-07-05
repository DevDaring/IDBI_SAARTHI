import type { UploadColumn } from '../types'

export interface MappingState {
  mapping: Record<string, string | null>
  target: string | null
  protectedSet: Set<string>
}

function ConfidenceChip({ value }: { value: number | undefined }) {
  if (value === undefined) return null
  const pctVal = Math.round(value * 100)
  let cls = 'bg-slate-100 text-ink-mute'
  if (value >= 0.75) cls = 'bg-safe/15 text-safe'
  else if (value >= 0.45) cls = 'bg-risk/15 text-risk-dark'
  else cls = 'bg-danger/15 text-danger'
  return <span className={`chip ${cls}`}>{pctVal}% conf</span>
}

const NONE = '__none__'

export default function MappingEditor({
  canonicalFields,
  columns,
  confidence,
  notes,
  state,
  onChange,
}: {
  canonicalFields: string[]
  columns: UploadColumn[]
  confidence: Record<string, number>
  notes: Record<string, string>
  state: MappingState
  onChange: (next: MappingState) => void
}) {
  const columnNames = columns.map((c) => c.name)

  function setMapping(field: string, value: string) {
    onChange({
      ...state,
      mapping: { ...state.mapping, [field]: value === NONE ? null : value },
    })
  }

  function setTarget(field: string) {
    const col = state.mapping[field] ?? null
    onChange({ ...state, target: col })
  }

  function toggleProtected(field: string) {
    const col = state.mapping[field]
    if (!col) return
    const next = new Set(state.protectedSet)
    if (next.has(col)) next.delete(col)
    else next.add(col)
    onChange({ ...state, protectedSet: next })
  }

  return (
    <div className="space-y-3">
      {canonicalFields.map((field) => {
        const mappedCol = state.mapping[field] ?? null
        const isTarget = mappedCol !== null && state.target === mappedCol
        const isProtected = mappedCol !== null && state.protectedSet.has(mappedCol)
        const note = notes[field]
        return (
          <div
            key={field}
            className="rounded-xl border border-slate-200 p-4 transition-colors hover:border-slate-300"
          >
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0 sm:w-48">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-bold text-ink">
                    {field}
                  </span>
                  {isTarget && (
                    <span className="chip bg-ink text-white">target</span>
                  )}
                  {isProtected && (
                    <span className="chip bg-risk/15 text-risk-dark">
                      protected
                    </span>
                  )}
                </div>
                <ConfidenceChip value={confidence[field]} />
              </div>

              <div className="flex flex-1 flex-col gap-2 sm:flex-row sm:items-center">
                <select
                  value={mappedCol ?? NONE}
                  onChange={(e) => setMapping(field, e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-ink focus:border-risk focus:outline-none focus:ring-2 focus:ring-risk/30 sm:flex-1"
                >
                  <option value={NONE}>(none)</option>
                  {columnNames.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>

                <div className="flex flex-shrink-0 gap-1.5">
                  <button
                    type="button"
                    disabled={!mappedCol}
                    onClick={() => setTarget(field)}
                    className={`rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition-colors disabled:opacity-40 ${
                      isTarget
                        ? 'border-ink bg-ink text-white'
                        : 'border-slate-300 text-ink-mute hover:bg-slate-100'
                    }`}
                    title="Mark as prediction target"
                  >
                    Target
                  </button>
                  <button
                    type="button"
                    disabled={!mappedCol}
                    onClick={() => toggleProtected(field)}
                    className={`rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition-colors disabled:opacity-40 ${
                      isProtected
                        ? 'border-risk bg-risk/15 text-risk-dark'
                        : 'border-slate-300 text-ink-mute hover:bg-slate-100'
                    }`}
                    title="Mark as protected attribute"
                  >
                    Protected
                  </button>
                </div>
              </div>
            </div>

            {note && (
              <p className="mt-2 text-xs text-ink-mute">
                <span className="font-semibold">Note:</span> {note}
              </p>
            )}
          </div>
        )
      })}
    </div>
  )
}
