import { useCallback, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadFile, errorMessage } from '../api/client'
import type { UploadResult } from '../types'
import { Section, Spinner } from '../components/ui'

function Hero() {
  return (
    <div className="mb-8 text-center">
      <span className="chip mx-auto mb-4 bg-ink/5 text-ink-mute">
        <span className="devanagari text-risk-dark">सारथी</span> · your credit
        charioteer
      </span>
      <h1 className="mx-auto max-w-3xl text-3xl font-extrabold tracking-tight text-ink sm:text-4xl">
        Predict who defaults,{' '}
        <span className="text-risk-dark">see it 12 months early</span>, and show
        the one move that fixes it.
      </h1>
      <p className="mx-auto mt-3 max-w-2xl text-sm text-ink-mute sm:text-base">
        Drop your MSME loan book below. SAARTHI maps your columns, trains an
        early-warning model, and produces faithful, judge-verified explanations
        for every loan — recourse included.
      </p>
    </div>
  )
}

export default function Upload() {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<UploadResult | null>(null)

  const handleFile = useCallback(async (file: File) => {
    setError(null)
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a .csv file.')
      return
    }
    setUploading(true)
    setResult(null)
    try {
      const res = await uploadFile(file)
      setResult(res)
      try {
        sessionStorage.setItem(`saarthi.upload.${res.upload_id}`, JSON.stringify(res))
      } catch {
        /* ignore storage quota */
      }
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setUploading(false)
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files?.[0]
      if (file) void handleFile(file)
    },
    [handleFile],
  )

  return (
    <div className="animate-fade-in">
      <Hero />

      {/* Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
        }}
        className={`mx-auto flex max-w-3xl cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-10 text-center transition-all ${
          dragging
            ? 'border-risk bg-risk/5 shadow-glow'
            : 'border-slate-300 bg-white hover:border-risk/60 hover:bg-slate-50'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) void handleFile(f)
            e.target.value = ''
          }}
        />
        <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-ink text-2xl text-white">
          {uploading ? (
            <span className="text-risk">
              <Spinner size={26} />
            </span>
          ) : (
            '📂'
          )}
        </div>
        <p className="text-base font-bold text-ink">
          {uploading
            ? 'Uploading & profiling…'
            : 'Drag & drop your loan CSV here'}
        </p>
        <p className="mt-1 text-sm text-ink-mute">
          or <span className="font-semibold text-risk-dark">click to select</span>{' '}
          a file
        </p>
      </div>

      {error && (
        <div className="mx-auto mt-4 max-w-3xl rounded-xl border border-danger/30 bg-danger/5 px-4 py-3 text-sm text-danger">
          ⚠️ {error}
        </div>
      )}

      {/* Profile result */}
      {result && (
        <div className="mx-auto mt-8 max-w-5xl animate-fade-in">
          <Section
            title="Column profile"
            subtitle={`${result.filename} · ${result.n_rows.toLocaleString()} rows · ${result.columns.length} columns`}
            icon="🧾"
            right={
              <button
                className="btn-accent"
                onClick={() =>
                  navigate(`/mapping/${encodeURIComponent(result.upload_id)}`)
                }
              >
                Continue to mapping →
              </button>
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-ink-mute">
                    <th className="py-2.5 pr-4">Column</th>
                    <th className="py-2.5 pr-4">Type</th>
                    <th className="py-2.5 pr-4">Null %</th>
                    <th className="py-2.5 pr-4">Sample values</th>
                  </tr>
                </thead>
                <tbody>
                  {result.columns.map((col) => (
                    <tr
                      key={col.name}
                      className="border-b border-slate-100 align-top"
                    >
                      <td className="py-3 pr-4 font-mono font-medium text-ink">
                        {col.name}
                      </td>
                      <td className="py-3 pr-4">
                        <span className="chip bg-slate-100 text-ink-mute">
                          {col.dtype}
                        </span>
                      </td>
                      <td className="py-3 pr-4 tabular-nums">
                        <span
                          className={
                            col.null_pct > 30
                              ? 'font-semibold text-danger'
                              : 'text-ink-mute'
                          }
                        >
                          {col.null_pct.toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-ink-mute">
                        <div className="flex flex-wrap gap-1.5">
                          {(col.sample ?? []).slice(0, 5).map((s, i) => (
                            <span
                              key={i}
                              className="rounded-md bg-slate-50 px-2 py-0.5 font-mono text-xs"
                            >
                              {s === null || s === '' ? '∅' : String(s)}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}
    </div>
  )
}
