import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { mapColumns, runAnalysis, errorMessage } from '../api/client'
import type { MapResult, UploadColumn, UploadResult } from '../types'
import { getConsensus } from '../lib/settings'
import { Loading, ErrorCard, Section } from '../components/ui'
import MappingEditor, { type MappingState } from '../components/MappingEditor'

function loadCachedUpload(uploadId: string): UploadResult | null {
  try {
    const raw = sessionStorage.getItem(`saarthi.upload.${uploadId}`)
    return raw ? (JSON.parse(raw) as UploadResult) : null
  } catch {
    return null
  }
}

/** Build a column list, falling back to mapped column names if no cache. */
function deriveColumns(
  cached: UploadResult | null,
  map: MapResult,
): UploadColumn[] {
  if (cached?.columns?.length) return cached.columns
  const names = new Set<string>()
  Object.values(map.mapping ?? {}).forEach((v) => v && names.add(v))
  map.protected?.forEach((p) => names.add(p))
  if (map.target) names.add(map.target)
  return [...names].map((name) => ({
    name,
    dtype: 'unknown',
    sample: [],
    null_pct: 0,
  }))
}

export default function Mapping() {
  const { uploadId = '' } = useParams()
  const navigate = useNavigate()

  const [map, setMap] = useState<MapResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [state, setState] = useState<MappingState>({
    mapping: {},
    target: null,
    protectedSet: new Set(),
  })

  const cached = useMemo(() => loadCachedUpload(uploadId), [uploadId])

  function load() {
    setLoading(true)
    setError(null)
    mapColumns(uploadId)
      .then((res) => {
        setMap(res)
        setState({
          mapping: { ...(res.mapping ?? {}) },
          target: res.target ?? null,
          protectedSet: new Set(res.protected ?? []),
        })
      })
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadId])

  if (loading) return <Loading label="AI is proposing a column mapping…" />
  if (error)
    return (
      <ErrorCard
        message={error}
        onRetry={load}
        title="Could not map columns"
      />
    )
  if (!map) return null

  const columns = deriveColumns(cached, map)
  const canonicalFields = Object.keys(map.mapping ?? {})

  async function handleRun() {
    if (!state.target) {
      setError('Please choose a target field before running the analysis.')
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const { job_id } = await runAnalysis({
        upload_id: uploadId,
        mapping: state.mapping,
        target: state.target,
        protected: [...state.protectedSet],
        consensus: getConsensus(),
      })
      navigate(`/processing/${encodeURIComponent(job_id)}`)
    } catch (e) {
      setError(errorMessage(e))
      setSubmitting(false)
    }
  }

  return (
    <div className="animate-fade-in space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-ink">
          Confirm the column mapping
        </h1>
        <p className="mt-1 text-sm text-ink-mute">
          SAARTHI proposed how your columns map to its canonical fields. Adjust
          anything, then pick the{' '}
          <span className="font-semibold text-ink">target</span> (what to
          predict) and any{' '}
          <span className="font-semibold text-risk-dark">protected</span>{' '}
          attributes for the fairness audit.
        </p>
      </div>

      <Section title="Canonical fields" icon="🗺️">
        <MappingEditor
          canonicalFields={canonicalFields}
          columns={columns}
          confidence={map.confidence ?? {}}
          notes={map.notes ?? {}}
          state={state}
          onChange={setState}
        />
      </Section>

      {error && (
        <div className="rounded-xl border border-danger/30 bg-danger/5 px-4 py-3 text-sm text-danger">
          ⚠️ {error}
        </div>
      )}

      <div className="flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
        <div className="text-sm text-ink-mute">
          Target:{' '}
          <span className="font-semibold text-ink">
            {state.target ?? 'not set'}
          </span>{' '}
          · Protected:{' '}
          <span className="font-semibold text-ink">
            {state.protectedSet.size > 0
              ? [...state.protectedSet].join(', ')
              : 'none'}
          </span>
        </div>
        <button
          className="btn-accent"
          onClick={handleRun}
          disabled={submitting || !state.target}
        >
          {submitting ? 'Starting…' : 'Run analysis →'}
        </button>
      </div>
    </div>
  )
}
