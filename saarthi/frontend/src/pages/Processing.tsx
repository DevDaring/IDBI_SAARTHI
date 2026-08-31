import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getStatus, errorMessage } from '../api/client'
import type { StatusResult } from '../types'
import { ErrorCard } from '../components/ui'
import StageProgress from '../components/StageProgress'

const POLL_MS = 1200

export default function Processing() {
  const { jobId = '' } = useParams()
  const navigate = useNavigate()
  const [status, setStatus] = useState<StatusResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | null>(null)
  const stopped = useRef(false)

  useEffect(() => {
    stopped.current = false

    async function poll() {
      try {
        const s = await getStatus(jobId)
        if (stopped.current) return
        setStatus(s)
        if (s.error) {
          setError(s.error)
          return
        }
        if (s.done) {
          // small beat so the user sees "complete"
          timer.current = window.setTimeout(() => {
            if (!stopped.current)
              navigate(`/dashboard/${encodeURIComponent(jobId)}`)
          }, 600)
          return
        }
        timer.current = window.setTimeout(poll, POLL_MS)
      } catch (e) {
        if (stopped.current) return
        setError(errorMessage(e))
      }
    }

    void poll()
    return () => {
      stopped.current = true
      if (timer.current) window.clearTimeout(timer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  if (error) {
    return (
      <ErrorCard
        title="Analysis failed"
        message={error}
        onRetry={() => navigate('/')}
      />
    )
  }

  const percent = status?.percent ?? 0
  const stage = status?.stage ?? 'ingest'
  const done = status?.done ?? false
  const message = status?.message ?? 'Connecting to the pipeline…'

  return (
    <div className="mx-auto max-w-3xl animate-fade-in py-6">
      <div className="mb-6 text-center">
        <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-ink text-2xl">
          <span className="animate-pulse-soft">⚙️</span>
        </div>
        <h1 className="text-2xl font-extrabold tracking-tight text-ink">
          Building your early-warning report
        </h1>
        <p className="mt-1 text-sm text-ink-mute">{message}</p>
      </div>

      <div className="card p-6">
        <StageProgress
          currentStage={stage}
          percent={percent}
          done={done}
          error={status?.error}
        />
      </div>

      <p className="mt-4 text-center text-xs text-ink-mute">
        Job <span className="font-mono">{jobId}</span> · polling every{' '}
        {POLL_MS / 1000}s
      </p>
    </div>
  )
}
