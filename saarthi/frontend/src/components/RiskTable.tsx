import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { TopRiskLoan, RiskBand } from '../types'
import { bandFromPd, bandStyle, pct } from '../lib/theme'

type SortKey = 'loan_id' | 'pd'
type SortDir = 'asc' | 'desc'

const PAGE_SIZE = 10

export default function RiskTable({
  loans,
  jobId,
}: {
  loans: TopRiskLoan[]
  jobId: string
}) {
  const navigate = useNavigate()
  const [sortKey, setSortKey] = useState<SortKey>('pd')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [page, setPage] = useState(0)

  const sorted = useMemo(() => {
    const arr = [...(loans ?? [])]
    arr.sort((a, b) => {
      let cmp = 0
      if (sortKey === 'pd') cmp = (a.pd ?? 0) - (b.pd ?? 0)
      else cmp = String(a.loan_id).localeCompare(String(b.loan_id))
      return sortDir === 'asc' ? cmp : -cmp
    })
    return arr
  }, [loans, sortKey, sortDir])

  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE))
  const safePage = Math.min(page, pageCount - 1)
  const slice = sorted.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE)

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'pd' ? 'desc' : 'asc')
    }
    setPage(0)
  }

  function sortArrow(key: SortKey) {
    if (key !== sortKey) return '↕'
    return sortDir === 'asc' ? '↑' : '↓'
  }

  if (!loans || loans.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-ink-mute">
        No loans to display.
      </p>
    )
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-ink-mute">
              <th className="py-2.5 pr-4">
                <button
                  className="font-semibold hover:text-ink"
                  onClick={() => toggleSort('loan_id')}
                >
                  Loan ID {sortArrow('loan_id')}
                </button>
              </th>
              <th className="py-2.5 pr-4">
                <button
                  className="font-semibold hover:text-ink"
                  onClick={() => toggleSort('pd')}
                >
                  PD {sortArrow('pd')}
                </button>
              </th>
              <th className="py-2.5 pr-4">Risk band</th>
              <th className="py-2.5 pr-2 text-right">Detail</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((loan) => {
              const band: RiskBand = loan.risk_band ?? bandFromPd(loan.pd ?? 0)
              const style = bandStyle(band)
              return (
                <tr
                  key={loan.loan_id}
                  onClick={() =>
                    navigate(
                      `/loan/${encodeURIComponent(jobId)}/${encodeURIComponent(
                        loan.loan_id,
                      )}`,
                    )
                  }
                  className="cursor-pointer border-b border-slate-100 transition-colors hover:bg-slate-50"
                >
                  <td className="py-3 pr-4 font-mono font-medium text-ink">
                    {loan.loan_id}
                  </td>
                  <td className="py-3 pr-4">
                    <div className="flex items-center gap-2">
                      <span className="w-12 tabular-nums font-semibold text-ink">
                        {pct(loan.pd, 1)}
                      </span>
                      <span className="hidden h-1.5 w-24 overflow-hidden rounded-full bg-slate-100 sm:block">
                        <span
                          className={`block h-full rounded-full ${style.dot}`}
                          style={{
                            width: `${Math.min(100, Math.round((loan.pd ?? 0) * 100))}%`,
                          }}
                        />
                      </span>
                    </div>
                  </td>
                  <td className="py-3 pr-4">
                    <span className={`chip ${style.bg} ${style.text}`}>
                      <span className={`h-2 w-2 rounded-full ${style.dot}`} />
                      {style.label}
                    </span>
                  </td>
                  <td className="py-3 pr-2 text-right text-risk-dark">
                    View →
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pageCount > 1 && (
        <div className="mt-4 flex items-center justify-between text-xs text-ink-mute">
          <span>
            Showing {safePage * PAGE_SIZE + 1}–
            {Math.min(sorted.length, safePage * PAGE_SIZE + PAGE_SIZE)} of{' '}
            {sorted.length}
          </span>
          <div className="flex items-center gap-1">
            <button
              className="btn-ghost px-3 py-1.5"
              disabled={safePage === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
            >
              ← Prev
            </button>
            <span className="px-2 tabular-nums">
              {safePage + 1} / {pageCount}
            </span>
            <button
              className="btn-ghost px-3 py-1.5"
              disabled={safePage >= pageCount - 1}
              onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
