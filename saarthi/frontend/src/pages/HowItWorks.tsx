import { Link } from 'react-router-dom'

/* ── small building blocks ──────────────────────────────────────────────── */
function Fig({ src, alt, caption }: { src: string; alt: string; caption?: string }) {
  return (
    <figure className="my-5 overflow-hidden rounded-xl border border-slate-200 bg-white">
      <div className="overflow-x-auto p-4">
        <img src={src} alt={alt} className="mx-auto min-w-[280px] max-w-full" />
      </div>
      {caption && (
        <figcaption className="border-t border-slate-100 bg-slate-50 px-4 py-2 text-center text-xs text-ink-mute">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}

function Step({
  n,
  title,
  color,
  children,
}: {
  n: number | string
  title: string
  color: string
  children: React.ReactNode
}) {
  return (
    <section className="card p-5 sm:p-6">
      <div className="flex items-center gap-3">
        <span
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-sm font-bold text-white"
          style={{ backgroundColor: color }}
        >
          {n}
        </span>
        <h2 className="text-lg font-bold text-ink">{title}</h2>
      </div>
      <div className="mt-3 space-y-3 text-[15px] leading-relaxed text-ink-mute">
        {children}
      </div>
    </section>
  )
}

function Callout({
  tone = 'amber',
  title,
  children,
}: {
  tone?: 'amber' | 'teal' | 'red'
  title: string
  children: React.ReactNode
}) {
  const map = {
    amber: 'border-risk/40 bg-risk/5',
    teal: 'border-safe/40 bg-safe/5',
    red: 'border-danger/40 bg-danger/5',
  } as const
  return (
    <div className={`rounded-xl border px-4 py-3 text-sm ${map[tone]}`}>
      <p className="font-semibold text-ink">{title}</p>
      <p className="mt-1 text-ink-mute">{children}</p>
    </div>
  )
}

const CODES: [string, string][] = [
  ['LIQUIDITY_STRESS', 'tight cash flow'],
  ['REPAYMENT_HISTORY_POOR', 'past late payments'],
  ['LEVERAGE_HIGH', 'too much debt'],
  ['COLLATERAL_LOW', 'weak security'],
  ['SECTOR_RISK', 'risky industry'],
  ['REVENUE_DECLINE', 'weak turnover'],
  ['TENURE_RISK', 'loan-term risk'],
  ['TEXT_DISTRESS_SIGNAL', 'worrying purpose text'],
]

/* ── page ───────────────────────────────────────────────────────────────── */
export default function HowItWorks() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {/* hero */}
      <header className="rounded-2xl bg-ink px-6 py-8 text-white sm:px-8">
        <p className="text-xs font-semibold uppercase tracking-widest text-risk">
          How SAARTHI works
        </p>
        <h1 className="mt-2 text-2xl font-extrabold sm:text-3xl">
          A credit officer's early-warning radar — explained simply
        </h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-slate-300">
          You upload a spreadsheet of loans. SAARTHI predicts{' '}
          <span className="text-white">who is likely to default</span>, shows it{' '}
          <span className="text-white">up to 12 months early</span>, explains{' '}
          <span className="text-white">why in plain English</span> (double-checked
          by a second AI so it can't make things up), and tells you the{' '}
          <span className="text-white">one move</span> that lowers the risk — all
          while staying fair. No jargon, every answer backed by evidence.
        </p>
      </header>

      {/* big picture */}
      <section className="card p-5 sm:p-6">
        <h2 className="text-lg font-bold text-ink">The whole journey in one picture</h2>
        <p className="mt-2 text-[15px] leading-relaxed text-ink-mute">
          Ten steps run automatically. Blue = your data, teal = the maths model,
          amber = the AI that explains, red = the AI that verifies, and it ends on
          a dashboard a loan officer can use on day one.
        </p>
        <Fig
          src="/how/pipeline.svg"
          alt="SAARTHI pipeline from upload to dashboard"
          caption="Upload → map → train → predict → explain → verify → recommend → fairness → dashboard"
        />
      </section>

      <Step n={1} title="It reads your messy spreadsheet" color="#334155">
        <p>
          Every bank names its columns differently —{' '}
          <code className="rounded bg-slate-100 px-1">DisbursedAmount</code>,{' '}
          <code className="rounded bg-slate-100 px-1">MIS_Status</code>,{' '}
          <code className="rounded bg-slate-100 px-1">NAICS</code>. An AI reads
          your column names <em>and</em> a few sample values and maps them onto one
          fixed set of fields SAARTHI understands. You can correct any mapping
          before running.
        </p>
        <Fig src="/how/mapping.svg" alt="Mapping messy columns to a fixed schema" />
        <Callout tone="red" title="Fairness is built-in from the start">
          Sensitive fields like <b>gender</b> and <b>region</b> are marked
          “protected”. They are used <b>only</b> to audit for bias — never as
          inputs the model predicts from.
        </Callout>
      </Step>

      <Step n={2} title="It learns who defaults" color="#0D9488">
        <p>
          SAARTHI studies thousands of past loans and learns the patterns that
          separated the ones that were repaid from the ones that went bad
          (a gradient-boosting model, LightGBM). Think of it as a very experienced
          loan officer who has read a million files.
        </p>
        <p>
          Crucially, the score is <b>calibrated</b>: when it says{' '}
          <b>“70% chance of default”</b>, roughly 70 out of 100 such loans really
          do default. So the number means what it says.
        </p>
        <Callout tone="teal" title="Example">
          A loan with a low credit score (530), tight cash flow (DSCR 0.54) and 3
          past late payments comes out as <b>high risk (PD ≈ 0.9)</b>. A healthy
          loan with strong cash flow comes out <b>low (PD ≈ 0.05)</b>.
        </Callout>
      </Step>

      <Step n={3} title="It explains WHY — not just a number" color="#F59E0B">
        <p>
          A number alone is useless to an officer. SAARTHI uses <b>SHAP</b> to
          measure how much <em>each</em> factor pushed this loan's risk up or down.
          Red bars push risk up, green pull it down — and the sizes add up to the
          final score.
        </p>
        <Fig src="/how/shap.svg" alt="SHAP contribution of each factor for one loan" />
        <p>
          Every loan is then described in the <b>same fixed vocabulary</b>, so a
          manager can compare any two loans at a glance:
        </p>
        <div className="flex flex-wrap gap-2">
          {CODES.map(([code, plain]) => (
            <span
              key={code}
              className="chip bg-slate-100 text-ink"
              title={plain}
            >
              {code.replaceAll('_', ' ').toLowerCase()}
            </span>
          ))}
        </div>
      </Step>

      <Step n={4} title="It writes plain English — and a second AI checks it" color="#DC2626">
        <p>
          One AI turns the drivers into 2–3 clear sentences. Then a{' '}
          <b>different</b> AI model — the <b>faithfulness judge</b> — compares
          those sentences against the model's real evidence. If the explanation
          invents a reason or flips a direction, it is rewritten. Only then does it
          earn a <span className="font-semibold text-safe-dark">✓ Verified</span> badge.
        </p>
        <Fig src="/how/judge.svg" alt="Faithfulness judge loop" />
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-sm italic text-ink">
            “The borrower's credit score of 530 and 3 prior delinquencies signal
            weak repayment history. A DSCR of 0.54 shows liquidity stress, and
            collateral is low relative to the exposure.”
          </p>
          <p className="mt-2 text-xs font-semibold text-safe-dark">
            ✓ Verified by faithfulness judge · written by deepseek, checked by gemini
          </p>
        </div>
        <Callout tone="amber" title="Why this matters">
          This is the trust story: the AI can <b>describe</b> the model, but it is
          not allowed to <b>hallucinate</b>. What you read is always backed by the
          model's own evidence.
        </Callout>
      </Step>

      <Step n={5} title="The one move that fixes it" color="#0F766E">
        <p>
          For risky loans, SAARTHI searches for the <b>smallest realistic change</b>
          — extend the tenure, add collateral, add a working-capital line — that
          would push the risk back below the safe line, and shows the projected
          before → after.
        </p>
        <Fig src="/how/recourse.svg" alt="Recourse: PD before and after the recommended action" />
      </Step>

      <Step n={6} title="It sees trouble up to 12 months early" color="#D97706">
        <p>
          Instead of one static score, SAARTHI plots how the risk builds{' '}
          <b>month by month</b> and flags the month it crosses the alert line — so
          you can act while there is still time.
        </p>
        <Fig src="/how/curve.svg" alt="12-month cumulative default-risk curve with onset marker" />
      </Step>

      <Step n={7} title="It stays fair" color="#334155">
        <p>
          Using the protected fields (audit-only), SAARTHI checks whether any group
          is treated worse <em>at the same risk level</em>. It reports the gap and
          flags it for review only when the difference isn't explained by
          legitimate risk factors — so real bias is caught without penalising sound
          lending.
        </p>
      </Step>

      {/* golden rule */}
      <section className="card border-2 border-risk/30 p-5 sm:p-6">
        <h2 className="text-lg font-bold text-ink">The golden rule that makes it trustworthy</h2>
        <p className="mt-2 text-[15px] leading-relaxed text-ink-mute">
          Three different players, three different jobs — and the AI never touches
          the number.
        </p>
        <Fig src="/how/roles.svg" alt="Model gives the number, AI writes the words, judge verifies" />
      </section>

      {/* CTA */}
      <div className="rounded-2xl bg-gradient-to-r from-ink to-ink-soft px-6 py-7 text-center text-white">
        <p className="text-lg font-bold">Ready to see it on real loans?</p>
        <p className="mx-auto mt-1 max-w-md text-sm text-slate-300">
          Drop in a CSV and watch the whole pipeline run — with a faithful,
          verified explanation for every loan.
        </p>
        <Link
          to="/"
          className="mt-4 inline-flex items-center gap-2 rounded-xl bg-risk px-5 py-2.5 font-semibold text-ink transition hover:bg-risk-dark hover:text-white"
        >
          Upload a dataset →
        </Link>
      </div>
    </div>
  )
}
