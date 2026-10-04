import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import { Run, scoreOf } from './types'

interface Point {
  round: number
  best: number | null
  kept?: number
  discarded?: number
  status: string
  hypothesis: string | null
}

// How the dev score moved over the run: every round's score, and the best prompt so far as a stepped line.
export default function ScoreChart({ run }: { run: Run }) {
  const baseline = scoreOf(run.baseline)
  if (baseline == null) return null

  let best = baseline
  const points: Point[] = [{ round: 0, best: baseline * 100, kept: baseline * 100, status: 'Starting prompt', hypothesis: null }]
  for (const it of run.iterations) {
    const score = scoreOf(it)
    if (it.status === 'keep' && score != null) best = score
    points.push({
      round: it.iteration,
      best: best * 100,
      ...(score == null ? {} : it.status === 'keep' ? { kept: score * 100 } : { discarded: score * 100 }),
      status: it.status === 'keep' ? 'Kept' : it.status === 'discard' ? 'Discarded' : 'Failed (no score)',
      hypothesis: it.hypothesis,
    })
  }

  const values = points.flatMap((p) => [p.best, p.kept, p.discarded]).filter((v): v is number => v != null)
  const low = Math.max(0, Math.floor((Math.min(...values) - 2) / 5) * 5)
  const high = Math.min(100, Math.ceil((Math.max(...values) + 2) / 5) * 5)

  return (
    <div>
      <div className="legend" style={{ marginBottom: 6 }}>
        <span>
          <svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" stroke="var(--accent)" strokeWidth="2" /></svg>
          Best so far
        </span>
        <span>
          <svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>
          Kept round
        </span>
        <span>
          <svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill="var(--surface)" stroke="var(--text-muted)" strokeWidth="1.5" /></svg>
          Discarded round
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis
            dataKey="round"
            type="number"
            domain={[0, Math.max(run.loops, 1)]}
            allowDecimals={false}
            tickCount={Math.min(run.loops, 10) + 1}
            tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
            stroke="var(--axis)"
            tickLine={false}
          />
          <YAxis
            domain={[low, high]}
            tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
            tickFormatter={(v) => `${v}%`}
            stroke="var(--axis)"
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ stroke: 'var(--axis)' }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const p = payload[0].payload as Point
              const score = p.kept ?? p.discarded
              return (
                <div className="chart-tip">
                  <strong>{p.round === 0 ? 'Starting prompt' : `Round ${p.round} · ${p.status}`}</strong>
                  <div className="num">
                    {score != null && `Score ${score.toFixed(1)}% · `}best so far {p.best?.toFixed(1)}%
                  </div>
                  {p.hypothesis && <div className="secondary" style={{ marginTop: 4 }}>{p.hypothesis.slice(0, 160)}{p.hypothesis.length > 160 ? '…' : ''}</div>}
                </div>
              )
            }}
          />
          <Line dataKey="best" type="stepAfter" stroke="var(--accent)" strokeWidth={2} dot={false} activeDot={false} isAnimationActive={false} />
          <Scatter
            dataKey="discarded"
            isAnimationActive={false}
            shape={(props: { cx?: number; cy?: number }) => <circle cx={props.cx} cy={props.cy} r={4} fill="var(--surface)" stroke="var(--text-muted)" strokeWidth={1.5} />}
          />
          <Scatter
            dataKey="kept"
            isAnimationActive={false}
            shape={(props: { cx?: number; cy?: number }) => <circle cx={props.cx} cy={props.cy} r={5} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="small muted">Dev score by round (round 0 is the starting prompt).</p>
    </div>
  )
}
