import { useEffect, useState } from 'react'
import { api, apiError, pct, when } from '../api'
import { useApp, useTask } from '../context'
import { Banner, Button, Card, EmptyState, Field, ProgressBar, StatusBadge } from '../ui'

interface RunRow {
  eval_id: string
  engine: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  progress_done: number
  progress_total: number
  error: string | null
  accuracy: number | null
  macro_f1: number | null
  ms_per_item: number | null
}

interface Comparison {
  comparison_id: string
  version: string
  sample_size: number
  timestamp: string
  active: boolean
  runs: RunRow[]
}

const POLL_MS = 2000

// One prompt, the same sample of golden rows, several models side by side.
export default function Compare() {
  const { task, versions, open } = useTask()
  const { engines, engineName, refreshActivity } = useApp()
  const [version, setVersion] = useState('v1_baseline')
  const [sampleSize, setSampleSize] = useState(100)
  const [chosen, setChosen] = useState<string[] | null>(null)
  const [comparisons, setComparisons] = useState<Comparison[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Every model is ticked until the user changes the selection.
  const selectedEngines = chosen ?? engines.map((e) => e.id)

  const load = async (): Promise<Comparison[]> => {
    try {
      const res = await api.get<Comparison[]>('/comparisons', { params: { task: task.id } })
      setComparisons(res.data)
      return res.data
    } catch {
      return []
    }
  }

  useEffect(() => {
    load().then((items) => items[0] && setSelectedId(items[0].comparison_id))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const anyActive = comparisons.some((c) => c.active)
  useEffect(() => {
    if (!anyActive) return
    const timer = window.setInterval(load, POLL_MS)
    return () => window.clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anyActive])

  const start = async () => {
    setError(null)
    try {
      const res = await api.post<{ comparison_id: string }[]>('/compare', {
        task: task.id,
        version,
        engines: selectedEngines,
        sample_size: sampleSize,
      })
      setSelectedId(res.data[0].comparison_id)
      await load()
      refreshActivity()
    } catch (err) {
      setError(apiError(err, 'Failed to start the comparison'))
    }
  }

  const toggle = (id: string) =>
    setChosen(selectedEngines.includes(id) ? selectedEngines.filter((e) => e !== id) : [...selectedEngines, id])

  const selected = comparisons.find((c) => c.comparison_id === selectedId) ?? null
  const done = selected?.runs.filter((r) => r.status === 'completed') ?? []
  const bestAccuracy = done.length > 0 ? Math.max(...done.map((r) => r.accuracy ?? 0)) : null
  const fastest = done.length > 0 ? Math.min(...done.map((r) => r.ms_per_item ?? Infinity)) : null
  const spec = (id: string) => engines.find((e) => e.id === id)

  return (
    <div className="page">
      <Card
        title="Compare models"
        sub={`Runs one prompt on several models, on the same random sample of golden ${task.items}, one model after another. The prompt is the version's instructions and label descriptions, put into each model's own input format.`}
      >
        <div className="stack">
          <div className="row end">
            <Field label="Prompt (model version)">
              <select className="input" value={version} onChange={(e) => setVersion(e.target.value)}>
                {versions.map((v) => (
                  <option key={v.version} value={v.version}>{v.version}</option>
                ))}
              </select>
            </Field>
            <Field label="Sample size">
              <input
                className="input narrow num"
                type="number"
                min={10}
                max={10000}
                value={sampleSize}
                onChange={(e) => setSampleSize(Math.max(10, Math.min(10000, parseInt(e.target.value) || 100)))}
              />
            </Field>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th></th>
                  <th>Model</th>
                  <th>Size</th>
                  <th>Speed</th>
                  <th>About</th>
                </tr>
              </thead>
              <tbody>
                {engines.map((e) => (
                  <tr key={e.id} className="clickable" onClick={() => toggle(e.id)}>
                    <td style={{ width: 1 }}>
                      <input type="checkbox" checked={selectedEngines.includes(e.id)} onChange={() => toggle(e.id)} onClick={(ev) => ev.stopPropagation()} aria-label={`Include ${e.name}`} />
                    </td>
                    <td className="nowrap"><strong>{e.name}</strong></td>
                    <td className="num secondary">{e.params ?? '–'}</td>
                    <td className="secondary nowrap">{e.passes}</td>
                    <td className="secondary">
                      {e.notes} <span className="muted mono">{e.repo}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            A model is downloaded from Hugging Face the first time it is used (70 MB to 1.7 GB each). Only one is kept in
            memory at a time. The models that take one pass per label get slower as the number of labels grows.
          </p>
          {task.golden_rows === 0 && <Banner tone="warning">This example has no golden data yet. Add it in the Dataset tab.</Banner>}
          {error && <Banner tone="error">{error}</Banner>}
          <div className="row">
            <Button variant="primary" onClick={start} disabled={anyActive || selectedEngines.length < 2 || task.golden_rows === 0}>
              {anyActive ? 'Comparison in progress' : `Compare ${selectedEngines.length} models`}
            </Button>
            {selectedEngines.length < 2 && <span className="small secondary">Choose at least two models.</span>}
          </div>
        </div>
      </Card>

      {selected && (
        <Card
          title={`${selected.version} on ${selected.sample_size} ${task.items}`}
          sub={`Started ${when(selected.timestamp)}. Click a model to open its full evaluation.`}
          flush
        >
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Status</th>
                  <th style={{ width: '34%' }}>Accuracy</th>
                  <th>Macro-F1</th>
                  <th>Time per {task.item}</th>
                  <th>Size</th>
                </tr>
              </thead>
              <tbody>
                {selected.runs.map((r) => (
                  <tr key={r.eval_id} className="clickable" onClick={() => open('evaluation', { tab: 'evaluation', evalId: r.eval_id })}>
                    <td className="nowrap">
                      <strong>{engineName(r.engine)}</strong>
                      {r.status === 'completed' && r.accuracy === bestAccuracy && <> <span className="badge good">Most accurate</span></>}
                      {r.status === 'completed' && r.ms_per_item === fastest && done.length > 1 && <> <span className="badge accent">Fastest</span></>}
                    </td>
                    <td><StatusBadge status={r.status} /></td>
                    <td>
                      {r.status === 'completed' ? (
                        <div className="meter">
                          <div className="track" style={{ height: 10 }}>
                            <div style={{ width: `${(r.accuracy ?? 0) * 100}%`, background: 'var(--series-1)' }} />
                          </div>
                          <span className="num" style={{ width: 52, textAlign: 'right' }}>{pct(r.accuracy)}</span>
                        </div>
                      ) : r.status === 'running' ? (
                        <div className="stack" style={{ gap: 4 }}>
                          <ProgressBar done={r.progress_done} total={r.progress_total} />
                          <span className="small secondary num">{r.progress_done} of {r.progress_total}</span>
                        </div>
                      ) : r.status === 'queued' ? (
                        <span className="muted">Waiting for its turn</span>
                      ) : (
                        <span className="small clamp" style={{ color: 'var(--critical)' }}>{r.error}</span>
                      )}
                    </td>
                    <td className="num">{pct(r.macro_f1)}</td>
                    <td className="num">{r.ms_per_item != null ? `${r.ms_per_item} ms` : '–'}</td>
                    <td className="num secondary">{spec(r.engine)?.params ?? '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ padding: '4px 16px 12px' }}>
            Time is the scoring only, on this machine, after the model has loaded. Accuracy on {selected.sample_size} rows has a margin of
            roughly ±{(100 / Math.sqrt(selected.sample_size)).toFixed(0)} points, so small gaps between models are not meaningful.
            A prompt written for one model can suit another badly: a model that scores near chance here usually needs its
            own wording (shorter label descriptions, a hypothesis template), which the Prompt Optimizer can find for it.
          </p>
        </Card>
      )}

      <Card title="Past comparisons" flush>
        {comparisons.length === 0 ? (
          <EmptyState title="No comparisons yet for this example" />
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Prompt</th>
                  <th>Sample</th>
                  <th>Models</th>
                  <th>Most accurate</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((c) => {
                  const finished = c.runs.filter((r) => r.status === 'completed')
                  const top = finished.reduce<RunRow | null>((best, r) => (best === null || (r.accuracy ?? 0) > (best.accuracy ?? 0) ? r : best), null)
                  return (
                    <tr key={c.comparison_id} className={`clickable ${selectedId === c.comparison_id ? 'selected' : ''}`} onClick={() => setSelectedId(c.comparison_id)}>
                      <td className="nowrap">{when(c.timestamp)}</td>
                      <td>{c.version}</td>
                      <td className="num">{c.sample_size}</td>
                      <td className="num">{c.active ? <StatusBadge status="running" /> : c.runs.length}</td>
                      <td>{top ? `${engineName(top.engine)} · ${pct(top.accuracy)}` : '–'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
