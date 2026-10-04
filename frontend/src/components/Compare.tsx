import { useEffect, useRef, useState } from 'react'
import { api, apiError, pct, when } from '../api'
import { useApp, useTask } from '../context'
import { Banner, Button, Card, EmptyState, Field, NumberInput, ProgressBar, StatusBadge } from '../ui'

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

// What a prompt usually needs to suit each family of model (the optimizer is told the same: PROMPT_NOTES in engines.py).
const WORDING: Record<string, string> = {
  laya: 'a JSON state whose field the instructions name, and short plain-language descriptions',
  gliclass: 'short instructions and plain-phrase label descriptions',
  verdict: 'label descriptions that complete the sentence “It is …”',
  nli: 'a hypothesis template (instructions with {} in them) and short noun-phrase descriptions',
  mlm: 'a clear question and short, distinct answer options',
}

// The 95% margin, in points, of an accuracy measured on n rows.
const margin = (accuracy: number, n: number) => 196 * Math.sqrt((accuracy * (1 - accuracy)) / n)

const completed = (c: Comparison) => c.runs.filter((r) => r.status === 'completed')

// The runs that share the top accuracy, once the comparison has finished and has at least two results to rank.
const mostAccurate = (c: Comparison): RunRow[] => {
  const done = completed(c)
  if (c.active || done.length < 2) return []
  const best = Math.max(...done.map((r) => r.accuracy ?? 0))
  return done.filter((r) => (r.accuracy ?? 0) === best)
}

// One prompt, the same sample of golden rows, several models side by side.
export default function Compare() {
  const { task, versions, refreshVersions, open } = useTask()
  const { engines, engineName, activity, refreshActivity } = useApp()
  const [version, setVersion] = useState('v1_baseline')
  const [sampleSize, setSampleSize] = useState(100)
  const [chosen, setChosen] = useState<string[] | null>(null)
  const [comparisons, setComparisons] = useState<Comparison[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  // Every model is ticked until the user changes the selection.
  const selectedEngines = chosen ?? engines.map((e) => e.id)
  // The version picked here can be renamed or deleted in the Models tab while this page stays open.
  const currentVersion = versions.some((v) => v.version === version) ? version : versions[0]?.version ?? 'v1_baseline'

  const load = async (): Promise<Comparison[] | null> => {
    try {
      const res = await api.get<Comparison[]>('/comparisons', { params: { task: task.id } })
      setComparisons(res.data)
      setLoadError(null)
      return res.data
    } catch (err) {
      setLoadError(apiError(err, 'Failed to load the comparisons'))
      return null
    }
  }

  useEffect(() => {
    load().then((items) => items?.[0] && setSelectedId(items[0].comparison_id))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const anyActive = comparisons.some((c) => c.active)
  // A comparison of this example started from another window shows in the app's activity before it shows here.
  const startedElsewhere = activity.evaluations.some((e) => e.task === task.id && e.comparison_id !== null)
  // A failed load is tried again, so a comparison that was started is not lost from view.
  const polling = anyActive || startedElsewhere || loadError !== null
  useEffect(() => {
    if (!polling) return
    const timer = window.setInterval(load, POLL_MS)
    return () => window.clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [polling])

  // When a comparison ends, the version's latest evaluation (Models tab) may be one of its runs.
  const wasActive = useRef(false)
  useEffect(() => {
    if (wasActive.current && !anyActive) {
      refreshVersions()
      refreshActivity()
    }
    wasActive.current = anyActive
  }, [anyActive, refreshVersions, refreshActivity])

  // A comparison times each model, so it runs only when nothing else is (the server refuses it otherwise).
  const otherWork = !anyActive && (activity.loop !== null || activity.evaluations.length > 0)

  const start = async () => {
    setError(null)
    try {
      const res = await api.post<{ comparison_id: string }[]>('/compare', {
        task: task.id,
        version: currentVersion,
        engines: selectedEngines,
        sample_size: sampleSize,
      })
      setSelectedId(res.data[0].comparison_id)
      await load()
    } catch (err) {
      setError(apiError(err, 'Failed to start the comparison'))
    }
    refreshActivity()
  }

  const toggle = (id: string) =>
    setChosen(selectedEngines.includes(id) ? selectedEngines.filter((e) => e !== id) : [...selectedEngines, id])

  const spec = (id: string) => engines.find((e) => e.id === id)
  const selected = comparisons.find((c) => c.comparison_id === selectedId) ?? null
  const done = selected ? completed(selected) : []
  // The badges wait for the whole comparison: the first model to finish is not yet the most accurate of anything.
  const ranked = selected !== null && !selected.active && done.length > 1
  const best = selected ? mostAccurate(selected) : []
  const fastest = ranked ? Math.min(...done.map((r) => r.ms_per_item ?? Infinity)) : null

  const n = selected?.sample_size ?? 0
  // The widest margin among the models' accuracies; before any has finished, the worst case (an accuracy of 50%).
  const widest = Math.max(1, Math.round(Math.max(...(done.length > 0 ? done.map((r) => margin(r.accuracy ?? 0.5, n)) : [margin(0.5, n)]))))
  // A model that cannot be told from guessing on this sample: within the margin of one in however many labels.
  const chance = 1 / task.labels.length
  const nearChance = done.filter((r) => (r.accuracy ?? 0) * 100 <= chance * 100 + margin(chance, n))
  // The compared version as it is now, if it still exists.
  const comparedVersion = selected ? versions.find((v) => v.version === selected.version) : undefined
  const biasedFor = comparedVersion?.config.label_bias ? engineName(comparedVersion.config.engine) : null

  return (
    <div className="page">
      <Card
        title="Compare models"
        sub={`Runs one prompt on several models, on the same random sample of golden ${task.items}, one model after another. The prompt is the version's instructions and label descriptions, put into each model's own input format. A label bias saved with the version is applied only on the model it was fitted on.`}
      >
        <div className="stack">
          <div className="row end">
            <Field label="Prompt (model version)">
              <select className="input" value={currentVersion} onChange={(e) => setVersion(e.target.value)}>
                {versions.map((v) => (
                  <option key={v.version} value={v.version}>{v.version}</option>
                ))}
              </select>
            </Field>
            <Field label="Sample size">
              <NumberInput value={sampleSize} min={10} max={10000} onChange={setSampleSize} />
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
          {loadError && <Banner tone="error">The comparisons could not be loaded ({loadError}). Trying again every few seconds.</Banner>}
          {error && <Banner tone="error">{error}</Banner>}
          <div className="row">
            <Button variant="primary" onClick={start} disabled={anyActive || otherWork || selectedEngines.length < 2 || task.golden_rows === 0}>
              {anyActive ? 'Comparison in progress' : `Compare ${selectedEngines.length} models`}
            </Button>
            {selectedEngines.length < 2 && <span className="small secondary">Choose at least two models.</span>}
            {otherWork && (
              <span className="small secondary">
                Other work is in progress (see the top bar). A comparison times each model, so it starts only when nothing else is running.
              </span>
            )}
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
                      {best.includes(r) && <> <span className="badge good">Most accurate</span></>}
                      {ranked && r.status === 'completed' && r.ms_per_item === fastest && <> <span className="badge accent">Fastest</span></>}
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
          <div className="small muted stack" style={{ padding: '4px 16px 12px', gap: 6 }}>
            <p>
              Time is the scoring only, on this machine, after the model has loaded. On {selected.sample_size} {task.items} an
              accuracy has a margin of up to about ±{widest} points, so a gap between two models that is smaller than that may be
              noise. The models are scored on the same {task.items}; a larger sample is what settles a close call.
            </p>
            {biasedFor && (
              <p>
                {selected.version} carries a label bias fitted on {biasedFor}. It is applied there only: the other models run
                without one, so part of their gap is calibration and not the wording of the prompt.
              </p>
            )}
            {nearChance.map((r) => (
              <p key={r.eval_id}>
                {engineName(r.engine)} scores within noise of chance ({pct(chance, 0)} with {task.labels.length} labels). A
                prompt written for one model can suit another badly: this one usually needs{' '}
                {WORDING[spec(r.engine)?.family ?? ''] ?? 'its own wording'}.{' '}
                <button className="link" onClick={() => open('optimizer', { tab: 'optimizer', version: comparedVersion?.version, engine: r.engine })}>
                  Find it with the Prompt Optimizer
                </button>
              </p>
            ))}
          </div>
        </Card>
      )}

      <Card title="Past comparisons" flush>
        {comparisons.length === 0 ? (
          <EmptyState title={loadError ? 'The comparisons could not be loaded' : 'No comparisons yet for this example'} />
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
                  const top = mostAccurate(c)
                  return (
                    <tr key={c.comparison_id} className={`clickable ${selectedId === c.comparison_id ? 'selected' : ''}`} onClick={() => setSelectedId(c.comparison_id)}>
                      <td className="nowrap">{when(c.timestamp)}</td>
                      <td>{c.version}</td>
                      <td className="num">{c.sample_size}</td>
                      <td className="num">{c.active ? <StatusBadge status="running" /> : c.runs.length}</td>
                      <td>{top.length > 0 ? `${top.map((r) => engineName(r.engine)).join(', ')} · ${pct(top[0].accuracy)}` : '–'}</td>
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
