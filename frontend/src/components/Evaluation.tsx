import { useEffect, useState } from 'react'
import { api, apiError, pct, when } from '../api'
import { useApp, useTask } from '../context'
import { Banner, Button, Card, EmptyState, Field, LabelBadge, ProgressBar, StatTile, StatusBadge } from '../ui'

interface ClassMetrics {
  precision: number
  recall: number
  f1: number
  support: number
}

interface ErrorCase {
  text: string
  predicted: string
  actual: string
  confidence: number
}

interface Summary {
  eval_id: string
  engine: string
  ms_per_item: number | null
  version: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  progress_done: number
  progress_total: number
  error: string | null
  sample_size: number
  accuracy: number | null
  macro_f1: number | null
  timestamp: string
}

interface Result extends Summary {
  total_cases?: number
  per_class_metrics?: Record<string, ClassMetrics>
  // confusion_matrix[actual][predicted]
  confusion_matrix?: Record<string, Record<string, number>>
  error_count?: number
  top_errors?: ErrorCase[]
  total_golden_data?: number
  eval_language?: string | null
  excluded_other_language?: number
}

const POLL_MS = 1500
const ERROR_PREVIEW_CHARS = 300

export default function Evaluation() {
  const { task, color, versions, refreshVersions, intent, clearIntent } = useTask()
  const { refreshActivity, engines, engineName } = useApp()
  // '' means the model the version was saved for.
  const [engine, setEngine] = useState('')
  const [version, setVersion] = useState('v1_baseline')
  const [sampleSize, setSampleSize] = useState(200)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [history, setHistory] = useState<Summary[]>([])

  const loadHistory = async (): Promise<Summary[]> => {
    try {
      const res = await api.get<Summary[]>('/evaluations', { params: { task: task.id } })
      setHistory(res.data)
      return res.data
    } catch {
      return []
    }
  }

  const openEvaluation = async (evalId: string) => {
    try {
      const res = await api.get<Result>(`/evaluations/${evalId}`)
      setResult(res.data)
      return res.data
    } catch (err) {
      setError(apiError(err, 'Failed to load the evaluation'))
      return null
    }
  }

  // On opening the page, pick up an evaluation that is already running (e.g. after a reload).
  useEffect(() => {
    loadHistory().then((items) => {
      const running = items.find((item) => item.status === 'running' || item.status === 'queued')
      if (running) openEvaluation(running.eval_id)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (intent?.tab !== 'evaluation') return
    if (intent.version) setVersion(intent.version)
    if (intent.evalId) {
      openEvaluation(intent.evalId)
      loadHistory()
    }
    clearIntent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intent, clearIntent])

  // Follow the evaluation on screen until it stops running. This keeps going while the page is hidden.
  const active = result?.status === 'running' || result?.status === 'queued'
  const followId = active ? result.eval_id : null
  useEffect(() => {
    if (!followId) return
    const timer = window.setInterval(async () => {
      const latest = await openEvaluation(followId)
      if (latest && latest.status !== 'running' && latest.status !== 'queued') {
        loadHistory()
        refreshVersions()
      }
    }, POLL_MS)
    return () => window.clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [followId])

  const start = async () => {
    setError(null)
    try {
      const res = await api.post<Result>('/evaluate', { task: task.id, version, engine: engine || undefined, sample_size: sampleSize })
      setResult(res.data)
      loadHistory()
      refreshActivity()
    } catch (err) {
      setError(apiError(err, 'Failed to start the evaluation'))
    }
  }

  const running = active
  const labels = result?.confusion_matrix ? Object.keys(result.confusion_matrix) : []

  return (
    <div className="page">
      <Card title="Evaluate a model version" sub="Scores the version on a random sample of the golden dataset. It runs in the background: you can leave this page and come back.">
        <div className="stack">
          <div className="row end">
            <Field label="Model version">
              <select className="input" value={version} onChange={(e) => setVersion(e.target.value)} disabled={running}>
                {versions.map((v) => (
                  <option key={v.version} value={v.version}>{v.version}</option>
                ))}
              </select>
            </Field>
            <Field label="Model">
              <select className="input" value={engine} onChange={(e) => setEngine(e.target.value)} disabled={running}>
                <option value="">The version's own</option>
                {engines.map((e) => (
                  <option key={e.id} value={e.id}>{e.name}</option>
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
                onChange={(e) => setSampleSize(Math.max(10, Math.min(10000, parseInt(e.target.value) || 200)))}
                disabled={running}
              />
            </Field>
            <Button variant="primary" onClick={start} disabled={running || task.golden_rows === 0}>
              {running ? 'Evaluating…' : 'Run evaluation'}
            </Button>
          </div>
          {task.golden_rows === 0 && <Banner tone="warning">This example has no golden data yet. Add it in the Dataset tab.</Banner>}
          {error && <Banner tone="error">{error}</Banner>}
        </div>
      </Card>

      {result && active && (
        <Card title={`Evaluating ${result.version} on ${engineName(result.engine)}`}>
          <div className="stack">
            <ProgressBar done={result.progress_done} total={result.progress_total} />
            <span className="small secondary num">
              {result.progress_done} of {result.progress_total} {task.items}
            </span>
          </div>
        </Card>
      )}

      {result && (result.status === 'failed' || result.status === 'interrupted') && (
        <Banner tone="error">
          The evaluation of {result.version} {result.status === 'failed' ? 'failed' : 'was interrupted'}: {result.error}
        </Banner>
      )}

      {result && result.status === 'completed' && result.per_class_metrics && result.confusion_matrix && (
        <>
          <div>
            <div className="stats">
              <StatTile label="Accuracy" value={pct(result.accuracy, 2)} />
              <StatTile label="Macro-F1" value={pct(result.macro_f1, 2)} />
              <StatTile label="Correct" value={(result.total_cases ?? 0) - (result.error_count ?? 0)} />
              <StatTile label="Errors" value={result.error_count ?? 0} />
              <StatTile label="Sample" value={result.total_cases ?? 0} note={`of ${result.total_golden_data ?? '–'} golden ${task.items}`} />
            </div>
            <p className="small muted" style={{ marginTop: 8 }}>
              {result.version} on {engineName(result.engine)} · evaluated {when(result.timestamp)}
              {result.ms_per_item != null && ` · ${result.ms_per_item} ms per ${task.item}`}
              {result.eval_language && ` · ${result.eval_language} only (${result.excluded_other_language} in other languages left out)`}
            </p>
          </div>

          <div className="grid-2">
            <Card title="Per label" flush>
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Label</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>Rows</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.per_class_metrics).map(([label, m]) => (
                      <tr key={label}>
                        <td><LabelBadge label={label} color={color(label)} /></td>
                        {[m.precision, m.recall, m.f1].map((value, i) => (
                          <td key={i}>
                            <div className="meter">
                              <div className="track">
                                <div style={{ width: `${value * 100}%`, background: color(label) }} />
                              </div>
                              <span className="num small">{value.toFixed(2)}</span>
                            </div>
                          </td>
                        ))}
                        <td className="num">{m.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card title="Confusion matrix" sub="Rows are the golden label, columns what the model predicted.">
              <div className="table-wrap">
                <table className="matrix num">
                  <thead>
                    <tr>
                      <th></th>
                      {labels.map((label) => (
                        <th key={label}>{label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {labels.map((actual) => (
                      <tr key={actual}>
                        <th>{actual}</th>
                        {labels.map((predicted) => {
                          const count = result.confusion_matrix![actual][predicted]
                          return (
                            <td key={predicted} className={predicted === actual ? 'hit' : count > 0 ? 'miss' : ''} title={`Golden ${actual}, predicted ${predicted}`}>
                              {count}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          <Card title="Most confident errors" sub="Wrong predictions the model was surest about." flush>
            {(result.top_errors ?? []).length === 0 ? (
              <EmptyState title="No errors in this sample" />
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{task.input_label}</th>
                      <th>Predicted</th>
                      <th>Golden</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.top_errors!.map((e, i) => (
                      <tr key={i}>
                        <td className="wrap">{e.text.length > ERROR_PREVIEW_CHARS ? `${e.text.slice(0, ERROR_PREVIEW_CHARS)}…` : e.text}</td>
                        <td><LabelBadge label={e.predicted} color={color(e.predicted)} /></td>
                        <td><LabelBadge label={e.actual} color={color(e.actual)} /></td>
                        <td className="num">{(e.confidence * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}

      <Card title="Past evaluations" flush>
        {history.length === 0 ? (
          <EmptyState title="No evaluations yet for this example" />
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Evaluated</th>
                  <th>Version</th>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Sample</th>
                  <th>Accuracy</th>
                  <th>Macro-F1</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.eval_id} className={`clickable ${result?.eval_id === item.eval_id ? 'selected' : ''}`} onClick={() => openEvaluation(item.eval_id)}>
                    <td className="nowrap">{when(item.timestamp)}</td>
                    <td>{item.version}</td>
                    <td className="secondary">{engineName(item.engine)}</td>
                    <td><StatusBadge status={item.status} /></td>
                    <td className="num">{item.sample_size}</td>
                    <td className="num">{pct(item.accuracy)}</td>
                    <td className="num">{pct(item.macro_f1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
