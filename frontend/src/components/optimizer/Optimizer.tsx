import { useEffect, useState } from 'react'
import { api, apiError, pct, when } from '../../api'
import { useApp, useTask } from '../../context'
import { Banner, Button, Card, EmptyState, Field, StatusBadge } from '../../ui'
import RunView from './RunView'
import { Run, RunSummary } from './types'

const POLL_MS = 2000

interface Options {
  metrics: Record<string, string>
  laya_models: string[]
  serp_available: boolean
  keep_confidence: number
}

interface RunSettings {
  // '' means the model the start version was saved for.
  engine: string
  sampleSize: number
  loops: number
  layaModel: string
  metric: string
  calibrate: boolean
  useSerp: boolean
}

const DEFAULTS: RunSettings = { engine: '', sampleSize: 400, loops: 10, layaModel: 'typed-decisions', metric: 'balanced', calibrate: true, useSerp: false }

// Run settings are remembered per example, in this browser.
function useRunSettings(taskId: string): [RunSettings, (patch: Partial<RunSettings>) => void] {
  const key = `system1.optimizer.${taskId}`
  const [settings, setSettings] = useState<RunSettings>(() => {
    try {
      return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(key) ?? '{}') }
    } catch {
      return DEFAULTS
    }
  })
  const update = (patch: Partial<RunSettings>) => {
    const next = { ...settings, ...patch }
    setSettings(next)
    localStorage.setItem(key, JSON.stringify(next))
  }
  return [settings, update]
}

export default function Optimizer() {
  const { task, versions, refreshVersions, intent, clearIntent } = useTask()
  const { llm, activity, refreshActivity, openSettings, tasks, go, engines } = useApp()
  const [settings, update] = useRunSettings(task.id)
  const [startVersion, setStartVersion] = useState('v1_baseline')
  const [options, setOptions] = useState<Options | null>(null)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [run, setRun] = useState<Run | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)

  const running = run?.status === 'running'
  const otherLoop = activity.loop && activity.loop.task !== task.id ? activity.loop : null
  const myLoopId = activity.loop?.task === task.id ? activity.loop.run_id : null

  const loadRuns = () =>
    api.get<{ runs: RunSummary[] }>('/karpathy-loop/runs', { params: { task: task.id } }).then((res) => setRuns(res.data.runs)).catch(() => undefined)

  const loadRun = async (runId: string) => {
    try {
      const res = await api.get<Run>(`/karpathy-loop/runs/${runId}`)
      setRun(res.data)
      if (res.data.status !== 'running') {
        loadRuns()
        refreshVersions()
        refreshActivity()
      }
    } catch (err) {
      setError(apiError(err, 'Failed to load the run'))
    }
  }

  useEffect(() => {
    api.get<Options>('/karpathy-loop/options', { params: { task: task.id } }).then((res) => setOptions(res.data)).catch(() => undefined)
    loadRuns()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Another tab (Compare Models) can ask for a prompt and a model to be set up, ready to start.
  useEffect(() => {
    if (intent?.tab !== 'optimizer') return
    if (!running) {
      if (intent.version) setStartVersion(intent.version)
      if (intent.engine) update({ engine: intent.engine })
    }
    clearIntent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intent, clearIntent])

  // Attach to this example's run in progress, wherever it was started from.
  useEffect(() => {
    if (myLoopId && run?.run_id !== myLoopId) loadRun(myLoopId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [myLoopId])

  // Follow the run on screen while it is running. This keeps going while the page is hidden.
  const followId = running ? run.run_id : null
  useEffect(() => {
    if (!followId) return
    const timer = window.setInterval(() => loadRun(followId), POLL_MS)
    return () => window.clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [followId])

  const start = async () => {
    setStarting(true)
    setError(null)
    try {
      const res = await api.post<Run>('/karpathy-loop', {
        task: task.id,
        loops: settings.loops,
        sample_size: settings.sampleSize,
        start_version: startVersion,
        laya_model: settings.layaModel,
        engine: settings.engine || undefined,
        metric: settings.metric,
        calibrate: settings.calibrate,
        use_serp: settings.useSerp && task.supports_serp,
      })
      setRun({ ...res.data, saved_versions: [] })
      loadRuns()
      refreshActivity()
    } catch (err) {
      setError(apiError(err, 'Failed to start the run'))
      refreshActivity()
    } finally {
      setStarting(false)
    }
  }

  const stop = async () => {
    if (!run) return
    try {
      await api.post(`/karpathy-loop/runs/${run.run_id}/stop`)
    } catch (err) {
      setError(apiError(err, 'Failed to stop the run'))
    }
  }

  const active = llm ? llm.providers[llm.active] : null
  const llmReady = active ? (active.kind === 'cli' ? active.cli_found : !active.needs_key || active.api_key_set || active.kind === 'anthropic') : false
  const objective = (run?.metric && options?.metrics[run.metric]) || 'accuracy'

  return (
    <div className="page">
      <Card
        title="Optimize the prompt"
        sub={`An LLM rewrites the prompt the model is given. Each proposal is scored on a dev set of golden ${task.items} and kept only if it beats the current best by more than noise; ${task.items} the LLM never sees are scored at the end. (The method is Karpathy's autoresearch loop.)`}
      >
        <div className="stack">
          <div className="row between">
            <span className="secondary">
              Proposals by{' '}
              {active ? (
                <strong>
                  {active.label}
                  {active.model ? ` · ${active.model}` : ''}
                </strong>
              ) : (
                '…'
              )}
            </span>
            <button className="link" onClick={openSettings}>Change LLM</button>
          </div>
          {active && !llmReady && (
            <Banner tone="warning" action={<Button small onClick={openSettings}>Open settings</Button>}>
              {active.kind === 'cli' ? `${active.label} was not found on the server.` : `No API key is set for ${active.label}.`}
            </Banner>
          )}

          <div className="row end">
            <Field label="Start from">
              <select className="input" value={startVersion} onChange={(e) => setStartVersion(e.target.value)} disabled={running}>
                {versions.map((v) => (
                  <option key={v.version} value={v.version}>
                    {v.version}
                    {v.accuracy != null ? ` (${pct(v.accuracy)})` : ''}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Model to optimize">
              <select className="input" value={settings.engine} onChange={(e) => update({ engine: e.target.value })} disabled={running}>
                <option value="">The start version's own</option>
                {engines.map((e) => (
                  <option key={e.id} value={e.id}>{e.name}</option>
                ))}
              </select>
            </Field>
            <Field label="Rounds">
              <input className="input narrow num" type="number" min={1} max={100} value={settings.loops} disabled={running}
                onChange={(e) => update({ loops: Math.max(1, Math.min(100, parseInt(e.target.value) || 10)) })} />
            </Field>
            <Field label="Dev set size">
              <input className="input narrow num" type="number" min={20} max={10000} value={settings.sampleSize} disabled={running}
                onChange={(e) => update({ sampleSize: Math.max(20, Math.min(10000, parseInt(e.target.value) || 400)) })} />
            </Field>
            {(settings.engine === '' || settings.engine === 'laya') && (
              <Field label="Laya checkpoint">
                <select className="input" value={settings.layaModel} onChange={(e) => update({ layaModel: e.target.value })} disabled={running}>
                  {(options?.laya_models ?? [settings.layaModel]).map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </Field>
            )}
            <Field label="Objective">
              <select className="input" value={settings.metric} onChange={(e) => update({ metric: e.target.value })} disabled={running}>
                {Object.entries(options?.metrics ?? { [settings.metric]: settings.metric }).map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
            </Field>
          </div>
          <label className="check">
            <input type="checkbox" checked={settings.calibrate} onChange={(e) => update({ calibrate: e.target.checked })} disabled={running} />
            Calibrate label bias on the dev set (stops one label swallowing every prediction)
          </label>
          {task.supports_serp && (
            <label className="check">
              <input type="checkbox" checked={settings.useSerp} onChange={(e) => update({ useSerp: e.target.checked })} disabled={running || !options?.serp_available} />
              Let the LLM add search-result context to the input
              {options && !options.serp_available ? ' (needs data/test_db.json)' : ' (only golden queries have it)'}
            </label>
          )}

          {otherLoop && (
            <Banner tone="warning" action={<Button small onClick={() => go(otherLoop.task, 'optimizer')}>Go to it</Button>}>
              A run for “{tasks.find((t) => t.id === otherLoop.task)?.name ?? otherLoop.task}” is in progress. Only one run goes at a time.
            </Banner>
          )}
          {task.golden_rows === 0 && <Banner tone="warning">This example has no golden data yet. Add it in the Dataset tab.</Banner>}
          {error && <Banner tone="error">{error}</Banner>}
          <div className="row">
            <Button variant="primary" onClick={start} disabled={starting || running || otherLoop !== null || task.golden_rows === 0}>
              {starting ? 'Starting…' : running ? 'Run in progress' : `Start run (${settings.loops} rounds)`}
            </Button>
          </div>
        </div>
      </Card>

      {run && (
        <RunView run={run} objective={objective} keepConfidence={options?.keep_confidence ?? 0.8} onStop={stop} onChanged={() => loadRun(run.run_id)} />
      )}

      <Card title="Past runs" flush>
        {runs.length === 0 ? (
          <EmptyState title="No runs yet for this example" />
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Status</th>
                  <th>LLM</th>
                  <th>Rounds</th>
                  <th>Best round</th>
                  <th>Dev accuracy: start → best</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} className={`clickable ${run?.run_id === r.run_id ? 'selected' : ''}`} onClick={() => loadRun(r.run_id)}>
                    <td className="nowrap">{when(r.started_at)}</td>
                    <td><StatusBadge status={r.status} /></td>
                    <td className="secondary">
                      {r.llm.provider}
                      {r.llm.model ? ` · ${r.llm.model}` : ''}
                    </td>
                    <td className="num">{r.completed_iterations}/{r.loops}</td>
                    <td className="num">{r.best_iteration ? r.best_iteration : '–'}</td>
                    <td className="num">{pct(r.baseline_accuracy)} → {pct(r.best_accuracy)}</td>
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
