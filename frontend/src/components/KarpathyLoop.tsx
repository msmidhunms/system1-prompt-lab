import { Fragment, useState, useEffect, useRef } from 'react'
import axios from 'axios'
import LLMSettings, { apiError } from './LLMSettings'
import '../styles/KarpathyLoop.css'

const API = 'http://localhost:8000/api'
const POLL_MS = 2000
const LABELS = ['informational', 'navigational', 'commercial', 'transactional']

interface GoldenDataStats {
  imported_data: number
  user_feedback_count: number
  unique_feedback_queries: number
  total_golden_data: number
}

interface PromptConfig {
  state_template: string
  instructions: string
  criteria: Record<string, string>
}

interface Metrics {
  accuracy: number
  macro_f1: number
  correct: number
  total: number
}

interface LoopIteration {
  iteration: number
  timestamp: string
  status: 'keep' | 'discard' | 'crash'
  hypothesis: string | null
  config: PromptConfig | null
  accuracy: number | null
  macro_f1: number | null
  delta_vs_best: number | null
  best_accuracy: number
  warnings: string[]
  error: string | null
}

interface Run {
  run_id: string
  status: 'running' | 'completed' | 'stopped' | 'failed' | 'interrupted'
  phase: string
  started_at: string
  finished_at: string | null
  llm: { provider: string; model: string }
  loops: number
  seed: number
  start_version: string
  dev_size: number
  holdout_size: number
  baseline_config: PromptConfig
  baseline: Metrics | null
  best_config: PromptConfig
  best_accuracy: number | null
  best_iteration: number
  improved: boolean
  iterations: LoopIteration[]
  holdout: { baseline: Metrics; best: Metrics } | null
  error: string | null
  log: string[]
}

interface RunSummary {
  run_id: string
  status: Run['status']
  started_at: string
  llm: { provider: string; model: string }
  loops: number
  completed_iterations: number
  dev_size: number
  baseline_accuracy: number | null
  best_accuracy: number | null
  best_iteration: number | null
  improved: boolean
}

interface ModelVersion {
  version: string
  accuracy: number | null
}

const pct = (value: number | null | undefined) => (value == null ? '-' : `${(value * 100).toFixed(2)}%`)
const signedPts = (value: number) => `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)} pts`

function ConfigView({ config, previous }: { config: PromptConfig; previous?: PromptConfig }) {
  const changed = (a: string, b: string | undefined) => previous !== undefined && a !== b
  return (
    <dl className="config-view">
      <div className="config-row">
        <dt className={changed(config.state_template, previous?.state_template) ? 'changed' : ''}>state template</dt>
        <dd>{config.state_template}</dd>
      </div>
      <div className="config-row">
        <dt className={changed(config.instructions, previous?.instructions) ? 'changed' : ''}>instructions</dt>
        <dd>{config.instructions}</dd>
      </div>
      {LABELS.map((label) => (
        <div key={label} className="config-row">
          <dt className={changed(config.criteria[label], previous?.criteria[label]) ? 'changed' : ''}>{label}</dt>
          <dd>{config.criteria[label]}</dd>
        </div>
      ))}
    </dl>
  )
}

export default function KarpathyLoop() {
  const [stats, setStats] = useState<GoldenDataStats>({
    imported_data: 0,
    user_feedback_count: 0,
    unique_feedback_queries: 0,
    total_golden_data: 0,
  })
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [run, setRun] = useState<Run | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [sampleSize, setSampleSize] = useState(200)
  const [numLoops, setNumLoops] = useState(10)
  const [startVersion, setStartVersion] = useState('v1_baseline')
  const [activeLLM, setActiveLLM] = useState('')
  const [expanded, setExpanded] = useState<number | null>(null)
  const [newModelName, setNewModelName] = useState('')
  const [showLog, setShowLog] = useState(false)
  const pollRef = useRef<number | null>(null)

  const running = run?.status === 'running'

  useEffect(() => {
    axios.get(`${API}/golden-data/stats`).then((res) => setStats(res.data)).catch((err) => console.error('Failed to load stats:', err))
    loadVersions()
    // Reattach to a loop that is already running on the server.
    loadRuns().then((activeId) => {
      if (activeId) loadRun(activeId)
    })
    return stopPolling
  }, [])

  useEffect(() => {
    if (!running || !run) return
    pollRef.current = window.setInterval(() => loadRun(run.run_id), POLL_MS)
    return stopPolling
  }, [running, run?.run_id])

  const stopPolling = () => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const loadVersions = () =>
    axios.get<ModelVersion[]>(`${API}/models`).then((res) => setVersions(res.data)).catch((err) => console.error('Failed to load versions:', err))

  const loadRuns = async (): Promise<string | null> => {
    try {
      const res = await axios.get<{ active_run_id: string | null; runs: RunSummary[] }>(`${API}/karpathy-loop/runs`)
      setRuns(res.data.runs)
      return res.data.active_run_id
    } catch (err) {
      console.error('Failed to load runs:', err)
      return null
    }
  }

  const loadRun = async (runId: string) => {
    try {
      const res = await axios.get<Run>(`${API}/karpathy-loop/runs/${runId}`)
      setRun(res.data)
      if (res.data.status !== 'running') loadRuns()
    } catch (err) {
      setError(apiError(err, 'Failed to load run'))
    }
  }

  const startKarpathyLoop = async () => {
    setStarting(true)
    setError(null)
    setNotice(null)
    setExpanded(null)
    try {
      const res = await axios.post<Run>(`${API}/karpathy-loop`, {
        loops: numLoops,
        sample_size: sampleSize,
        start_version: startVersion,
      })
      setRun(res.data)
    } catch (err) {
      setError(apiError(err, 'Failed to start Karpathy loop'))
    } finally {
      setStarting(false)
    }
  }

  const stopRun = async () => {
    if (!run) return
    try {
      await axios.post(`${API}/karpathy-loop/runs/${run.run_id}/stop`)
      setNotice('Stopping after the current round...')
    } catch (err) {
      setError(apiError(err, 'Failed to stop run'))
    }
  }

  const saveImprovedModel = async () => {
    if (!run || !newModelName.trim()) return
    setError(null)
    try {
      await axios.post(`${API}/save-model`, {
        model_name: newModelName.trim(),
        run_id: run.run_id,
        description: `Karpathy loop ${run.run_id}, best round ${run.best_iteration}`,
      })
      setNotice(`Saved as "${newModelName.trim()}". It is now selectable in Model Evaluation and as a starting point here.`)
      setNewModelName('')
      loadVersions()
    } catch (err) {
      setError(apiError(err, 'Failed to save model'))
    }
  }

  // The config each round was compared against: the best one at the time it ran.
  const bestBefore = (index: number): PromptConfig | undefined => {
    if (!run) return undefined
    for (let i = index - 1; i >= 0; i--) {
      const prev = run.iterations[i]
      if (prev.status === 'keep' && prev.config) return prev.config
    }
    return run.baseline_config
  }

  const devGain = run?.baseline && run.best_accuracy != null ? run.best_accuracy - run.baseline.accuracy : null

  return (
    <div className="karpathy-loop">
      <div className="section">
        <h2>Karpathy Loop - Prompt Autoresearch</h2>
        <p className="description">
          An LLM rewrites the text Laya is given (state template, question instructions, label descriptions). Each
          proposal is scored on a dev set of golden queries and kept only if accuracy improves. A holdout set the LLM
          never sees is scored at the end.
        </p>

        <div className="golden-data-summary">
          <h3>Golden Dataset Summary</h3>
          <div className="summary-stats">
            <div className="stat">
              <span className="stat-label">Imported Keywords:</span>
              <span className="stat-value">{stats.imported_data}</span>
            </div>
            <div className="stat">
              <span className="stat-label">User Feedback Entries:</span>
              <span className="stat-value">{stats.user_feedback_count}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Unique Feedback Queries:</span>
              <span className="stat-value">{stats.unique_feedback_queries}</span>
            </div>
            <div className="stat highlight">
              <span className="stat-label">Total Golden Data:</span>
              <span className="stat-value">{stats.total_golden_data}</span>
            </div>
          </div>
        </div>

        <LLMSettings disabled={running} onActiveChange={setActiveLLM} />

        <div className="loop-controls">
          <div className="control-group">
            <label htmlFor="sample-size">Dev Set Size:</label>
            <input
              id="sample-size"
              type="number"
              value={sampleSize}
              onChange={(e) => setSampleSize(Math.max(20, Math.min(10000, parseInt(e.target.value) || 200)))}
              disabled={running}
              min="20"
              max="10000"
            />
          </div>

          <div className="control-group">
            <label htmlFor="num-loops">Rounds:</label>
            <input
              id="num-loops"
              type="number"
              value={numLoops}
              onChange={(e) => setNumLoops(Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))}
              disabled={running}
              min="1"
              max="100"
            />
          </div>

          <div className="control-group">
            <label htmlFor="start-version">Start From:</label>
            <select id="start-version" value={startVersion} onChange={(e) => setStartVersion(e.target.value)} disabled={running}>
              {versions.map((v) => (
                <option key={v.version} value={v.version}>
                  {v.version}
                  {v.accuracy != null ? ` (${pct(v.accuracy)})` : ''}
                </option>
              ))}
            </select>
          </div>

          {running ? (
            <button onClick={stopRun} className="stop-loop-btn">
              Stop After This Round
            </button>
          ) : (
            <button onClick={startKarpathyLoop} disabled={starting || stats.total_golden_data === 0} className="start-loop-btn">
              {starting ? 'Starting...' : `Start Karpathy Loop (${numLoops} Rounds)`}
            </button>
          )}
        </div>
        {activeLLM && <p className="control-hint">Proposals come from: {activeLLM}</p>}

        {error && <div className="error-message">{error}</div>}
        {notice && <div className="success-message">{notice}</div>}
      </div>

      {run && (
        <div className={`section results-section status-${run.status}`}>
          <div className="run-header">
            <h3>Run {run.run_id}</h3>
            <span className={`status-badge ${run.status}`}>{run.status}</span>
          </div>
          <p className="run-meta">
            {run.llm.provider}
            {run.llm.model ? ` · ${run.llm.model}` : ''} · started from {run.start_version} · {run.dev_size} dev /{' '}
            {run.holdout_size} holdout queries
          </p>

          {running && (
            <div className="progress-line">
              <div className="spinner small"></div>
              <span>
                {run.phase} ({run.iterations.length} of {run.loops} rounds done)
              </span>
            </div>
          )}
          {run.status === 'failed' && <div className="error-message">Run failed: {run.error}</div>}
          {run.status === 'interrupted' && (
            <div className="error-message">The server restarted while this run was in progress. Results up to that point are shown.</div>
          )}

          <div className="results-summary">
            <div className="result-card">
              <label>Baseline (dev)</label>
              <div className="metric-value">{pct(run.baseline?.accuracy)}</div>
            </div>
            <div className="result-card">
              <label>Best (dev)</label>
              <div className="metric-value">{pct(run.best_accuracy)}</div>
            </div>
            <div className="result-card">
              <label>Dev Gain</label>
              <div className={`metric-value ${devGain != null && devGain > 0 ? 'positive' : 'neutral'}`}>
                {devGain == null ? '-' : signedPts(devGain)}
              </div>
            </div>
            <div className="result-card">
              <label>Best Round</label>
              <div className="metric-value">{run.best_iteration === 0 ? 'baseline' : `${run.best_iteration}/${run.loops}`}</div>
            </div>
            <div className="result-card">
              <label>Holdout: Baseline → Best</label>
              <div className="metric-value small">
                {run.holdout ? `${pct(run.holdout.baseline.accuracy)} → ${pct(run.holdout.best.accuracy)}` : 'scored at the end'}
              </div>
            </div>
          </div>

          <div className="iterations-list">
            <h4>Experiments</h4>
            {run.iterations.length === 0 ? (
              <p className="empty-message">No rounds finished yet.</p>
            ) : (
              <table className="iterations-table">
                <thead>
                  <tr>
                    <th>Round</th>
                    <th>Result</th>
                    <th>Dev Accuracy</th>
                    <th>vs Best</th>
                    <th>Hypothesis</th>
                  </tr>
                </thead>
                <tbody>
                  {run.iterations.map((iter, index) => (
                    <Fragment key={iter.iteration}>
                      <tr
                        className={`iteration-row ${iter.status}`}
                        onClick={() => setExpanded(expanded === iter.iteration ? null : iter.iteration)}
                      >
                        <td>
                          {iter.iteration}
                          {iter.iteration === run.best_iteration && <span className="best-badge">Best</span>}
                        </td>
                        <td>
                          <span className={`status-badge ${iter.status}`}>{iter.status}</span>
                        </td>
                        <td>{pct(iter.accuracy)}</td>
                        <td>
                          {iter.delta_vs_best != null && (
                            <span
                              className={`accuracy-change ${iter.delta_vs_best > 0 ? 'positive' : iter.delta_vs_best < 0 ? 'negative' : 'neutral'}`}
                            >
                              {signedPts(iter.delta_vs_best)}
                            </span>
                          )}
                        </td>
                        <td className="hypothesis">{iter.status === 'crash' ? iter.error : iter.hypothesis}</td>
                      </tr>
                      {expanded === iter.iteration && iter.config && (
                        <tr className="iteration-detail">
                          <td colSpan={5}>
                            <p className="control-hint">Proposed prompt. Highlighted fields differ from the best prompt at the time.</p>
                            <ConfigView config={iter.config} previous={bestBefore(index)} />
                            {iter.warnings.map((w) => (
                              <p key={w} className="warning-line">
                                {w}
                              </p>
                            ))}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="best-config">
            <h4>{run.best_iteration === 0 ? 'Current Best Prompt (baseline)' : `Current Best Prompt (round ${run.best_iteration})`}</h4>
            <ConfigView config={run.best_config} previous={run.best_iteration === 0 ? undefined : run.baseline_config} />
          </div>

          {!running && run.improved && (
            <div className="section model-saving-section">
              <h3>Save Improved Prompt</h3>
              <p>Save the best prompt as a named model version so it can be evaluated or used as the next starting point.</p>
              <div className="name-model-form">
                <input
                  type="text"
                  placeholder="e.g., laya_v2_improved"
                  value={newModelName}
                  onChange={(e) => setNewModelName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && saveImprovedModel()}
                />
                <button onClick={saveImprovedModel} disabled={!newModelName.trim()}>
                  Save Model
                </button>
              </div>
            </div>
          )}

          {!running && !run.improved && run.status !== 'failed' && (
            <div className="section no-improvement">
              <p>No proposal beat the starting prompt on the dev set in this run.</p>
            </div>
          )}

          <button className="secondary-btn log-toggle" onClick={() => setShowLog(!showLog)}>
            {showLog ? 'Hide Log' : 'Show Log'}
          </button>
          {showLog && <pre className="run-log">{run.log.join('\n')}</pre>}
          <p className="control-hint">
            Checkpoints, prompts and LLM replies for every round are in autoresearch/runs/{run.run_id}/
          </p>
        </div>
      )}

      {runs.length > 0 && (
        <div className="section">
          <h3>Past Runs</h3>
          <table className="iterations-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>LLM</th>
                <th>Rounds</th>
                <th>Dev: Baseline → Best</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className={`iteration-row ${run?.run_id === r.run_id ? 'selected' : ''}`} onClick={() => loadRun(r.run_id)}>
                  <td>{r.run_id}</td>
                  <td>
                    <span className={`status-badge ${r.status}`}>{r.status}</span>
                  </td>
                  <td>
                    {r.llm.provider}
                    {r.llm.model ? ` · ${r.llm.model}` : ''}
                  </td>
                  <td>
                    {r.completed_iterations}/{r.loops}
                  </td>
                  <td>
                    {pct(r.baseline_accuracy)} → {pct(r.best_accuracy)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="section info-section">
        <h3>How Karpathy Loop Works</h3>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <div>
              <strong>Baseline</strong>
              <p>Score the starting prompt with Laya on the dev set</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div>
              <strong>Propose</strong>
              <p>The LLM sees the metrics, the errors and past experiments, and rewrites the prompt</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div>
              <strong>Evaluate</strong>
              <p>Laya is run again with the new prompt on the same dev set</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">4</div>
            <div>
              <strong>Keep or Discard</strong>
              <p>Higher accuracy becomes the new best. Anything else is discarded and logged</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">5</div>
            <div>
              <strong>Holdout and Save</strong>
              <p>The best prompt is checked on unseen queries, then saved as a model version</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
