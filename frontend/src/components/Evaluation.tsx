import { useState, useEffect } from 'react'
import { api, apiError, pct } from '../api'
import { useTask } from '../TaskContext'
import '../styles/Evaluation.css'

interface PerClassMetrics {
  precision: number
  recall: number
  f1: number
  support: number
}

// confusion_matrix[actual][predicted]
interface ConfusionMatrix {
  [actual: string]: {
    [predicted: string]: number
  }
}

interface ErrorCase {
  text: string
  predicted: string
  actual: string
  confidence: number
  correct: boolean
}

interface EvaluationResult {
  eval_id: string
  model: string
  version: string
  total_cases: number
  accuracy: number
  macro_f1: number
  per_class_metrics: Record<string, PerClassMetrics>
  confusion_matrix: ConfusionMatrix
  error_count: number
  error_rate: number
  top_errors: ErrorCase[]
  timestamp: string
  sample_size?: number
  total_golden_data?: number
  eval_language?: string | null
  excluded_other_language?: number
}

interface EvaluationSummary {
  eval_id: string
  version: string
  sample_size: number
  accuracy: number
  macro_f1: number
  timestamp: string
}

const ERROR_PREVIEW_CHARS = 300

export default function Evaluation() {
  const { task, color } = useTask()
  const [version, setVersion] = useState('v1_baseline')
  const [versions, setVersions] = useState<string[]>(['v1_baseline'])
  const [sampleSize, setSampleSize] = useState(100)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<EvaluationResult | null>(null)
  const [history, setHistory] = useState<EvaluationSummary[]>([])

  const loadHistory = () =>
    api
      .get<EvaluationSummary[]>('/evaluations', { params: { task: task.id } })
      .then((res) => setHistory(res.data))
      .catch((err) => console.error('Failed to load past evaluations:', err))

  useEffect(() => {
    api
      .get<{ version: string }[]>('/models', { params: { task: task.id } })
      .then((res) => setVersions(res.data.map((m) => m.version)))
      .catch((err) => console.error('Failed to load model versions:', err))
    loadHistory()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await api.post<EvaluationResult>('/evaluate', {
        task: task.id,
        model: 'laya',
        version,
        sample_size: sampleSize,
      })
      setResult(response.data)
      loadHistory()
    } catch (err) {
      setError(apiError(err, 'Failed to run evaluation'))
    } finally {
      setLoading(false)
    }
  }

  const openEvaluation = async (evalId: string) => {
    setError(null)
    try {
      const response = await api.get<EvaluationResult>(`/evaluations/${evalId}`)
      setResult(response.data)
    } catch (err) {
      setError(apiError(err, 'Failed to load the evaluation'))
    }
  }

  const getIntentColor = color

  const getAccuracyGrade = (accuracy: number): string => {
    if (accuracy >= 0.9) return 'A'
    if (accuracy >= 0.8) return 'B'
    if (accuracy >= 0.7) return 'C'
    if (accuracy >= 0.6) return 'D'
    return 'F'
  }

  const getAccuracyColor = (accuracy: number): string => {
    if (accuracy >= 0.9) return '#51cf66'
    if (accuracy >= 0.8) return '#69db7c'
    if (accuracy >= 0.7) return '#ffd43b'
    if (accuracy >= 0.6) return '#ff922b'
    return '#ff6b6b'
  }

  return (
    <div className="evaluation">
      <div className="section">
        <h2>Model Evaluation: {task.name}</h2>
        <p className="description">
          Score a model version of this example on a random sample of its golden dataset. Every evaluation is saved
          and listed under Past Evaluations.
        </p>

        <div className="eval-controls">
          <div className="control-group">
            <label htmlFor="version">Version:</label>
            <select
              id="version"
              value={version}
              onChange={(e) => setVersion(e.target.value)}
              disabled={loading}
            >
              {versions.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="sample-size">Sample Size:</label>
            <input
              id="sample-size"
              type="number"
              value={sampleSize}
              onChange={(e) => setSampleSize(Math.max(10, Math.min(10000, parseInt(e.target.value) || 100)))}
              disabled={loading}
              min="10"
              max="10000"
            />
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="analyze-btn"
          >
            {loading ? 'Running Evaluation...' : 'Run Evaluation'}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}
      </div>

      {loading && (
        <div className="section loading-section">
          <h3>Evaluation in Progress...</h3>
          <div className="progress-indicator">
            <div className="spinner"></div>
            <p>Running Laya on the sampled golden {task.items}...</p>
          </div>
        </div>
      )}

      {result && (
        <>
          <div className="section results-header">
            <div className="accuracy-display">
              <div className="accuracy-grade" style={{ color: getAccuracyColor(result.accuracy) }}>
                {getAccuracyGrade(result.accuracy)}
              </div>
              <div className="accuracy-info">
                <div className="accuracy-value">
                  {(result.accuracy * 100).toFixed(2)}%
                </div>
                <div className="accuracy-label">Accuracy</div>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-label">Macro-F1</span>
                <span className="metric-value">{pct(result.macro_f1)}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Total Cases</span>
                <span className="metric-value">{result.total_cases}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Correct</span>
                <span className="metric-value" style={{ color: '#51cf66' }}>
                  {result.total_cases - result.error_count}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Errors</span>
                <span className="metric-value" style={{ color: '#ff6b6b' }}>
                  {result.error_count}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Error Rate</span>
                <span className="metric-value">{(result.error_rate * 100).toFixed(2)}%</span>
              </div>
            </div>
          </div>

          <div className="section per-class-metrics">
            <h3>Per-Class Metrics</h3>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.per_class_metrics).map(([intent, metrics]) => (
                  <tr key={intent}>
                    <td>
                      <span
                        className="intent-badge"
                        style={{
                          backgroundColor: `${getIntentColor(intent)}33`,
                          color: getIntentColor(intent),
                        }}
                      >
                        {intent}
                      </span>
                    </td>
                    <td>
                      <div className="metric-bar">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${metrics.precision * 100}%`,
                            backgroundColor: getIntentColor(intent),
                          }}
                        />
                        <span className="metric-value">{metrics.precision.toFixed(3)}</span>
                      </div>
                    </td>
                    <td>
                      <div className="metric-bar">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${metrics.recall * 100}%`,
                            backgroundColor: getIntentColor(intent),
                          }}
                        />
                        <span className="metric-value">{metrics.recall.toFixed(3)}</span>
                      </div>
                    </td>
                    <td>
                      <div className="metric-bar">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${metrics.f1 * 100}%`,
                            backgroundColor: getIntentColor(intent),
                          }}
                        />
                        <span className="metric-value">{metrics.f1.toFixed(3)}</span>
                      </div>
                    </td>
                    <td className="support-value">{metrics.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="section confusion-matrix-section">
            <h3>Confusion Matrix</h3>
            <p className="description">Rows are the golden label, columns are what the model predicted</p>
            <div className="confusion-matrix">
              <table>
                <thead>
                  <tr>
                    <th>golden ↓ / predicted →</th>
                    {Object.keys(result.confusion_matrix).map((label) => (
                      <th key={label}>{label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.confusion_matrix).map(([actual, row]) => (
                    <tr key={actual}>
                      <td className="row-header">{actual}</td>
                      {Object.entries(row).map(([predicted, count]) => (
                        <td
                          key={`${actual}-${predicted}`}
                          className={`matrix-cell ${predicted === actual ? 'correct' : count > 0 ? 'incorrect' : ''}`}
                          title={`Golden: ${actual}, predicted: ${predicted}`}
                        >
                          {count}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="section top-errors-section">
            <h3>Top Errors</h3>
            <p className="description">The wrong predictions the model was most confident about</p>
            {result.top_errors.length === 0 ? (
              <div className="no-errors">Perfect! No errors found.</div>
            ) : (
              <div className="errors-list">
                {result.top_errors.map((error, idx) => (
                  <div key={idx} className="error-item">
                    <div className="error-header">
                      <span className="error-number">#{idx + 1}</span>
                      <span className="error-keyword">
                        {error.text.length > ERROR_PREVIEW_CHARS ? `${error.text.slice(0, ERROR_PREVIEW_CHARS)}…` : error.text}
                      </span>
                      <span className="error-confidence">
                        {(error.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                    <div className="error-details">
                      <div className="error-detail">
                        <span className="label">Predicted:</span>
                        <span
                          className="intent-badge"
                          style={{
                            backgroundColor: `${getIntentColor(error.predicted)}33`,
                            color: getIntentColor(error.predicted),
                          }}
                        >
                          {error.predicted}
                        </span>
                      </div>
                      <span className="arrow">→</span>
                      <div className="error-detail">
                        <span className="label">Actual:</span>
                        <span
                          className="intent-badge"
                          style={{
                            backgroundColor: `${getIntentColor(error.actual)}33`,
                            color: getIntentColor(error.actual),
                          }}
                        >
                          {error.actual}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="section info-section">
            <div className="eval-metadata">
              <p>
                <strong>Model:</strong> {result.model}
              </p>
              <p>
                <strong>Version:</strong> {result.version}
              </p>
              <p>
                <strong>Sample Size:</strong> {result.sample_size} / {result.total_golden_data}
              </p>
              {result.eval_language && (
                <p>
                  <strong>Language:</strong> {result.eval_language} only ({result.excluded_other_language} golden{' '}
                  {task.items} in other languages excluded)
                </p>
              )}
              <p>
                <strong>Evaluated:</strong> {new Date(`${result.timestamp}Z`).toLocaleString()}
              </p>
            </div>
          </div>
        </>
      )}

      <div className="section">
        <h3>Past Evaluations</h3>
        {history.length === 0 ? (
          <p className="description">No evaluations have been run for this example yet.</p>
        ) : (
          <table className="metrics-table history-table">
            <thead>
              <tr>
                <th>Evaluated</th>
                <th>Version</th>
                <th>Sample</th>
                <th>Accuracy</th>
                <th>Macro-F1</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr
                  key={item.eval_id}
                  className={result?.eval_id === item.eval_id ? 'selected' : ''}
                  onClick={() => openEvaluation(item.eval_id)}
                >
                  <td>{new Date(`${item.timestamp}Z`).toLocaleString()}</td>
                  <td>{item.version}</td>
                  <td>{item.sample_size}</td>
                  <td>{pct(item.accuracy)}</td>
                  <td>{pct(item.macro_f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
