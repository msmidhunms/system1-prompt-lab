import { useState, useEffect } from 'react'
import axios from 'axios'
import '../styles/Evaluation.css'

interface PerClassMetrics {
  precision: number
  recall: number
  f1: number
  support: number
}

interface ConfusionMatrix {
  [predicted: string]: {
    [actual: string]: number
  }
}

interface ErrorCase {
  keyword: string
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
  per_class_metrics: Record<string, PerClassMetrics>
  confusion_matrix: ConfusionMatrix
  error_count: number
  error_rate: number
  top_errors: ErrorCase[]
  timestamp: string
  sample_size?: number
  total_golden_data?: number
}

const INTENT_COLORS: Record<string, string> = {
  informational: '#64c8ff',
  navigational: '#ffb164',
  commercial: '#64ff96',
  transactional: '#c864ff',
}

export default function Evaluation() {
  const [model, setModel] = useState('laya')
  const [version, setVersion] = useState('v1_baseline')
  const [versions, setVersions] = useState<string[]>(['v1_baseline'])
  const [sampleSize, setSampleSize] = useState(100)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<EvaluationResult | null>(null)

  useEffect(() => {
    axios
      .get<{ version: string }[]>('http://localhost:8000/api/models')
      .then((res) => setVersions(res.data.map((m) => m.version)))
      .catch((err) => console.error('Failed to load model versions:', err))
  }, [])

  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post('http://localhost:8000/api/evaluate', {
        model,
        version,
      }, {
        params: {
          sample_size: sampleSize,
        }
      })
      setResult(response.data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to run evaluation'
      )
    } finally {
      setLoading(false)
    }
  }

  const getIntentColor = (intent: string): string => {
    return INTENT_COLORS[intent.toLowerCase()] || '#999'
  }

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
        <h2>Model Evaluation & Benchmarking</h2>
        <p className="description">
          Run evaluation on the baseline model against the golden dataset to measure accuracy and performance
        </p>

        <div className="eval-controls">
          <div className="control-group">
            <label htmlFor="model">Model:</label>
            <select
              id="model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={loading}
            >
              <option value="laya">Laya (System 1)</option>
              <option value="laya_v2">Laya V2</option>
            </select>
          </div>

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
            <p>Running predictions on all keywords in the database...</p>
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
                  <th>Intent</th>
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
            <p className="description">Predicted intent vs Actual intent</p>
            <div className="confusion-matrix">
              <table>
                <thead>
                  <tr>
                    <th></th>
                    {Object.keys(result.confusion_matrix)
                      .sort()
                      .map((intent) => (
                        <th key={intent}>{intent.slice(0, 3).toUpperCase()}</th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(result.confusion_matrix)
                    .sort()
                    .map((predicted) => (
                      <tr key={predicted}>
                        <td className="row-header">{predicted.slice(0, 3).toUpperCase()}</td>
                        {Object.keys(result.confusion_matrix[predicted])
                          .sort()
                          .map((actual) => {
                            const count = result.confusion_matrix[predicted][actual]
                            const isCorrect = predicted === actual
                            return (
                              <td
                                key={`${predicted}-${actual}`}
                                className={`matrix-cell ${isCorrect ? 'correct' : 'incorrect'}`}
                                title={`Predicted: ${predicted}, Actual: ${actual}`}
                              >
                                {count}
                              </td>
                            )
                          })}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="section top-errors-section">
            <h3>Top Errors</h3>
            <p className="description">Keywords with highest confidence predictions that were incorrect</p>
            {result.top_errors.length === 0 ? (
              <div className="no-errors">Perfect! No errors found.</div>
            ) : (
              <div className="errors-list">
                {result.top_errors.map((error, idx) => (
                  <div key={idx} className="error-item">
                    <div className="error-header">
                      <span className="error-number">#{idx + 1}</span>
                      <span className="error-keyword">{error.keyword}</span>
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
              <p>
                <strong>Evaluated:</strong> {new Date(result.timestamp).toLocaleString()}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
