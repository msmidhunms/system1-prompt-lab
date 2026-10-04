import { useState, useEffect } from 'react'
import axios from 'axios'
import '../styles/KarpathyLoop.css'

interface GoldenDataPoint {
  id: string
  query: string
  correct_intent: string
  source: string
  feedback_count: number
}

interface GoldenDataStats {
  imported_data: number
  user_feedback_count: number
  unique_feedback_queries: number
  total_golden_data: number
}

interface LoopIteration {
  iteration: number
  model_version: string
  accuracy: number
  improvements: string[]
  timestamp: string
}

interface ExperimentResult {
  baseline_accuracy: number
  final_accuracy: number
  best_iteration: number
  best_model_name: string
  iterations: LoopIteration[]
  improved: boolean
}

export default function KarpathyLoop() {
  const [goldenData, setGoldenData] = useState<GoldenDataPoint[]>([])
  const [stats, setStats] = useState<GoldenDataStats>({
    imported_data: 0,
    user_feedback_count: 0,
    unique_feedback_queries: 0,
    total_golden_data: 0,
  })
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<ExperimentResult | null>(null)
  const [newModelName, setNewModelName] = useState('')
  const [showNameModel, setShowNameModel] = useState(false)
  const [sampleSize, setSampleSize] = useState(100)
  const [numLoops, setNumLoops] = useState(10)

  useEffect(() => {
    loadGoldenData()
    loadStats()
  }, [])

  const loadStats = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/golden-data/stats')
      setStats(response.data)
    } catch (err) {
      console.error('Failed to load stats:', err)
    }
  }

  const loadGoldenData = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get('http://localhost:8000/api/golden-data')
      setGoldenData(response.data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to load golden dataset'
      )
    } finally {
      setLoading(false)
    }
  }

  const startKarpathyLoop = async () => {
    if (goldenData.length === 0) {
      setError('No data in golden dataset. Please add some feedback first.')
      return
    }

    setRunning(true)
    setError(null)
    setResults(null)

    try {
      // TODO: Replace with actual API endpoint
      const response = await axios.post('http://localhost:8000/api/karpathy-loop', {
        golden_data_ids: goldenData.map((d) => d.id),
        loops: 10,
        baseline_model: 'laya',
        baseline_version: 'v1_baseline',
      })

      setResults(response.data)
      if (response.data.improved) {
        setShowNameModel(true)
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to run Karpathy loop'
      )
    } finally {
      setRunning(false)
    }
  }

  const saveImprovedModel = async () => {
    if (!results || !newModelName.trim()) {
      setError('Please enter a model name')
      return
    }

    try {
      // TODO: Replace with actual API endpoint
      await axios.post('http://localhost:8000/api/save-model', {
        model_name: newModelName.trim(),
        base_version: results.best_model_name,
        accuracy: results.final_accuracy,
        description: `Improved model from Karpathy loop - Best iteration: ${results.best_iteration}`,
      })

      setShowNameModel(false)
      setNewModelName('')
      // Optionally reload or show success message
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to save model'
      )
    }
  }

  return (
    <div className="karpathy-loop">
      <div className="section">
        <h2>Karpathy Loop - Model Improvement</h2>
        <p className="description">
          Run iterative experiments to improve the model based on the golden dataset.
          This process will run 10 loops, improving the model and recording metrics at each step.
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

        {stats.total_golden_data > 0 && (
          <button
            onClick={startKarpathyLoop}
            disabled={running}
            className="start-loop-btn"
          >
            {running ? 'Running Experiment...' : 'Start Karpathy Loop (10 Iterations)'}
          </button>
        )}

        {error && <div className="error-message">{error}</div>}
      </div>

      {running && (
        <div className="section running-section">
          <h3>Experiment in Progress...</h3>
          <div className="progress-indicator">
            <div className="spinner"></div>
            <p>Running 10 iterations to find the best model improvement...</p>
          </div>
        </div>
      )}

      {results && (
        <div className="section results-section">
          <h3>Experiment Results</h3>

          <div className="results-summary">
            <div className="result-card">
              <label>Baseline Accuracy</label>
              <div className="metric-value">{(results.baseline_accuracy * 100).toFixed(2)}%</div>
            </div>
            <div className="result-card">
              <label>Final Accuracy</label>
              <div className="metric-value">{(results.final_accuracy * 100).toFixed(2)}%</div>
            </div>
            <div className="result-card">
              <label>Improvement</label>
              <div className={`metric-value ${results.improved ? 'positive' : 'neutral'}`}>
                {((results.final_accuracy - results.baseline_accuracy) * 100).toFixed(2)}%
              </div>
            </div>
            <div className="result-card">
              <label>Best Iteration</label>
              <div className="metric-value">{results.best_iteration}/10</div>
            </div>
          </div>

          <div className="iterations-list">
            <h4>Iteration Details</h4>
            <div className="iterations-grid">
              {results.iterations.map((iter) => (
                <div key={iter.iteration} className="iteration-card">
                  <div className="iteration-header">
                    <span className="iteration-number">Loop {iter.iteration}</span>
                    {iter.iteration === results.best_iteration && (
                      <span className="best-badge">Best</span>
                    )}
                  </div>
                  <div className="iteration-details">
                    <p>
                      <strong>Model:</strong> {iter.model_version}
                    </p>
                    <p>
                      <strong>Accuracy:</strong> {(iter.accuracy * 100).toFixed(2)}%
                    </p>
                    {iter.improvements.length > 0 && (
                      <div className="improvements">
                        <strong>Improvements:</strong>
                        <ul>
                          {iter.improvements.map((imp, idx) => (
                            <li key={idx}>{imp}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {results.improved && (
            <div className="section model-saving-section">
              <h3>Save Improved Model</h3>
              <p>The experiment found an improved model! Name it for future reference.</p>

              {!showNameModel ? (
                <button onClick={() => setShowNameModel(true)} className="save-model-btn">
                  Save as New Model
                </button>
              ) : (
                <div className="name-model-form">
                  <input
                    type="text"
                    placeholder="e.g., laya_v2_improved"
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && saveImprovedModel()}
                  />
                  <button onClick={saveImprovedModel} disabled={!newModelName.trim()}>
                    Save Model
                  </button>
                  <button
                    onClick={() => {
                      setShowNameModel(false)
                      setNewModelName('')
                    }}
                    className="cancel-btn"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          )}

          {!results.improved && (
            <div className="section no-improvement">
              <p>No improvement found in this run. The baseline model is already optimal.</p>
            </div>
          )}
        </div>
      )}

      <div className="section info-section">
        <h3>How Karpathy Loop Works</h3>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <div>
              <strong>Initialize</strong>
              <p>Start with baseline model and golden dataset</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div>
              <strong>Improve Input</strong>
              <p>Analyze and improve the input features and representations</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div>
              <strong>Evaluate</strong>
              <p>Evaluate against the golden dataset to measure accuracy</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">4</div>
            <div>
              <strong>Iterate</strong>
              <p>Repeat for 10 loops, recording metrics and improvements</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">5</div>
            <div>
              <strong>Save</strong>
              <p>If improved, save the best model with a new name and score</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
