import { useEffect, useState } from 'react'
import { api, apiError } from '../api'
import { useTask } from '../TaskContext'
import '../styles/TryIt.css'

type FeedbackType = 'correct' | 'incorrect' | 'unsure' | null

interface PredictionResult {
  predicted_label: string
  confidence: number
  probabilities: Record<string, number>
  model: string
  version: string
  golden_label: string | null
}

export default function TryIt() {
  const { task, color, refreshTasks } = useTask()
  const [text, setText] = useState('')
  const [version, setVersion] = useState('v1_baseline')
  const [versions, setVersions] = useState<string[]>(['v1_baseline'])
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [feedback, setFeedback] = useState<FeedbackType>(null)
  const [correctLabel, setCorrectLabel] = useState('')
  const [notes, setNotes] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  useEffect(() => {
    api
      .get<{ version: string }[]>('/models', { params: { task: task.id } })
      .then((res) => setVersions(res.data.map((m) => m.version)))
      .catch((err) => console.error('Failed to load model versions:', err))
  }, [task.id])

  const resetFeedback = () => {
    setFeedback(null)
    setCorrectLabel('')
    setNotes('')
  }

  const handleAnalyse = async () => {
    if (!text.trim()) {
      setError(`Please enter a ${task.item}`)
      return
    }

    setLoading(true)
    setError(null)
    setSuccess(null)
    setPrediction(null)
    resetFeedback()

    try {
      const response = await api.post<PredictionResult>('/predict', { task: task.id, text: text.trim(), model: 'laya', version })
      setPrediction(response.data)
    } catch (err) {
      setError(apiError(err, 'Failed to get prediction. Make sure the backend is running.'))
    } finally {
      setLoading(false)
    }
  }

  const submitFeedback = async () => {
    if (!prediction) return

    try {
      const res = await api.post<{ golden_status: 'created' | 'updated' | null }>('/feedback', {
        task: task.id,
        text: text.trim(),
        predicted_label: prediction.predicted_label,
        feedback_type: feedback,
        corrected_label: correctLabel || null,
        confidence: prediction.confidence,
        notes: notes || null,
        model: prediction.model,
        version: prediction.version,
      })
      const status = res.data.golden_status
      setSuccess(
        status === 'created'
          ? 'Feedback saved. This input was added to the golden dataset.'
          : status === 'updated'
            ? 'Feedback saved. The golden dataset row for this input was updated.'
            : 'Feedback saved.'
      )
      if (status) refreshTasks()
      setText('')
      setPrediction(null)
      resetFeedback()
    } catch (err) {
      setError(apiError(err, 'Failed to submit feedback'))
    }
  }

  const badge = (label: string) => (
    <span className="intent-badge" style={{ backgroundColor: `${color(label)}33`, color: color(label) }}>
      {label}
    </span>
  )

  return (
    <div className="serp-analysis">
      <div className="section">
        <h2>Try It: {task.name}</h2>
        <p className="description">
          Enter a {task.item} to classify it with Laya. Marking the prediction right or wrong adds the {task.item} to
          the golden dataset.
        </p>

        <div className="input-group">
          <textarea
            placeholder={`${task.input_label}...`}
            value={text}
            rows={task.id === 'search_intent' ? 1 : 3}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleAnalyse())}
            disabled={loading}
          />
          <select value={version} onChange={(e) => setVersion(e.target.value)} disabled={loading} title="Model version">
            {versions.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
          <button onClick={handleAnalyse} disabled={loading || !text.trim()} className="analyse-btn">
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">{success}</div>}

        {prediction && (
          <div className="prediction-result">
            <div className="prediction-card">
              <h3>Predicted Label</h3>
              <div className="intent-display">
                {badge(prediction.predicted_label)}
                <div className="confidence">Confidence: {(prediction.confidence * 100).toFixed(1)}%</div>
              </div>
              <p className="intent-description">{task.label_descriptions[prediction.predicted_label]}</p>
              {Object.keys(prediction.probabilities).length > 0 && (
                <p className="intent-description">
                  {task.labels.map((label) => `${label} ${((prediction.probabilities[label] ?? 0) * 100).toFixed(0)}%`).join(' · ')}
                </p>
              )}
              {prediction.golden_label && (
                <p className="intent-description">
                  This {task.item} is already in the golden dataset, labelled {badge(prediction.golden_label)}
                </p>
              )}
            </div>

            {!feedback ? (
              <div className="feedback-section">
                <h3>Is this prediction correct?</h3>
                <div className="feedback-buttons">
                  <button onClick={() => setFeedback('correct')} className="feedback-btn feedback-correct">
                    ✓ Correct
                  </button>
                  <button onClick={() => setFeedback('incorrect')} className="feedback-btn feedback-incorrect">
                    ✗ Incorrect
                  </button>
                  <button onClick={() => setFeedback('unsure')} className="feedback-btn feedback-unsure">
                    ? Unsure
                  </button>
                </div>
              </div>
            ) : (
              <div className="feedback-details">
                <p className="feedback-status">
                  You marked this as: <strong>{feedback.toUpperCase()}</strong>
                </p>

                {(feedback === 'incorrect' || feedback === 'unsure') && (
                  <div className="correct-intent-section">
                    <label htmlFor="correct-label">What should the label be?</label>
                    <select id="correct-label" value={correctLabel} onChange={(e) => setCorrectLabel(e.target.value)}>
                      <option value="">Select the correct label...</option>
                      {task.labels.map((label) => (
                        <option key={label} value={label}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <div className="notes-section">
                  <label htmlFor="notes">Additional notes (optional):</label>
                  <textarea
                    id="notes"
                    placeholder="Share any additional context or reasoning..."
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    rows={3}
                  />
                </div>

                <div className="submit-feedback-buttons">
                  <button onClick={resetFeedback} className="cancel-btn">
                    Cancel
                  </button>
                  <button
                    onClick={submitFeedback}
                    disabled={feedback === 'incorrect' && !correctLabel}
                    className="submit-btn"
                  >
                    Submit Feedback
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="section info-section">
        <h3>Labels</h3>
        <div className="intent-grid">
          {task.labels.map((label) => (
            <div key={label} className="intent-info">
              <strong style={{ color: color(label) }}>{label}</strong>
              <p>{task.label_descriptions[label]}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
