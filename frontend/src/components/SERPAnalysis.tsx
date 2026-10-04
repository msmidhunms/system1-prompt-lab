import { useState } from 'react'
import axios from 'axios'
import '../styles/SERPAnalysis.css'

type FeedbackType = 'correct' | 'incorrect' | 'unsure' | null
type IntentType = 'informational' | 'navigational' | 'commercial' | 'transactional'

interface PredictionResult {
  predicted_intent: IntentType
  confidence: number
  model: string
  version: string
}

const INTENT_LABELS: Record<IntentType, string> = {
  informational: 'Informational',
  navigational: 'Navigational',
  commercial: 'Commercial',
  transactional: 'Transactional',
}

const INTENT_DESCRIPTIONS: Record<IntentType, string> = {
  informational: 'User seeks information, education, or knowledge (e.g., "how to", "what is")',
  navigational: 'User seeks to reach a specific website or resource (e.g., "facebook login")',
  commercial: 'User researches products/services before purchasing (e.g., "best laptop")',
  transactional: 'User intends to complete a transaction (e.g., "buy laptop")',
}

export default function SERPAnalysis() {
  const [query, setQuery] = useState('')
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [feedback, setFeedback] = useState<FeedbackType>(null)
  const [selectedCorrectIntent, setSelectedCorrectIntent] = useState<IntentType | null>(null)
  const [notes, setNotes] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const handleAnalyse = async () => {
    if (!query.trim()) {
      setError('Please enter a search query')
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)
    setFeedback(null)
    setSelectedCorrectIntent(null)
    setNotes('')

    try {
      // TODO: Replace with actual API endpoint
      const response = await axios.post('http://localhost:8000/api/predict', {
        query: query.trim(),
        model: 'laya',
        version: 'v1_baseline',
      })

      setPrediction(response.data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to get prediction. Make sure the backend is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  const handleFeedback = async (feedbackType: FeedbackType) => {
    if (!prediction) return

    setFeedback(feedbackType)

    // Reset for new feedback submission
    if (feedbackType === 'incorrect' || feedbackType === 'unsure') {
      // User needs to select correct intent
      return
    }
  }

  const submitFeedback = async () => {
    if (!prediction) return

    try {
      const feedbackData = {
        query: query.trim(),
        predicted_intent: prediction.predicted_intent,
        feedback_type: feedback,
        corrected_intent: selectedCorrectIntent,
        confidence: prediction.confidence,
        notes: notes || null,
        model: prediction.model,
        version: prediction.version,
      }

      // TODO: Replace with actual API endpoint
      await axios.post('http://localhost:8000/api/feedback', feedbackData)

      setSuccess(true)
      setTimeout(() => {
        setQuery('')
        setPrediction(null)
        setFeedback(null)
        setSelectedCorrectIntent(null)
        setNotes('')
        setSuccess(false)
      }, 2000)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to submit feedback'
      )
    }
  }

  return (
    <div className="serp-analysis">
      <div className="section">
        <h2>Analyze Search Intent</h2>
        <p className="description">
          Enter a search query to analyze its intent using the System 1 model
        </p>

        <div className="input-group">
          <input
            type="text"
            placeholder="Enter a search query (e.g., 'how to bake a cake')"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAnalyse()}
            disabled={loading}
          />
          <button
            onClick={handleAnalyse}
            disabled={loading || !query.trim()}
            className="analyse-btn"
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">Feedback submitted successfully!</div>}

        {prediction && (
          <div className="prediction-result">
            <div className="prediction-card">
              <h3>Predicted Intent</h3>
              <div className="intent-display">
                <span className={`intent-badge intent-${prediction.predicted_intent}`}>
                  {INTENT_LABELS[prediction.predicted_intent]}
                </span>
                <div className="confidence">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <p className="intent-description">
                {INTENT_DESCRIPTIONS[prediction.predicted_intent]}
              </p>
            </div>

            {!feedback ? (
              <div className="feedback-section">
                <h3>Is this prediction correct?</h3>
                <div className="feedback-buttons">
                  <button
                    onClick={() => handleFeedback('correct')}
                    className="feedback-btn feedback-correct"
                  >
                    ✓ Correct
                  </button>
                  <button
                    onClick={() => handleFeedback('incorrect')}
                    className="feedback-btn feedback-incorrect"
                  >
                    ✗ Incorrect
                  </button>
                  <button
                    onClick={() => handleFeedback('unsure')}
                    className="feedback-btn feedback-unsure"
                  >
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
                    <label htmlFor="correct-intent">What should be the correct intent?</label>
                    <select
                      id="correct-intent"
                      value={selectedCorrectIntent || ''}
                      onChange={(e) => setSelectedCorrectIntent(e.target.value as IntentType)}
                    >
                      <option value="">Select correct intent...</option>
                      {Object.entries(INTENT_LABELS).map(([key, label]) => (
                        <option key={key} value={key}>
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
                  <button
                    onClick={() => {
                      setFeedback(null)
                      setSelectedCorrectIntent(null)
                      setNotes('')
                    }}
                    className="cancel-btn"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={submitFeedback}
                    disabled={
                      (feedback === 'incorrect' || feedback === 'unsure') &&
                      !selectedCorrectIntent
                    }
                    className="submit-btn"
                  >
                    Submit Feedback
                  </button>
                </div>
              </div>
            )}

            {feedback === 'correct' && (
              <button onClick={submitFeedback} className="submit-btn submit-correct">
                Save & Continue
              </button>
            )}
          </div>
        )}
      </div>

      <div className="section info-section">
        <h3>Intent Types</h3>
        <div className="intent-grid">
          {Object.entries(INTENT_LABELS).map(([key, label]) => (
            <div key={key} className="intent-info">
              <strong>{label}</strong>
              <p>{INTENT_DESCRIPTIONS[key as IntentType]}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
