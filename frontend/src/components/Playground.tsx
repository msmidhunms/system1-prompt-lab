import { useEffect, useState } from 'react'
import { api, apiError } from '../api'
import { useApp, useTask } from '../context'
import { Banner, Button, Card, Field, LabelBadge } from '../ui'

type Feedback = 'correct' | 'incorrect' | 'unsure' | null

interface Prediction {
  text: string
  predicted_label: string
  confidence: number
  probabilities: Record<string, number>
  model: string
  version: string
  golden_label: string | null
}

export default function Playground() {
  const { task, color, versions, intent, clearIntent } = useTask()
  const { refreshTasks, engines, engineName } = useApp()
  // '' means the model the version was saved for.
  const [engine, setEngine] = useState('')
  const [text, setText] = useState('')
  const [version, setVersion] = useState('v1_baseline')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [feedback, setFeedback] = useState<Feedback>(null)
  const [correctLabel, setCorrectLabel] = useState('')
  const [notes, setNotes] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  // Another tab asked to try a particular version.
  useEffect(() => {
    if (intent?.tab === 'playground' && intent.version) {
      setVersion(intent.version)
      clearIntent()
    }
  }, [intent, clearIntent])

  const classify = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setNotice(null)
    setPrediction(null)
    setFeedback(null)
    setCorrectLabel('')
    setNotes('')
    try {
      const res = await api.post<Prediction>('/predict', { task: task.id, text: text.trim(), version, engine: engine || undefined })
      setPrediction(res.data)
    } catch (err) {
      setError(apiError(err, 'Failed to classify'))
    } finally {
      setLoading(false)
    }
  }

  const submitFeedback = async () => {
    if (!prediction || !feedback) return
    try {
      const res = await api.post<{ golden_status: 'created' | 'updated' | null }>('/feedback', {
        task: task.id,
        text: prediction.text,
        predicted_label: prediction.predicted_label,
        feedback_type: feedback,
        corrected_label: correctLabel || null,
        confidence: prediction.confidence,
        notes: notes || null,
        model: prediction.model,
        version: prediction.version,
      })
      const status = res.data.golden_status
      setNotice(
        status === 'created'
          ? `Feedback saved. This ${task.item} was added to the golden dataset.`
          : status === 'updated'
            ? `Feedback saved. The golden dataset row for this ${task.item} was updated.`
            : 'Feedback saved.'
      )
      if (status) refreshTasks()
      setText('')
      setPrediction(null)
      setFeedback(null)
    } catch (err) {
      setError(apiError(err, 'Failed to save the feedback'))
    }
  }

  return (
    <div className="page">
      <Card title="Classify one input" sub={`Run a model version on a ${task.item}. Marking the result right or wrong adds the ${task.item} to the golden dataset.`}>
        <div className="stack">
          <Field label={task.input_label}>
            <textarea
              className="input"
              rows={task.id === 'search_intent' ? 1 : 4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), classify())}
              disabled={loading}
            />
          </Field>
          <div className="row end">
            <Field label="Model version">
              <select className="input" value={version} onChange={(e) => setVersion(e.target.value)} disabled={loading}>
                {versions.map((v) => (
                  <option key={v.version} value={v.version}>{v.version}</option>
                ))}
              </select>
            </Field>
            <Field label="Model">
              <select className="input" value={engine} onChange={(e) => setEngine(e.target.value)} disabled={loading}>
                <option value="">The version's own</option>
                {engines.map((e) => (
                  <option key={e.id} value={e.id}>{e.name}</option>
                ))}
              </select>
            </Field>
            <Button variant="primary" onClick={classify} disabled={loading || !text.trim()}>
              {loading ? 'Classifying…' : 'Classify'}
            </Button>
          </div>
          {error && <Banner tone="error">{error}</Banner>}
          {notice && <Banner tone="success">{notice}</Banner>}
        </div>
      </Card>

      {prediction && (
        <Card title="Prediction">
          <div className="stack">
            <div className="row">
              <LabelBadge label={prediction.predicted_label} color={color(prediction.predicted_label)} />
              <span className="secondary">{(prediction.confidence * 100).toFixed(1)}% confidence · {engineName(prediction.model)}</span>
              {prediction.golden_label && (
                <span className="secondary">
                  · already in the golden dataset as <LabelBadge label={prediction.golden_label} color={color(prediction.golden_label)} />
                </span>
              )}
            </div>
            {task.labels.map((label) => {
              const p = prediction.probabilities[label] ?? 0
              return (
                <div key={label} className="meter">
                  <span style={{ width: 150 }} className="small">{label}</span>
                  <div className="track">
                    <div style={{ width: `${p * 100}%`, background: color(label) }} />
                  </div>
                  <span className="small num secondary" style={{ width: 44, textAlign: 'right' }}>{(p * 100).toFixed(0)}%</span>
                </div>
              )
            })}

            <h4>Is this correct?</h4>
            <div className="row">
              {(['correct', 'incorrect', 'unsure'] as const).map((kind) => (
                <Button key={kind} variant={feedback === kind ? 'primary' : 'default'} onClick={() => setFeedback(kind)}>
                  {kind === 'correct' ? 'Correct' : kind === 'incorrect' ? 'Incorrect' : 'Not sure'}
                </Button>
              ))}
            </div>
            {feedback && (
              <div className="stack">
                {feedback !== 'correct' && (
                  <Field label="What should the label be?">
                    <select className="input" value={correctLabel} onChange={(e) => setCorrectLabel(e.target.value)}>
                      <option value="">Select a label…</option>
                      {task.labels.map((label) => (
                        <option key={label} value={label}>{label}</option>
                      ))}
                    </select>
                  </Field>
                )}
                <Field label="Notes (optional)">
                  <textarea className="input" rows={2} value={notes} onChange={(e) => setNotes(e.target.value)} />
                </Field>
                <div className="row">
                  <Button variant="primary" onClick={submitFeedback} disabled={feedback === 'incorrect' && !correctLabel}>
                    Save feedback
                  </Button>
                  <Button onClick={() => setFeedback(null)}>Cancel</Button>
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      <Card title="Labels" sub="What each label means in this example's default prompt.">
        <dl className="config">
          {task.labels.map((label) => (
            <div key={label} style={{ display: 'contents' }}>
              <dt><LabelBadge label={label} color={color(label)} /></dt>
              <dd>{task.label_descriptions[label]}</dd>
            </div>
          ))}
        </dl>
      </Card>
    </div>
  )
}
