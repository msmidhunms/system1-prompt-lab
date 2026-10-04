import { useState } from 'react'
import { api, apiError, ModelVersion, PromptConfig, stateText, TaskInfo } from '../api'
import { Banner, Button, Dialog, Field } from '../ui'

const LAYA_MODELS = ['auto', 'english', 'multilingual', 'typed-decisions']
const words = (text: string) => text.trim().split(/\s+/).filter(Boolean).length

// A prompt, read-only. With `previous`, the parts that differ from it are marked.
export function ConfigView({ config, previous, labels }: { config: PromptConfig; previous?: PromptConfig; labels: string[] }) {
  const changed = (a: string, b: string | undefined) => (previous !== undefined && a !== b ? 'changed' : '')
  const state = stateText(config.state_template)
  return (
    <dl className="config">
      <dt className={changed(state, previous && stateText(previous.state_template))}>State template</dt>
      <dd className="mono">
        {state}
        {config.serp_results != null && ` (top ${config.serp_results} search results)`}
      </dd>
      <dt className={changed(config.instructions, previous?.instructions)}>Instructions</dt>
      <dd>{config.instructions}</dd>
      {labels.map((label) => (
        <FragmentRow key={label} term={label} className={changed(config.criteria[label], previous?.criteria[label])}>
          {config.criteria[label]}
        </FragmentRow>
      ))}
      {config.model && <FragmentRow term="Laya checkpoint">{config.model}</FragmentRow>}
      {config.label_bias && (
        <FragmentRow term="Label bias">
          {labels.map((label) => `${label} ${(config.label_bias?.[label] ?? 0).toFixed(1)}`).join(' · ')}
        </FragmentRow>
      )}
    </dl>
  )
}

function FragmentRow({ term, className = '', children }: { term: string; className?: string; children: React.ReactNode }) {
  return (
    <>
      <dt className={className}>{term}</dt>
      <dd>{children}</dd>
    </>
  )
}

// Write a new model version, starting from a copy of an existing prompt.
export function PromptEditor({ task, from, onSaved, onClose }: {
  task: TaskInfo
  from: { name: string; config: PromptConfig; base: string | null }
  onSaved: (saved: ModelVersion) => void
  onClose: () => void
}) {
  const [name, setName] = useState(from.name)
  const [description, setDescription] = useState('')
  const [state, setState] = useState(stateText(from.config.state_template))
  const [instructions, setInstructions] = useState(from.config.instructions)
  const [criteria, setCriteria] = useState<Record<string, string>>({ ...from.config.criteria })
  const [model, setModel] = useState(from.config.model ?? 'auto')
  const [bias, setBias] = useState<Record<string, string>>(
    Object.fromEntries(task.labels.map((label) => [label, String(from.config.label_bias?.[label] ?? 0)]))
  )
  const [error, setError] = useState<string | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])
  const [busy, setBusy] = useState(false)

  const save = async () => {
    setError(null)
    let stateTemplate: PromptConfig['state_template'] = state
    if (state.trim().startsWith('{')) {
      try {
        const parsed = JSON.parse(state)
        // "{query}" alone is a plain-text template, not JSON; anything that parses as an object is a JSON state.
        if (parsed && typeof parsed === 'object') stateTemplate = parsed
      } catch {
        if (state.trim() !== `{${task.input_field}}`) {
          setError('The state template starts with "{" but is not valid JSON.')
          return
        }
      }
    }
    const labelBias = Object.fromEntries(task.labels.map((label) => [label, Number(bias[label]) || 0]))
    const config: PromptConfig = {
      state_template: stateTemplate,
      instructions,
      criteria,
      model,
      ...(from.config.serp_results != null ? { serp_results: from.config.serp_results } : {}),
      ...(Object.values(labelBias).some((v) => v !== 0) ? { label_bias: labelBias } : {}),
    }
    setBusy(true)
    try {
      const res = await api.post<ModelVersion & { warnings: string[] }>('/models', {
        task: task.id,
        name: name.trim(),
        config,
        base_version: from.base,
        description: description.trim() || null,
      })
      if (res.data.warnings.length > 0) {
        // Saved, but worth showing before the dialog goes away.
        setWarnings(res.data.warnings)
        onSaved(res.data)
        return
      }
      onSaved(res.data)
      onClose()
    } catch (err) {
      setError(apiError(err, 'Failed to save the model'))
    } finally {
      setBusy(false)
    }
  }

  const saved = warnings.length > 0
  const instructionWords = words(instructions)

  return (
    <Dialog
      title="New model version"
      wide
      onClose={onClose}
      footer={
        saved ? (
          <Button variant="primary" onClick={onClose}>Done</Button>
        ) : (
          <>
            <Button onClick={onClose}>Cancel</Button>
            <Button variant="primary" onClick={save} disabled={busy || !name.trim()}>
              {busy ? 'Saving…' : 'Save as new version'}
            </Button>
          </>
        )
      }
    >
      <div className="stack">
        <p className="secondary small">
          {from.base ? `A copy of ${from.base}. ` : ''}Saving creates a new version; the one it was copied from is not changed.
        </p>
        <div className="grid-2">
          <Field label="Name" hint="Letters, digits, '_', '-' and '.'. Unique across examples.">
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} disabled={saved} />
          </Field>
          <Field label="Description (optional)">
            <input className="input" value={description} onChange={(e) => setDescription(e.target.value)} disabled={saved} />
          </Field>
        </div>
        <Field label="State template" hint={`How the ${task.item} is given to Laya. Must contain {${task.input_field}} exactly once. A JSON object becomes a JSON state.`}>
          <input className="input mono" value={state} onChange={(e) => setState(e.target.value)} disabled={saved} />
        </Field>
        <Field label="Instructions" hint={`${instructionWords} / ${task.max_instruction_words} words`} over={instructionWords > task.max_instruction_words}>
          <textarea className="input" rows={2} value={instructions} onChange={(e) => setInstructions(e.target.value)} disabled={saved} />
        </Field>
        {task.labels.map((label) => {
          const count = words(criteria[label] ?? '')
          return (
            <Field key={label} label={`Description of “${label}”`} hint={`${count} / ${task.max_criterion_words} words`} over={count > task.max_criterion_words}>
              <textarea className="input" rows={2} value={criteria[label] ?? ''} onChange={(e) => setCriteria({ ...criteria, [label]: e.target.value })} disabled={saved} />
            </Field>
          )
        })}
        <div className="row end">
          <Field label="Laya checkpoint">
            <select className="input" value={model} onChange={(e) => setModel(e.target.value)} disabled={saved}>
              {LAYA_MODELS.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </Field>
          {task.labels.map((label) => (
            <Field key={label} label={`Bias: ${label}`}>
              <input className="input narrow num" type="number" step="0.1" value={bias[label]} onChange={(e) => setBias({ ...bias, [label]: e.target.value })} disabled={saved} />
            </Field>
          ))}
        </div>
        <p className="small muted">Label bias is added to a label's log-probability before the highest one wins. Leave at 0 unless you know you need it.</p>
        {error && <Banner tone="error">{error}</Banner>}
        {saved && (
          <Banner tone="warning">
            Saved as {name.trim()}, with warnings: {warnings.join('; ')}.
          </Banner>
        )}
      </div>
    </Dialog>
  )
}
