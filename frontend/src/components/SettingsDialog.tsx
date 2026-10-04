import { useState } from 'react'
import { api, apiError } from '../api'
import { LLMConfig, useApp } from '../context'
import { Banner, Button, Dialog, Field } from '../ui'

// The LLM that proposes prompts in the optimizer. One setting for the whole app, stored on the server.
export default function SettingsDialog() {
  const { llm, setLLM, closeSettings } = useApp()
  const [provider, setProvider] = useState(llm?.active ?? '')
  const [model, setModel] = useState(llm ? llm.providers[llm.active].model : '')
  const [baseUrl, setBaseUrl] = useState(llm ? llm.providers[llm.active].base_url : '')
  const [apiKey, setApiKey] = useState('')
  const [replacingKey, setReplacingKey] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null)

  if (!llm) {
    return (
      <Dialog title="LLM settings" onClose={closeSettings}>
        <p className="secondary">Loading…</p>
      </Dialog>
    )
  }

  const info = llm.providers[provider]
  const locked = busy !== null

  const selectProvider = (name: string) => {
    const next = llm.providers[name]
    setProvider(name)
    setModel(next.model)
    setBaseUrl(next.base_url)
    setApiKey('')
    setReplacingKey(false)
    setMessage(null)
  }

  const save = async (): Promise<boolean> => {
    setBusy('save')
    setMessage(null)
    try {
      const res = await api.put<LLMConfig>('/llm/config', { provider, model, base_url: baseUrl, api_key: apiKey || null })
      setLLM(res.data)
      setApiKey('')
      setReplacingKey(false)
      setMessage({ ok: true, text: `Saved. The optimizer will use ${res.data.providers[provider].label}.` })
      return true
    } catch (err) {
      setMessage({ ok: false, text: apiError(err, 'Failed to save the settings') })
      return false
    } finally {
      setBusy(null)
    }
  }

  const removeKey = async () => {
    setBusy('save')
    try {
      const res = await api.put<LLMConfig>('/llm/config', { provider, clear_api_key: true })
      setLLM(res.data)
      setMessage({ ok: true, text: 'Stored API key removed.' })
    } catch (err) {
      setMessage({ ok: false, text: apiError(err, 'Failed to remove the API key') })
    } finally {
      setBusy(null)
    }
  }

  const test = async () => {
    if (!(await save())) return
    setBusy('test')
    try {
      const res = await api.post('/llm/test', { provider })
      setMessage(res.data.ok ? { ok: true, text: `Connection works. Reply: "${res.data.reply}"` } : { ok: false, text: res.data.error })
    } catch (err) {
      setMessage({ ok: false, text: apiError(err, 'The test failed') })
    } finally {
      setBusy(null)
    }
  }

  const loadModels = async () => {
    if (!(await save())) return
    setBusy('models')
    try {
      const res = await api.post<{ models: string[] }>('/llm/models', { provider })
      const fresh = await api.get<LLMConfig>('/llm/config')
      setLLM(fresh.data)
      setMessage({ ok: true, text: `${res.data.models.length} models available in the model field.` })
    } catch (err) {
      setMessage({ ok: false, text: apiError(err, 'Failed to list the models') })
    } finally {
      setBusy(null)
    }
  }

  const showKeyInput = info.kind !== 'cli' && (replacingKey || info.key_source !== 'stored')

  return (
    <Dialog
      title="LLM settings"
      onClose={closeSettings}
      footer={
        <>
          {info.kind !== 'cli' && (
            <Button onClick={loadModels} disabled={locked}>
              {busy === 'models' ? 'Loading…' : 'Load model list'}
            </Button>
          )}
          <Button onClick={test} disabled={locked}>
            {busy === 'test' ? 'Testing…' : 'Save and test'}
          </Button>
          <Button variant="primary" onClick={save} disabled={locked}>
            {busy === 'save' ? 'Saving…' : 'Save'}
          </Button>
        </>
      }
    >
      <div className="stack">
        <p className="secondary">
          The LLM that proposes new prompts in the Prompt Optimizer. This is one setting for every example, kept on the
          server. Active now: <strong>{llm.providers[llm.active].label}</strong>
          {llm.providers[llm.active].model && ` · ${llm.providers[llm.active].model}`}.
        </p>
        <Field label="Provider">
          <select className="input" value={provider} onChange={(e) => selectProvider(e.target.value)} disabled={locked}>
            {Object.entries(llm.providers).map(([name, p]) => (
              <option key={name} value={name}>
                {p.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Model">
          <input
            className="input"
            list="llm-model-options"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder={info.kind === 'cli' ? 'CLI default' : 'Model id'}
            disabled={locked}
          />
          <datalist id="llm-model-options">
            {info.models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </Field>
        {info.kind === 'openai_compat' && (
          <Field label="Base URL">
            <input className="input" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder={info.default_base_url || 'https://host/v1'} disabled={locked} />
          </Field>
        )}
        {info.kind !== 'cli' && (
          <Field
            label="API key"
            hint={
              info.key_source === 'env'
                ? `Using ${info.key_env} from the server environment. A key entered here takes its place.`
                : !info.needs_key
                  ? 'Optional for this provider.'
                  : undefined
            }
          >
            {showKeyInput ? (
              <input className="input" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Paste API key" autoComplete="off" disabled={locked} />
            ) : (
              <div className="row">
                <span className="badge good">Key saved</span>
                <Button small onClick={() => setReplacingKey(true)} disabled={locked}>
                  Replace
                </Button>
                <Button small variant="danger" onClick={removeKey} disabled={locked}>
                  Remove
                </Button>
              </div>
            )}
          </Field>
        )}
        {info.kind === 'cli' && (
          <p className="small secondary">
            {info.cli_found
              ? 'Found on the server. It uses the login of the account running the backend.'
              : 'Not found on the server PATH. Install it and log in before using this provider.'}
          </p>
        )}
        {message && <Banner tone={message.ok ? 'success' : 'error'}>{message.text}</Banner>}
      </div>
    </Dialog>
  )
}
