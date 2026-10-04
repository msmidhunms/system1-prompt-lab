import { useState, useEffect } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000/api'

interface ProviderInfo {
  label: string
  kind: 'anthropic' | 'openai_compat' | 'cli'
  model: string
  base_url: string
  default_base_url: string
  models: string[]
  needs_key: boolean
  key_env: string | null
  api_key_set: boolean
  key_source: 'stored' | 'env' | null
  cli_found: boolean | null
}

interface LLMConfig {
  active: string
  providers: Record<string, ProviderInfo>
}

interface Props {
  disabled?: boolean
  onActiveChange?: (label: string) => void
}

export function apiError(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    return err.message
  }
  return err instanceof Error ? err.message : fallback
}

export default function LLMSettings({ disabled = false, onActiveChange }: Props) {
  const [config, setConfig] = useState<LLMConfig | null>(null)
  const [provider, setProvider] = useState('')
  const [model, setModel] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [models, setModels] = useState<string[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null)

  useEffect(() => {
    axios
      .get<LLMConfig>(`${API}/llm/config`)
      .then((res) => applyConfig(res.data, res.data.active))
      .catch((err) => setMessage({ ok: false, text: apiError(err, 'Failed to load LLM settings') }))
  }, [])

  const applyConfig = (cfg: LLMConfig, name: string) => {
    const info = cfg.providers[name]
    setConfig(cfg)
    setProvider(name)
    setModel(info.model)
    setBaseUrl(info.base_url)
    setApiKey('')
    setModels(info.models)
    const active = cfg.providers[cfg.active]
    onActiveChange?.(`${active.label}${active.model ? ` · ${active.model}` : ''}`)
  }

  const selectProvider = (name: string) => {
    if (!config) return
    setMessage(null)
    applyConfig(config, name)
  }

  const save = async (): Promise<boolean> => {
    setBusy('save')
    setMessage(null)
    try {
      const res = await axios.put<LLMConfig>(`${API}/llm/config`, {
        provider,
        model,
        base_url: baseUrl,
        api_key: apiKey || null,
      })
      applyConfig(res.data, provider)
      setMessage({ ok: true, text: 'Saved. The loop will use this LLM.' })
      return true
    } catch (err) {
      setMessage({ ok: false, text: apiError(err, 'Failed to save LLM settings') })
      return false
    } finally {
      setBusy(null)
    }
  }

  const clearKey = async () => {
    setBusy('save')
    try {
      const res = await axios.put<LLMConfig>(`${API}/llm/config`, { provider, clear_api_key: true })
      applyConfig(res.data, provider)
      setMessage({ ok: true, text: 'Stored API key removed.' })
    } catch (err) {
      setMessage({ ok: false, text: apiError(err, 'Failed to remove API key') })
    } finally {
      setBusy(null)
    }
  }

  const test = async () => {
    if (!(await save())) return
    setBusy('test')
    setMessage(null)
    try {
      const res = await axios.post(`${API}/llm/test`, { provider })
      setMessage(
        res.data.ok
          ? { ok: true, text: `Connection works. Reply: "${res.data.reply}"` }
          : { ok: false, text: res.data.error }
      )
    } catch (err) {
      setMessage({ ok: false, text: apiError(err, 'LLM test failed') })
    } finally {
      setBusy(null)
    }
  }

  const loadModels = async () => {
    if (!(await save())) return
    setBusy('models')
    setMessage(null)
    try {
      const res = await axios.post<{ models: string[] }>(`${API}/llm/models`, { provider })
      setModels(res.data.models)
      setMessage({ ok: true, text: `${res.data.models.length} models available. Pick one from the model field.` })
    } catch (err) {
      setMessage({ ok: false, text: apiError(err, 'Failed to list models') })
    } finally {
      setBusy(null)
    }
  }

  if (!config) {
    return (
      <div className="llm-settings">
        <h3>LLM that rewrites the prompt</h3>
        {message ? <div className="error-message">{message.text}</div> : <p className="empty-message">Loading...</p>}
      </div>
    )
  }

  const info = config.providers[provider]
  const locked = disabled || busy !== null
  const keyPlaceholder =
    info.key_source === 'stored'
      ? 'Stored (leave blank to keep)'
      : info.key_source === 'env'
        ? `Using ${info.key_env} from the server environment`
        : info.needs_key
          ? 'Paste API key'
          : 'Optional'

  return (
    <div className="llm-settings">
      <h3>LLM that rewrites the prompt</h3>

      <div className="llm-grid">
        <div className="control-group">
          <label htmlFor="llm-provider">Provider</label>
          <select id="llm-provider" value={provider} onChange={(e) => selectProvider(e.target.value)} disabled={locked}>
            {Object.entries(config.providers).map(([name, p]) => (
              <option key={name} value={name}>
                {p.label}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="llm-model">Model</label>
          <input
            id="llm-model"
            type="text"
            list="llm-model-options"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder={info.kind === 'cli' ? 'CLI default' : 'Model id'}
            disabled={locked}
          />
          <datalist id="llm-model-options">
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </div>

        {info.kind === 'openai_compat' && (
          <div className="control-group">
            <label htmlFor="llm-base-url">Base URL</label>
            <input
              id="llm-base-url"
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder={info.default_base_url || 'https://host/v1'}
              disabled={locked}
            />
          </div>
        )}

        {info.kind !== 'cli' && (
          <div className="control-group">
            <label htmlFor="llm-api-key">API key</label>
            <input
              id="llm-api-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={keyPlaceholder}
              autoComplete="off"
              disabled={locked}
            />
          </div>
        )}
      </div>

      {info.kind === 'cli' && (
        <p className="control-hint">
          {info.cli_found
            ? 'Found on the server PATH. Uses the login of the account running the backend.'
            : 'Not found on the server PATH. Install it and log in before using this provider.'}
        </p>
      )}
      {info.kind === 'anthropic' && !info.api_key_set && (
        <p className="control-hint">
          No key stored. The server will fall back to ANTHROPIC_API_KEY or an `ant auth login` profile.
        </p>
      )}

      <div className="llm-actions">
        <button onClick={save} disabled={locked} className="secondary-btn">
          {busy === 'save' ? 'Saving...' : 'Save and use'}
        </button>
        <button onClick={test} disabled={locked} className="secondary-btn">
          {busy === 'test' ? 'Testing...' : 'Save and test'}
        </button>
        {info.kind !== 'cli' && (
          <button onClick={loadModels} disabled={locked} className="secondary-btn">
            {busy === 'models' ? 'Loading...' : 'Load model list'}
          </button>
        )}
        {info.key_source === 'stored' && (
          <button onClick={clearKey} disabled={locked} className="cancel-btn">
            Remove stored key
          </button>
        )}
        {config.active !== provider && <span className="control-hint">Not saved yet. Active: {config.providers[config.active].label}</span>}
      </div>

      {message && <div className={message.ok ? 'success-message' : 'error-message'}>{message.text}</div>}
    </div>
  )
}
