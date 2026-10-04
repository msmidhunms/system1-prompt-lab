import { useEffect, useState } from 'react'
import { api, apiError } from '../api'
import { useTask } from '../TaskContext'
import '../styles/GoldenDataset.css'

interface GoldenRow {
  id: string
  text: string
  label: string
  secondary_labels: string[]
  language: string | null
  source: string
  updated_at: string | null
}

interface PaginationInfo {
  current_page: number
  page_size: number
  total_items: number
  total_pages: number
}

interface GoldenStats {
  total: number
  by_label: Record<string, number>
  by_source: Record<string, number>
  eval_language: string | null
  usable: number
  excluded_other_language: number
}

interface UpsertResult {
  status: 'created' | 'updated'
  previous_label: string | null
  row: GoldenRow
}

interface Draft {
  text: string
  label: string
  secondary: string[]
  language: string
}

const PAGE_SIZE_OPTIONS = [10, 20, 50, 100]
const SOURCES = ['imported', 'manual', 'feedback']
const PREVIEW_CHARS = 280

export default function GoldenDataset() {
  const { task, color, refreshTasks } = useTask()
  const hasSecondary = task.id === 'search_intent'
  const hasLanguage = task.language != null

  const [rows, setRows] = useState<GoldenRow[]>([])
  const [pagination, setPagination] = useState<PaginationInfo>({ current_page: 1, page_size: 20, total_items: 0, total_pages: 0 })
  const [stats, setStats] = useState<GoldenStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [labelFilter, setLabelFilter] = useState('')
  const [sourceFilter, setSourceFilter] = useState('')

  const [newRow, setNewRow] = useState<Draft>({ text: '', label: task.labels[0], secondary: [], language: '' })
  const [saving, setSaving] = useState(false)
  const [importing, setImporting] = useState(false)

  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Draft>({ text: '', label: '', secondary: [], language: '' })
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)

  const loadRows = async (page: number, pageSize = pagination.page_size) => {
    setLoading(true)
    try {
      const res = await api.get<{ items: GoldenRow[]; pagination: PaginationInfo }>('/golden-data', {
        params: {
          task: task.id,
          page,
          page_size: pageSize,
          q: query || undefined,
          label: labelFilter || undefined,
          source: sourceFilter || undefined,
        },
      })
      setRows(res.data.items)
      setPagination(res.data.pagination)
    } catch (err) {
      setError(apiError(err, 'Failed to load the golden dataset'))
    } finally {
      setLoading(false)
    }
  }

  const loadStats = () =>
    api
      .get<GoldenStats>('/golden-data/stats', { params: { task: task.id } })
      .then((res) => setStats(res.data))
      .catch((err) => console.error('Failed to load stats:', err))

  // After a change: the page, the counts on this tab and the count in the header.
  const reload = (page = pagination.current_page) => {
    loadRows(page)
    loadStats()
    refreshTasks()
  }

  useEffect(() => {
    loadRows(1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, labelFilter, sourceFilter])

  useEffect(() => {
    loadStats()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const say = (message: string) => {
    setError(null)
    setNotice(message)
  }

  const addRow = async () => {
    if (!newRow.text.trim()) return
    setSaving(true)
    setError(null)
    setNotice(null)
    try {
      const res = await api.post<UpsertResult>('/golden-data', {
        task: task.id,
        text: newRow.text,
        label: newRow.label,
        secondary_labels: hasSecondary ? newRow.secondary.filter((s) => s !== newRow.label) : undefined,
        language: hasLanguage && newRow.language.trim() ? newRow.language.trim() : undefined,
      })
      const { status, previous_label, row } = res.data
      if (status === 'created') {
        say(`Added new row, labelled ${row.label}.`)
      } else if (previous_label === row.label) {
        say(`This ${task.item} was already in the dataset with the label ${row.label}. The existing row was kept.`)
      } else {
        say(`Updated existing row: ${previous_label} → ${row.label}. This ${task.item} was already in the dataset.`)
      }
      setNewRow({ ...newRow, text: '', secondary: [] })
      reload(1)
    } catch (err) {
      setError(apiError(err, 'Failed to save the row'))
    } finally {
      setSaving(false)
    }
  }

  const startEdit = (row: GoldenRow) => {
    setEditingId(row.id)
    setConfirmDelete(null)
    setDraft({ text: row.text, label: row.label, secondary: row.secondary_labels, language: row.language ?? '' })
  }

  const saveEdit = async () => {
    if (!editingId) return
    setError(null)
    try {
      await api.put(`/golden-data/${editingId}`, {
        text: draft.text,
        label: draft.label,
        secondary_labels: hasSecondary ? draft.secondary.filter((s) => s !== draft.label) : undefined,
        language: hasLanguage ? draft.language : undefined,
      })
      setEditingId(null)
      say('Row updated.')
      reload()
    } catch (err) {
      setError(apiError(err, 'Failed to update the row'))
    }
  }

  const deleteRow = async (id: string) => {
    setError(null)
    try {
      await api.delete(`/golden-data/${id}`)
      setConfirmDelete(null)
      say('Row deleted.')
      reload(rows.length === 1 && pagination.current_page > 1 ? pagination.current_page - 1 : pagination.current_page)
    } catch (err) {
      setError(apiError(err, 'Failed to delete the row'))
    }
  }

  const importDataset = async () => {
    setImporting(true)
    setError(null)
    setNotice(null)
    try {
      const res = await api.post<{ created: number; already_present: number }>(`/tasks/${task.id}/import`, {})
      say(`Imported ${res.data.created} rows (${res.data.already_present} were already in the dataset).`)
      reload(1)
    } catch (err) {
      setError(apiError(err, 'Failed to import the dataset'))
    } finally {
      setImporting(false)
    }
  }

  const toggle = (list: string[], value: string) => (list.includes(value) ? list.filter((v) => v !== value) : [...list, value])

  const badge = (label: string, secondary = false) => (
    <span
      key={label}
      className={secondary ? 'secondary-badge' : 'intent-badge'}
      style={{ backgroundColor: `${color(label)}${secondary ? '22' : '33'}`, color: color(label) }}
    >
      {label}
    </span>
  )

  const secondaryPicker = (value: Draft, onChange: (next: Draft) => void) => (
    <div className="secondary-picker">
      {task.labels
        .filter((label) => label !== value.label)
        .map((label) => (
          <label key={label}>
            <input
              type="checkbox"
              checked={value.secondary.includes(label)}
              onChange={() => onChange({ ...value, secondary: toggle(value.secondary, label) })}
            />
            {label}
          </label>
        ))}
    </div>
  )

  const columns = 4 + (hasSecondary ? 1 : 0) + (hasLanguage ? 1 : 0)
  const empty = stats !== null && stats.total === 0

  return (
    <div className="keyword-analysis">
      <div className="section">
        <h2>Golden Dataset: {task.name}</h2>
        <p className="description">
          The labelled {task.items} that Model Evaluation and the Karpathy Loop score against.
          {task.source && ` Source: ${task.source}.`}
        </p>

        {stats && (
          <div className="stats-section">
            <div className="stat-card">
              <span className="stat-label">Total rows</span>
              <span className="stat-value">{stats.total}</span>
            </div>
            {task.labels.map((label) => (
              <div key={label} className="stat-card">
                <span className="stat-label">{label}</span>
                <span className="stat-value" style={{ color: color(label) }}>
                  {stats.by_label[label] ?? 0}
                </span>
              </div>
            ))}
          </div>
        )}
        {stats && (
          <p className="dataset-hint">
            {SOURCES.filter((s) => stats.by_source[s]).map((s) => `${stats.by_source[s]} ${s}`).join(' · ') || 'No rows yet'}
            {stats.eval_language &&
              ` · only the ${stats.usable} rows in language "${stats.eval_language}" are used for evaluation and the loop`}
          </p>
        )}

        {error && <div className="error-message">{error}</div>}
        {notice && <div className="success-message">{notice}</div>}
      </div>

      <div className="section">
        <h3>Add or update a row</h3>
        <p className="description">
          If the {task.item} is already in the dataset (ignoring upper/lower case and extra spaces), its row is updated.
          Otherwise a new row is created.
        </p>
        <div className="add-row-form">
          <textarea
            placeholder={`${task.input_label}...`}
            value={newRow.text}
            rows={hasSecondary ? 1 : 3}
            onChange={(e) => setNewRow({ ...newRow, text: e.target.value })}
          />
          <div className="add-row-fields">
            <div className="field">
              <label htmlFor="new-label">Label</label>
              <select id="new-label" value={newRow.label} onChange={(e) => setNewRow({ ...newRow, label: e.target.value })}>
                {task.labels.map((label) => (
                  <option key={label} value={label}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            {hasSecondary && (
              <div className="field">
                <label>Secondary intents</label>
                {secondaryPicker(newRow, setNewRow)}
              </div>
            )}
            {hasLanguage && (
              <div className="field">
                <label htmlFor="new-language">Language</label>
                <input
                  id="new-language"
                  type="text"
                  placeholder="auto"
                  value={newRow.language}
                  maxLength={5}
                  onChange={(e) => setNewRow({ ...newRow, language: e.target.value })}
                />
              </div>
            )}
            <button onClick={addRow} disabled={saving || !newRow.text.trim()} className="save-btn">
              {saving ? 'Saving...' : 'Save row'}
            </button>
          </div>
        </div>
      </div>

      <div className="section table-section">
        <div className="table-header">
          <h3>Rows</h3>
          <div className="page-size-control">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                setQuery(search.trim())
              }}
            >
              <input
                type="search"
                placeholder={`Search ${task.items}...`}
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value)
                  if (e.target.value === '') setQuery('')
                }}
              />
              <button type="submit" className="save-btn">
                Search
              </button>
            </form>
            <select value={labelFilter} onChange={(e) => setLabelFilter(e.target.value)} aria-label="Filter by label">
              <option value="">All labels</option>
              {task.labels.map((label) => (
                <option key={label} value={label}>
                  {label}
                </option>
              ))}
            </select>
            <select value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)} aria-label="Filter by source">
              <option value="">All sources</option>
              {SOURCES.map((source) => (
                <option key={source} value={source}>
                  {source}
                </option>
              ))}
            </select>
            <select
              value={pagination.page_size}
              onChange={(e) => loadRows(1, Number(e.target.value))}
              aria-label="Rows per page"
            >
              {PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size}>
                  {size} per page
                </option>
              ))}
            </select>
          </div>
        </div>

        {empty ? (
          <div className="empty-state">
            <p>This example has no golden data yet.</p>
            {task.importable ? (
              <button onClick={importDataset} disabled={importing} className="save-btn">
                {importing ? 'Importing (downloads the dataset)...' : 'Import dataset'}
              </button>
            ) : (
              <p>Add rows with the form above.</p>
            )}
          </div>
        ) : loading && rows.length === 0 ? (
          <div className="loading">Loading rows...</div>
        ) : rows.length === 0 ? (
          <div className="empty-state">No rows match these filters.</div>
        ) : (
          <>
            <table className="keywords-table">
              <thead>
                <tr>
                  <th>{task.input_label}</th>
                  <th>Label</th>
                  {hasSecondary && <th>Secondary</th>}
                  {hasLanguage && <th>Language</th>}
                  <th>Source</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) =>
                  editingId === row.id ? (
                    <tr key={row.id} className="editing-row">
                      <td colSpan={columns}>
                        <div className="add-row-form">
                          <textarea
                            value={draft.text}
                            rows={hasSecondary ? 1 : 4}
                            onChange={(e) => setDraft({ ...draft, text: e.target.value })}
                          />
                          <div className="add-row-fields">
                            <div className="field">
                              <label>Label</label>
                              <select value={draft.label} onChange={(e) => setDraft({ ...draft, label: e.target.value })}>
                                {task.labels.map((label) => (
                                  <option key={label} value={label}>
                                    {label}
                                  </option>
                                ))}
                              </select>
                            </div>
                            {hasSecondary && (
                              <div className="field">
                                <label>Secondary intents</label>
                                {secondaryPicker(draft, setDraft)}
                              </div>
                            )}
                            {hasLanguage && (
                              <div className="field">
                                <label>Language</label>
                                <input
                                  type="text"
                                  value={draft.language}
                                  maxLength={5}
                                  onChange={(e) => setDraft({ ...draft, language: e.target.value })}
                                />
                              </div>
                            )}
                            <button onClick={saveEdit} disabled={!draft.text.trim()} className="save-btn">
                              Save
                            </button>
                            <button onClick={() => setEditingId(null)} className="row-btn">
                              Cancel
                            </button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    <tr key={row.id}>
                      <td className="keyword-cell text-cell">
                        {expanded === row.id || row.text.length <= PREVIEW_CHARS ? row.text : `${row.text.slice(0, PREVIEW_CHARS)}…`}
                        {row.text.length > PREVIEW_CHARS && (
                          <button className="link-btn" onClick={() => setExpanded(expanded === row.id ? null : row.id)}>
                            {expanded === row.id ? 'show less' : 'show all'}
                          </button>
                        )}
                      </td>
                      <td>{badge(row.label)}</td>
                      {hasSecondary && (
                        <td>
                          <div className="secondary-intents">
                            {row.secondary_labels.length > 0 ? (
                              row.secondary_labels.map((label) => badge(label, true))
                            ) : (
                              <span className="no-secondary">-</span>
                            )}
                          </div>
                        </td>
                      )}
                      {hasLanguage && <td className="model-cell">{row.language ?? '-'}</td>}
                      <td className="model-cell">{row.source}</td>
                      <td className="actions-cell">
                        {confirmDelete === row.id ? (
                          <>
                            <button onClick={() => deleteRow(row.id)} className="row-btn danger">
                              Confirm delete
                            </button>
                            <button onClick={() => setConfirmDelete(null)} className="row-btn">
                              Keep
                            </button>
                          </>
                        ) : (
                          <>
                            <button onClick={() => startEdit(row)} className="row-btn">
                              Edit
                            </button>
                            <button onClick={() => setConfirmDelete(row.id)} className="row-btn">
                              Delete
                            </button>
                          </>
                        )}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>

            <div className="pagination">
              <button
                onClick={() => loadRows(pagination.current_page - 1)}
                disabled={pagination.current_page <= 1}
                className="pagination-btn"
              >
                ← Previous
              </button>

              <div className="page-info">
                Page {pagination.current_page} of {pagination.total_pages}
                <span className="total-items">({pagination.total_items} rows)</span>
              </div>

              <button
                onClick={() => loadRows(pagination.current_page + 1)}
                disabled={pagination.current_page >= pagination.total_pages}
                className="pagination-btn"
              >
                Next →
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
