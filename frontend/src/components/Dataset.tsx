import { useEffect, useState } from 'react'
import { api, apiError } from '../api'
import { useApp, useTask } from '../context'
import { Banner, Button, Card, EmptyState, Field, LabelBadge, Menu, Spinner, StatTile } from '../ui'

interface GoldenRow {
  id: string
  text: string
  label: string
  secondary_labels: string[]
  language: string | null
  source: string
}

interface Pagination {
  current_page: number
  page_size: number
  total_items: number
  total_pages: number
}

interface Stats {
  total: number
  by_label: Record<string, number>
  by_source: Record<string, number>
  eval_language: string | null
  usable: number
}

interface Draft {
  text: string
  label: string
  secondary: string[]
  language: string
}

const SOURCES = ['imported', 'manual', 'feedback']
const PREVIEW_CHARS = 280

export default function Dataset() {
  const { task, color } = useTask()
  const { refreshTasks, refreshActivity } = useApp()
  const hasSecondary = task.id === 'search_intent'
  const hasLanguage = task.language != null
  const importing = task.import_status?.status === 'queued' || task.import_status?.status === 'downloading'

  const [rows, setRows] = useState<GoldenRow[]>([])
  const [pagination, setPagination] = useState<Pagination>({ current_page: 1, page_size: 20, total_items: 0, total_pages: 0 })
  const [stats, setStats] = useState<Stats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [labelFilter, setLabelFilter] = useState('')
  const [sourceFilter, setSourceFilter] = useState('')
  const [newRow, setNewRow] = useState<Draft>({ text: '', label: task.labels[0], secondary: [], language: '' })
  const [saving, setSaving] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Draft>({ text: '', label: '', secondary: [], language: '' })
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)

  const loadRows = async (page: number, pageSize = pagination.page_size) => {
    try {
      const res = await api.get<{ items: GoldenRow[]; pagination: Pagination }>('/golden-data', {
        params: { task: task.id, page, page_size: pageSize, q: query || undefined, label: labelFilter || undefined, source: sourceFilter || undefined },
      })
      setRows(res.data.items)
      setPagination(res.data.pagination)
    } catch (err) {
      setError(apiError(err, 'Failed to load the golden dataset'))
    }
  }

  const loadStats = () =>
    api.get<Stats>('/golden-data/stats', { params: { task: task.id } }).then((res) => setStats(res.data)).catch(() => undefined)

  const reload = (page = pagination.current_page) => {
    loadRows(page)
    loadStats()
    refreshTasks()
  }

  useEffect(() => {
    loadRows(1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, labelFilter, sourceFilter])

  // The row count changes when an import or feedback from another tab adds rows.
  useEffect(() => {
    loadStats()
    loadRows(pagination.current_page)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.golden_rows])

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
      const res = await api.post<{ status: 'created' | 'updated'; previous_label: string | null; row: GoldenRow }>('/golden-data', {
        task: task.id,
        text: newRow.text,
        label: newRow.label,
        secondary_labels: hasSecondary ? newRow.secondary.filter((s) => s !== newRow.label) : undefined,
        language: hasLanguage && newRow.language.trim() ? newRow.language.trim() : undefined,
      })
      const { status, previous_label, row } = res.data
      if (status === 'created') say(`Added new row, labelled ${row.label}.`)
      else if (previous_label === row.label) say(`This ${task.item} was already in the dataset with the label ${row.label}. The existing row was kept.`)
      else say(`Updated existing row: ${previous_label} → ${row.label}. This ${task.item} was already in the dataset.`)
      setNewRow({ ...newRow, text: '', secondary: [] })
      reload(1)
    } catch (err) {
      setError(apiError(err, 'Failed to save the row'))
    } finally {
      setSaving(false)
    }
  }

  const saveEdit = async () => {
    if (!editingId) return
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
    setError(null)
    try {
      await api.post('/setup/import', { tasks: [task.id] })
      await Promise.all([refreshTasks(), refreshActivity()])
    } catch (err) {
      setError(apiError(err, 'Failed to start the import'))
    }
  }

  const toggle = (list: string[], value: string) => (list.includes(value) ? list.filter((v) => v !== value) : [...list, value])

  const fields = (value: Draft, onChange: (next: Draft) => void) => (
    <>
      <Field label="Label">
        <select className="input" value={value.label} onChange={(e) => onChange({ ...value, label: e.target.value })}>
          {task.labels.map((label) => (
            <option key={label} value={label}>{label}</option>
          ))}
        </select>
      </Field>
      {hasSecondary && (
        <Field label="Secondary intents">
          <div className="row" style={{ height: 32 }}>
            {task.labels.filter((label) => label !== value.label).map((label) => (
              <label key={label} className="check">
                <input type="checkbox" checked={value.secondary.includes(label)} onChange={() => onChange({ ...value, secondary: toggle(value.secondary, label) })} />
                {label}
              </label>
            ))}
          </div>
        </Field>
      )}
      {hasLanguage && (
        <Field label="Language">
          <input className="input narrow" placeholder="auto" maxLength={5} value={value.language} onChange={(e) => onChange({ ...value, language: e.target.value })} />
        </Field>
      )}
    </>
  )

  const columns = 4 + (hasSecondary ? 1 : 0) + (hasLanguage ? 1 : 0)
  const empty = task.golden_rows === 0

  return (
    <div className="page">
      {error && <Banner tone="error">{error}</Banner>}
      {notice && <Banner tone="success">{notice}</Banner>}

      {empty ? (
        <Card>
          {importing ? (
            <EmptyState title="Importing the dataset">
              <div className="row" style={{ justifyContent: 'center' }}>
                <Spinner /> {task.import_status?.message}
              </div>
            </EmptyState>
          ) : (
            <EmptyState
              title="This example has no golden data yet"
              actions={task.importable ? <Button variant="primary" onClick={importDataset}>Download dataset</Button> : undefined}
            >
              <p>{task.source ? `Source: ${task.source}.` : ''}</p>
              {task.import_status?.status === 'failed' && <p style={{ color: 'var(--critical)' }}>The last import failed: {task.import_status.message}</p>}
              <p className="small" style={{ marginTop: 8 }}>
                {task.importable ? <>Or from a terminal: <code>{task.setup_hint}</code></> : task.setup_hint}
              </p>
              <p className="small muted" style={{ marginTop: 8 }}>Rows can also be added one at a time below.</p>
            </EmptyState>
          )}
        </Card>
      ) : (
        stats && (
          <div>
            <div className="stats">
              <StatTile label="Total rows" value={stats.total.toLocaleString()} />
              {task.labels.map((label) => (
                <StatTile key={label} label={label} dot={color(label)} value={(stats.by_label[label] ?? 0).toLocaleString()} />
              ))}
            </div>
            <p className="small muted" style={{ marginTop: 8 }}>
              {SOURCES.filter((s) => stats.by_source[s]).map((s) => `${stats.by_source[s]} ${s}`).join(' · ')}
              {task.source && ` · source: ${task.source}`}
              {stats.eval_language && ` · only the ${stats.usable} rows in language "${stats.eval_language}" are used for evaluation and optimization`}
            </p>
          </div>
        )
      )}

      <Card
        title="Add or update a row"
        sub={`If the ${task.item} is already in the dataset (ignoring upper/lower case and extra spaces), its row is updated. Otherwise a new row is created.`}
      >
        <div className="stack">
          <textarea className="input" placeholder={`${task.input_label}…`} rows={hasSecondary ? 1 : 3} value={newRow.text} onChange={(e) => setNewRow({ ...newRow, text: e.target.value })} />
          <div className="row end">
            {fields(newRow, setNewRow)}
            <Button variant="primary" onClick={addRow} disabled={saving || !newRow.text.trim()}>
              {saving ? 'Saving…' : 'Save row'}
            </Button>
          </div>
        </div>
      </Card>

      {!empty && (
        <Card
          title="Rows"
          flush
          actions={
            <>
              <form
                className="row"
                onSubmit={(e) => {
                  e.preventDefault()
                  setQuery(search.trim())
                }}
              >
                <input
                  className="input"
                  type="search"
                  placeholder={`Search ${task.items}…`}
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value)
                    if (e.target.value === '') setQuery('')
                  }}
                />
                <Button type="submit">Search</Button>
              </form>
              <select className="input" value={labelFilter} onChange={(e) => setLabelFilter(e.target.value)} aria-label="Filter by label">
                <option value="">All labels</option>
                {task.labels.map((label) => (
                  <option key={label} value={label}>{label}</option>
                ))}
              </select>
              <select className="input" value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)} aria-label="Filter by source">
                <option value="">All sources</option>
                {SOURCES.map((source) => (
                  <option key={source} value={source}>{source}</option>
                ))}
              </select>
            </>
          }
        >
          {rows.length === 0 ? (
            <EmptyState title="No rows match these filters" />
          ) : (
            <div className="table-wrap">
              <table className="table">
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
                      <tr key={row.id} className="detail">
                        <td colSpan={columns}>
                          <div className="stack">
                            <textarea className="input" rows={hasSecondary ? 1 : 4} value={draft.text} onChange={(e) => setDraft({ ...draft, text: e.target.value })} />
                            <div className="row end">
                              {fields(draft, setDraft)}
                              <Button variant="primary" onClick={saveEdit} disabled={!draft.text.trim()}>Save</Button>
                              <Button onClick={() => setEditingId(null)}>Cancel</Button>
                            </div>
                          </div>
                        </td>
                      </tr>
                    ) : (
                      <tr key={row.id}>
                        <td className="wrap">
                          {expanded === row.id || row.text.length <= PREVIEW_CHARS ? row.text : `${row.text.slice(0, PREVIEW_CHARS)}…`}
                          {row.text.length > PREVIEW_CHARS && (
                            <>
                              {' '}
                              <button className="link small" onClick={() => setExpanded(expanded === row.id ? null : row.id)}>
                                {expanded === row.id ? 'Show less' : 'Show all'}
                              </button>
                            </>
                          )}
                        </td>
                        <td><LabelBadge label={row.label} color={color(row.label)} /></td>
                        {hasSecondary && (
                          <td>
                            <div className="row">
                              {row.secondary_labels.length > 0 ? row.secondary_labels.map((label) => <LabelBadge key={label} label={label} color={color(label)} />) : <span className="muted">–</span>}
                            </div>
                          </td>
                        )}
                        {hasLanguage && <td className="secondary">{row.language ?? '–'}</td>}
                        <td className="secondary">{row.source}</td>
                        <td className="actions">
                          {confirmDelete === row.id ? (
                            <span className="row">
                              <Button small variant="danger" onClick={() => deleteRow(row.id)}>Confirm delete</Button>
                              <Button small onClick={() => setConfirmDelete(null)}>Keep</Button>
                            </span>
                          ) : (
                            <Menu
                              label={`Actions for row ${row.text.slice(0, 30)}`}
                              items={[
                                {
                                  label: 'Edit',
                                  onSelect: () => {
                                    setEditingId(row.id)
                                    setDraft({ text: row.text, label: row.label, secondary: row.secondary_labels, language: row.language ?? '' })
                                  },
                                },
                                { label: 'Delete', danger: true, onSelect: () => setConfirmDelete(row.id) },
                              ]}
                            />
                          )}
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
          )}
          <div className="row between" style={{ padding: '10px 16px' }}>
            <span className="small secondary">
              Page {pagination.current_page} of {Math.max(1, pagination.total_pages)} · {pagination.total_items.toLocaleString()} rows
            </span>
            <span className="row">
              <select className="input" value={pagination.page_size} onChange={(e) => loadRows(1, Number(e.target.value))} aria-label="Rows per page">
                {[10, 20, 50, 100].map((size) => (
                  <option key={size} value={size}>{size} per page</option>
                ))}
              </select>
              <Button onClick={() => loadRows(pagination.current_page - 1)} disabled={pagination.current_page <= 1}>Previous</Button>
              <Button onClick={() => loadRows(pagination.current_page + 1)} disabled={pagination.current_page >= pagination.total_pages}>Next</Button>
            </span>
          </div>
        </Card>
      )}
    </div>
  )
}
