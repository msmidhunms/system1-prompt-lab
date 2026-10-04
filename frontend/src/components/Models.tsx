import { useEffect, useState } from 'react'
import { api, apiError, ModelVersion, pct, PromptConfig, SOURCE_LABELS, when } from '../api'
import { useApp, useTask } from '../context'
import { Badge, Banner, Button, Card, Dialog, Menu, NameDialog } from '../ui'
import { ConfigView, PromptEditor } from './PromptConfig'

type EditSource = { name: string; config: PromptConfig; base: string | null }

export default function Models() {
  const { task, versions, refreshVersions, open, intent, clearIntent } = useTask()
  const { engineName } = useApp()
  const [viewing, setViewing] = useState<ModelVersion | null>(null)
  const [editing, setEditing] = useState<EditSource | null>(null)
  const [renaming, setRenaming] = useState<ModelVersion | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  // The optimizer can hand over a round's prompt to edit.
  useEffect(() => {
    if (intent?.tab === 'models' && intent.editFrom) {
      setEditing(intent.editFrom)
      clearIntent()
    }
  }, [intent, clearIntent])

  const copyName = (name: string) => {
    const taken = new Set(versions.map((v) => v.version))
    const base = name === 'v1_baseline' ? `${task.id}_custom` : `${name}_copy`
    let candidate = base
    for (let n = 2; taken.has(candidate); n++) candidate = `${base}${n}`
    return candidate
  }

  const remove = async (name: string) => {
    setError(null)
    try {
      await api.delete(`/models/${name}`)
      setConfirmDelete(null)
      setNotice(`Deleted ${name}.`)
      refreshVersions()
    } catch (err) {
      setError(apiError(err, 'Failed to delete the model'))
    }
  }

  return (
    <div className="page">
      {error && <Banner tone="error">{error}</Banner>}
      {notice && <Banner tone="success">{notice}</Banner>}
      <Card
        title="Model versions"
        sub="A version is a prompt for one model: how the input is presented, the question, and a description of each label. The optimizer saves its best prompts here, and you can write your own from a copy. Any version can also be tried or evaluated on another model."
        flush
      >
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Model</th>
                <th>Origin</th>
                <th>Based on</th>
                <th>Dev accuracy</th>
                <th>Holdout</th>
                <th>Latest evaluation</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {versions.map((v) => (
                <tr key={v.version} className="clickable" onClick={() => setViewing(v)}>
                  <td>
                    <strong>{v.version}</strong>
                    {v.description && <div className="small secondary">{v.description}</div>}
                  </td>
                  <td className="secondary">{v.source === 'baseline' ? 'Any' : engineName(v.config.engine)}</td>
                  <td><Badge tone={v.source === 'auto' ? 'accent' : undefined}>{SOURCE_LABELS[v.source]}</Badge></td>
                  <td className="secondary">{v.base_version ?? '–'}</td>
                  <td className="num">{pct(v.accuracy)}</td>
                  <td className="num">{pct(v.holdout_accuracy)}</td>
                  <td className="num">
                    {v.latest_evaluation ? `${pct(v.latest_evaluation.accuracy)} on ${v.latest_evaluation.sample_size}` : <span className="muted">Not evaluated</span>}
                  </td>
                  <td className="secondary nowrap">{v.created_at ? when(v.created_at) : '–'}</td>
                  <td className="actions">
                    {confirmDelete === v.version ? (
                      <span className="row" onClick={(e) => e.stopPropagation()}>
                        <Button small variant="danger" onClick={() => remove(v.version)}>Confirm delete</Button>
                        <Button small onClick={() => setConfirmDelete(null)}>Keep</Button>
                      </span>
                    ) : (
                      <Menu
                        label={`Actions for ${v.version}`}
                        items={[
                          { label: 'View prompt', onSelect: () => setViewing(v) },
                          { label: 'Duplicate and edit', onSelect: () => setEditing({ name: copyName(v.version), config: v.config, base: v.version }) },
                          { label: 'Evaluate', onSelect: () => open('evaluation', { tab: 'evaluation', version: v.version }) },
                          { label: 'Try in Playground', onSelect: () => open('playground', { tab: 'playground', version: v.version }) },
                          ...(v.source !== 'baseline'
                            ? [
                                { label: 'Rename', onSelect: () => setRenaming(v) },
                                { label: 'Delete', danger: true, onSelect: () => setConfirmDelete(v.version) },
                              ]
                            : []),
                        ]}
                      />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {viewing && (
        <Dialog
          title={viewing.version}
          wide
          onClose={() => setViewing(null)}
          footer={
            <>
              <Button onClick={() => setViewing(null)}>Close</Button>
              <Button
                variant="primary"
                onClick={() => {
                  setEditing({ name: copyName(viewing.version), config: viewing.config, base: viewing.version })
                  setViewing(null)
                }}
              >
                Duplicate and edit
              </Button>
            </>
          }
        >
          <ConfigView config={viewing.config} labels={task.labels} />
        </Dialog>
      )}

      {editing && (
        <PromptEditor
          task={task}
          from={editing}
          onClose={() => setEditing(null)}
          onSaved={(saved) => {
            setNotice(`Saved ${saved.version}. Evaluate it to see how it scores.`)
            refreshVersions()
          }}
        />
      )}

      {renaming && (
        <NameDialog
          title={`Rename ${renaming.version}`}
          label="New name"
          initial={renaming.version}
          confirm="Rename"
          onClose={() => setRenaming(null)}
          onSubmit={async (name) => {
            try {
              await api.patch(`/models/${renaming.version}`, { name })
              setNotice(`Renamed ${renaming.version} to ${name}.`)
              refreshVersions()
              return null
            } catch (err) {
              return apiError(err, 'Failed to rename the model')
            }
          }}
        />
      )}
    </div>
  )
}
