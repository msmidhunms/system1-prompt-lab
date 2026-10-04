import { useState } from 'react'
import { api, apiError } from '../api'
import { useApp } from '../context'
import { Banner, Button, Dialog, StatusBadge } from '../ui'

// Where each example's golden data comes from, and one button to fetch what is missing.
export default function SetupDialog({ onClose }: { onClose: () => void }) {
  const { tasks, refreshTasks, refreshActivity } = useApp()
  const [error, setError] = useState<string | null>(null)

  const start = async (ids: string[]) => {
    setError(null)
    try {
      await api.post('/setup/import', { tasks: ids })
      await Promise.all([refreshTasks(), refreshActivity()])
    } catch (err) {
      setError(apiError(err, 'Failed to start the import'))
    }
  }

  const busy = (status?: string) => status === 'queued' || status === 'downloading'
  const downloadable = tasks.filter((t) => t.golden_rows === 0 && t.importable && !busy(t.import_status?.status))

  return (
    <Dialog
      title="Set up data"
      wide
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Close</Button>
          <Button variant="primary" disabled={downloadable.length === 0} onClick={() => start(downloadable.map((t) => t.id))}>
            {downloadable.length > 0 ? `Download ${downloadable.length} missing dataset${downloadable.length > 1 ? 's' : ''}` : 'Nothing to download'}
          </Button>
        </>
      }
    >
      <div className="stack">
        <p className="secondary">
          Each example scores against a golden dataset. The public ones are downloaded from Hugging Face (2,000 rows
          each, a few MB) and kept under <code>data/tasks/</code>. The same from a terminal:{' '}
          <code>python backend/scripts/import_dataset.py --task all</code>
        </p>
        {error && <Banner tone="error">{error}</Banner>}
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Example</th>
                <th>Golden rows</th>
                <th>Source</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((t) => (
                <tr key={t.id}>
                  <td className="nowrap">{t.name}</td>
                  <td className="num">{t.golden_rows.toLocaleString()}</td>
                  <td className="small secondary">
                    {t.source}
                    {t.golden_rows === 0 && !t.importable && <div style={{ marginTop: 4 }}>{t.setup_hint}</div>}
                    {t.import_status?.status === 'failed' && <div style={{ color: 'var(--critical)', marginTop: 4 }}>{t.import_status.message}</div>}
                  </td>
                  <td className="actions">
                    {busy(t.import_status?.status) ? (
                      <StatusBadge status={t.import_status!.status} />
                    ) : t.golden_rows > 0 ? (
                      <StatusBadge status="done" />
                    ) : t.importable ? (
                      <Button small onClick={() => start([t.id])}>
                        Download
                      </Button>
                    ) : (
                      <span className="badge warn">Manual</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Dialog>
  )
}
