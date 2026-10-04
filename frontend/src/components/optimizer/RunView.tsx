import { Fragment, useState } from 'react'
import { api, apiError, pct, PromptConfig } from '../../api'
import { useTask } from '../../context'
import { Banner, Button, Card, Menu, NameDialog, Spinner, StatTile, StatusBadge } from '../../ui'
import { ConfigView } from '../PromptConfig'
import ScoreChart from './ScoreChart'
import { Run, Round, scoreOf } from './types'

const signedPts = (value: number) => `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)} pts`

export default function RunView({ run, objective, keepConfidence, onStop, onChanged }: {
  run: Run
  objective: string
  keepConfidence: number
  onStop: () => void
  onChanged: () => void
}) {
  const { open, refreshVersions } = useTask()
  const [expanded, setExpanded] = useState<number | null>(null)
  const [showLog, setShowLog] = useState(false)
  const [saving, setSaving] = useState<Round | 'best' | null>(null)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const running = run.status === 'running'
  const labels = run.labels
  const baselineScore = scoreOf(run.baseline)
  const bestScore = run.best_score ?? run.best_accuracy
  const gain = baselineScore != null && bestScore != null ? bestScore - baselineScore : null
  const autoVersion = run.saved_versions?.find((v) => v.source === 'auto')
  const otherVersions = run.saved_versions?.filter((v) => v.source !== 'auto') ?? []

  // The prompt each round was compared against: the best one at the time it ran.
  const bestBefore = (index: number): PromptConfig => {
    for (let i = index - 1; i >= 0; i--) {
      const prev = run.iterations[i]
      if (prev.status === 'keep' && prev.config) return prev.config
    }
    return run.baseline_config
  }

  const save = async (name: string): Promise<string | null> => {
    try {
      await api.post('/save-model', {
        model_name: name,
        run_id: run.run_id,
        iteration: saving === 'best' || saving === null ? undefined : saving.iteration,
        description: saving === 'best' || saving === null ? `Best prompt of run ${run.run_id}` : `Round ${saving.iteration} of run ${run.run_id}`,
      })
      setNotice(`Saved as ${name}. It is now in Models, Evaluation and the Playground.`)
      refreshVersions()
      onChanged()
      return null
    } catch (err) {
      return apiError(err, 'Failed to save the model')
    }
  }

  return (
    <>
      <Card
        title={
          <span className="row">
            Run {run.run_id} <StatusBadge status={run.status} />
          </span>
        }
        sub={
          <>
            {run.llm.provider}
            {run.llm.model ? ` · ${run.llm.model}` : ''} · started from {run.start_version}
            {run.laya_model ? ` · Laya ${run.laya_model}` : ''} · objective: {objective}
            {run.calibrate ? ' · calibrated' : ''}
            {run.use_serp ? ' · search-result context allowed' : ''} · {run.dev_size} dev / {run.holdout_size} holdout {run.items ?? 'rows'}
          </>
        }
        actions={running ? <Button variant="danger" onClick={onStop}>Stop after this round</Button> : undefined}
      >
        <div className="stack">
          {running && (
            <div className="row secondary">
              <Spinner /> {run.phase} · {run.iterations.length} of {run.loops} rounds done
            </div>
          )}
          {run.status === 'failed' && (
            <Banner tone="error">
              This run stopped on an error: <span className="clamp">{run.error}</span>
              {run.improved && ' The best prompt found before that is saved and scored below.'}
            </Banner>
          )}
          {run.status === 'interrupted' && <Banner tone="warning">The server restarted while this run was in progress. Results up to that point are shown.</Banner>}
          {notice && <Banner tone="success">{notice}</Banner>}

          <div className="grid-2">
            <div className="stats" style={{ alignContent: 'start' }}>
              <StatTile label="Dev score" value={pct(bestScore)} note={gain == null ? undefined : `${signedPts(gain)} vs start (${pct(baselineScore)})`} noteUp={gain != null && gain > 0} />
              <StatTile label="Dev accuracy" value={pct(run.best_accuracy)} note={`start ${pct(run.baseline?.accuracy)}`} />
              <StatTile label="Dev macro-F1" value={pct(run.best_macro_f1)} note={`start ${pct(run.baseline?.macro_f1)}`} />
              <StatTile
                label="Holdout accuracy"
                value={run.holdout ? pct(run.holdout.best.accuracy) : '–'}
                note={
                  run.holdout
                    ? `start ${pct(run.holdout.baseline.accuracy)}${run.holdout.p_better != null ? ` · P(better) ${run.holdout.p_better.toFixed(2)}` : ''}`
                    : running ? 'Scored at the end' : 'Not scored'
                }
              />
            </div>
            <ScoreChart run={run} />
          </div>
        </div>
      </Card>

      <Card
        title={run.best_iteration === 0 ? 'Best prompt: the starting prompt' : `Best prompt: round ${run.best_iteration}`}
        sub={
          run.best_iteration === 0
            ? running ? 'No round has beaten the starting prompt yet.' : 'No proposal beat the starting prompt by more than noise.'
            : autoVersion
              ? <>Saved automatically as <strong>{autoVersion.version}</strong>{otherVersions.length > 0 && ` · also saved from this run: ${otherVersions.map((v) => v.version).join(', ')}`}</>
              : 'Not saved as a model version yet.'
        }
        actions={
          run.best_iteration > 0 && (
            <>
              {autoVersion && <Button onClick={() => setRenaming(autoVersion.version)}>Rename</Button>}
              {autoVersion && <Button onClick={() => open('evaluation', { tab: 'evaluation', version: autoVersion.version })}>Evaluate</Button>}
              {!autoVersion && <Button variant="primary" onClick={() => setSaving('best')}>Save as model</Button>}
            </>
          )
        }
      >
        <ConfigView config={run.best_config} previous={run.best_iteration === 0 ? undefined : run.baseline_config} labels={labels} />
      </Card>

      <Card title="Rounds" sub="Click a round to see the prompt it proposed. Any round can be saved as a model from its menu." flush>
        {run.iterations.length === 0 ? (
          <p className="secondary" style={{ padding: '8px 16px 16px' }}>No rounds finished yet.</p>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Result</th>
                  <th>Score</th>
                  <th>vs best</th>
                  <th>Accuracy</th>
                  <th>Macro-F1</th>
                  <th>What it tried</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {run.iterations.map((round, index) => (
                  <Fragment key={round.iteration}>
                    <tr className={`clickable ${expanded === round.iteration ? 'selected' : ''}`} onClick={() => setExpanded(expanded === round.iteration ? null : round.iteration)}>
                      <td className="num nowrap">
                        {round.iteration}
                        {round.iteration === run.best_iteration && <> <span className="badge accent">Best</span></>}
                      </td>
                      <td><StatusBadge status={round.status} /></td>
                      <td className="num">{pct(scoreOf(round))}</td>
                      <td className="num nowrap" title={round.p_better != null ? `Probability this is a real improvement: ${round.p_better.toFixed(2)}` : undefined}>
                        {round.delta_vs_best != null && (
                          <>
                            {signedPts(round.delta_vs_best)}
                            {round.p_better != null && <span className="muted"> · P {round.p_better.toFixed(2)}</span>}
                          </>
                        )}
                      </td>
                      <td className="num">{pct(round.accuracy)}</td>
                      <td className="num">{pct(round.macro_f1)}</td>
                      <td className="secondary" style={{ maxWidth: 420 }}>
                        <div className={expanded === round.iteration ? '' : 'clamp'}>{round.status === 'crash' ? round.error : round.hypothesis}</div>
                      </td>
                      <td className="actions">
                        {round.config && (
                          <Menu
                            label={`Actions for round ${round.iteration}`}
                            items={[
                              { label: 'Save as model', onSelect: () => setSaving(round) },
                              {
                                label: 'Open in editor',
                                onSelect: () => open('models', { tab: 'models', editFrom: { name: `${run.task}_r${round.iteration}_edit`, config: round.config!, base: run.start_version } }),
                              },
                            ]}
                          />
                        )}
                      </td>
                    </tr>
                    {expanded === round.iteration && round.config && (
                      <tr className="detail">
                        <td colSpan={8}>
                          <div className="stack">
                            {round.predicted_counts && (
                              <span className="small secondary">Predicted: {labels.map((label) => `${label} ${round.predicted_counts?.[label] ?? 0}`).join(' · ')}</span>
                            )}
                            {round.status === 'discard' && round.delta_vs_best != null && round.delta_vs_best > 0 && (
                              <span className="small secondary">Scored higher, but the gain is within noise (needs P ≥ {keepConfidence}), so it was discarded.</span>
                            )}
                            <ConfigView config={round.config} previous={bestBefore(index)} labels={labels} />
                            <span className="small muted">Marked fields differ from the best prompt at the time.</span>
                            {round.warnings.map((w) => (
                              <span key={w} className="small" style={{ color: 'var(--critical)' }}>{w}</span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div style={{ padding: '10px 16px' }} className="stack">
          <div className="row">
            <Button small onClick={() => setShowLog(!showLog)}>{showLog ? 'Hide log' : 'Show log'}</Button>
            <span className="small muted">Prompts and LLM replies for every round are in autoresearch/runs/{run.run_id}/</span>
          </div>
          {showLog && <pre className="log">{run.log.join('\n')}</pre>}
        </div>
      </Card>

      {saving && (
        <NameDialog
          title={saving === 'best' ? 'Save the best prompt as a model' : `Save round ${saving.iteration} as a model`}
          label="Model name"
          initial={saving === 'best' ? `${run.task}_best` : `${run.task}_r${saving.iteration}`}
          confirm="Save"
          onSubmit={save}
          onClose={() => setSaving(null)}
        />
      )}
      {renaming && (
        <NameDialog
          title={`Rename ${renaming}`}
          label="New name"
          initial={renaming}
          confirm="Rename"
          onClose={() => setRenaming(null)}
          onSubmit={async (name) => {
            try {
              await api.patch(`/models/${renaming}`, { name })
              setNotice(`Renamed to ${name}.`)
              refreshVersions()
              onChanged()
              return null
            } catch (err) {
              return apiError(err, 'Failed to rename the model')
            }
          }}
        />
      )}
    </>
  )
}
