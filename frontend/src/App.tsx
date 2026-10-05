import { useEffect, useState } from 'react'
import { TABS, TabId, TaskInfo } from './api'
import { AppProvider, useApp, WorkspaceProvider } from './context'
import { Button, Spinner } from './ui'
import Playground from './components/Playground'
import Dataset from './components/Dataset'
import Models from './components/Models'
import Evaluation from './components/Evaluation'
import Compare from './components/Compare'
import Optimizer from './components/optimizer/Optimizer'
import SettingsDialog from './components/SettingsDialog'
import SetupDialog from './components/SetupDialog'

const PAGES: Record<TabId, () => JSX.Element> = {
  playground: Playground,
  dataset: Dataset,
  models: Models,
  evaluation: Evaluation,
  compare: Compare,
  optimizer: Optimizer,
}

// One example's pages. A page is created the first time its tab is opened and then kept
// (hidden when another tab is on screen), so nothing typed or in progress is lost.
function Workspace({ task, tab, visible }: { task: TaskInfo; tab: TabId; visible: boolean }) {
  const [opened, setOpened] = useState<TabId[]>([tab])
  useEffect(() => {
    if (visible) setOpened((tabs) => (tabs.includes(tab) ? tabs : [...tabs, tab]))
  }, [tab, visible])

  return (
    <WorkspaceProvider task={task}>
      {TABS.filter((t) => opened.includes(t.id)).map((t) => {
        const Page = PAGES[t.id]
        return (
          <div key={t.id} hidden={!visible || t.id !== tab}>
            <Page />
          </div>
        )
      })}
    </WorkspaceProvider>
  )
}

function ActivityIndicator() {
  const { activity, tasks, go } = useApp()
  const name = (id: string) => tasks.find((t) => t.id === id)?.name ?? id
  const items: { text: string; task: string; tab: TabId }[] = []
  if (activity.loop) {
    const { task, rounds_done, rounds } = activity.loop
    items.push({ text: `Optimizing ${name(task)} · round ${Math.min(rounds_done + 1, rounds)} of ${rounds}`, task, tab: 'optimizer' })
  }
  for (const e of activity.evaluations.filter((e) => e.status === 'running')) {
    items.push({
      text: `${e.comparison_id ? 'Comparing models on' : 'Evaluating'} ${name(e.task)} · ${e.progress_done} / ${e.progress_total}`,
      task: e.task,
      tab: e.comparison_id ? 'compare' : 'evaluation',
    })
  }
  for (const i of activity.imports) {
    items.push({ text: `Importing ${name(i.task)}`, task: i.task, tab: 'dataset' })
  }
  if (items.length === 0) return null
  const first = items[0]
  return (
    <button className="activity" onClick={() => go(first.task, first.tab)} title={items.map((i) => i.text).join('\n')}>
      <Spinner />
      {first.text}
      {items.length > 1 && ` (+${items.length - 1})`}
    </button>
  )
}

function Shell() {
  const { tasks, route, go, openSettings, settingsOpen } = useApp()
  const [setupOpen, setSetupOpen] = useState(false)
  // Examples whose pages exist. Visiting one adds it; it then stays alive in the background.
  const [visited, setVisited] = useState<string[]>([route.task])
  useEffect(() => {
    setVisited((ids) => (ids.includes(route.task) ? ids : [...ids, route.task]))
  }, [route.task])

  const task = tasks.find((t) => t.id === route.task)!
  const missing = tasks.filter((t) => t.golden_rows === 0).length

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          System 1 Prompt Lab
          <span>Evaluate, compare, optimize</span>
        </div>
        <div className="sidebar-heading">Examples</div>
        {tasks.map((t) => (
          <button key={t.id} className={`nav-item ${t.id === task.id ? 'active' : ''}`} onClick={() => go(t.id)}>
            {t.name}
            <span className="count">{t.golden_rows > 0 ? t.golden_rows.toLocaleString() : 'No data'}</span>
          </button>
        ))}
        <div className="sidebar-footer">
          <button className="nav-item" onClick={() => setSetupOpen(true)}>
            Set up data
            {missing > 0 && <span className="count">{missing} missing</span>}
          </button>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-row">
            <div>
              <h1>{task.name}</h1>
              <p className="summary">
                {task.summary} {task.labels.length} labels · {task.golden_rows.toLocaleString()} golden {task.items}
              </p>
            </div>
            <div className="topbar-actions">
              <ActivityIndicator />
              <Button onClick={openSettings}>Settings</Button>
            </div>
          </div>
          <nav className="tabs" aria-label="Sections">
            {TABS.map((t) => (
              <button key={t.id} className={`tab ${route.tab === t.id ? 'active' : ''}`} onClick={() => go(task.id, t.id)}>
                {t.label}
              </button>
            ))}
          </nav>
        </header>
        <main className="content">
          {tasks.filter((t) => visited.includes(t.id)).map((t) => (
            <Workspace key={t.id} task={t} tab={route.tab} visible={t.id === task.id} />
          ))}
        </main>
      </div>

      {settingsOpen && <SettingsDialog />}
      {setupOpen && <SetupDialog onClose={() => setSetupOpen(false)} />}
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  )
}
