import { useState } from 'react'
import './App.css'
import { TaskProvider, useTask } from './TaskContext'
import TryIt from './components/TryIt'
import KarpathyLoop from './components/KarpathyLoop'
import GoldenDataset from './components/GoldenDataset'
import Evaluation from './components/Evaluation'

type Page = 'try' | 'dataset' | 'evaluation' | 'karpathy'

const PAGES: { id: Page; label: string }[] = [
  { id: 'try', label: 'Try It' },
  { id: 'dataset', label: 'Golden Dataset' },
  { id: 'evaluation', label: 'Model Evaluation' },
  { id: 'karpathy', label: 'Karpathy Loop' },
]

function Workspace() {
  const { tasks, task, selectTask } = useTask()
  const [currentPage, setCurrentPage] = useState<Page>('try')

  return (
    <div className="App">
      <header>
        <div className="header-row">
          <h1>System 1 Experiments</h1>
          <div className="task-picker">
            <label htmlFor="task">Example</label>
            <select id="task" value={task.id} onChange={(e) => selectTask(e.target.value)}>
              {tasks.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        <p className="task-summary">
          {task.summary} Labels: {task.labels.join(', ')}. {task.golden_rows} golden {task.items}.
        </p>
        <nav className="nav-tabs">
          {PAGES.map((page) => (
            <button
              key={page.id}
              className={`nav-button ${currentPage === page.id ? 'active' : ''}`}
              onClick={() => setCurrentPage(page.id)}
            >
              {page.label}
            </button>
          ))}
        </nav>
      </header>
      {/* Keyed by the example, so switching it resets each page and reloads its data. */}
      <main key={task.id}>
        {currentPage === 'try' && <TryIt />}
        {currentPage === 'dataset' && <GoldenDataset />}
        {currentPage === 'evaluation' && <Evaluation />}
        {currentPage === 'karpathy' && <KarpathyLoop />}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <TaskProvider>
      <Workspace />
    </TaskProvider>
  )
}
