// App-wide state: the examples, the LLM settings and what is running in the background.
import { createContext, useCallback, useContext, useEffect, useRef, useState, ReactNode } from 'react'
import { api, apiError, labelColor, Activity, Engine, ModelVersion, PromptConfig, TabId, TABS, TaskInfo } from './api'

export interface ProviderInfo {
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

export interface LLMConfig {
  active: string
  providers: Record<string, ProviderInfo>
}

interface AppState {
  tasks: TaskInfo[]
  refreshTasks: () => Promise<void>
  // The models an example can be run on, Laya first.
  engines: Engine[]
  engineName: (id: string | null | undefined) => string
  llm: LLMConfig | null
  setLLM: (config: LLMConfig) => void
  activity: Activity
  refreshActivity: () => Promise<void>
  settingsOpen: boolean
  openSettings: () => void
  closeSettings: () => void
  // Where the app is: the example and the tab on screen.
  route: { task: string; tab: TabId }
  go: (task: string, tab?: TabId) => void
}

const AppContext = createContext<AppState | null>(null)

export function useApp(): AppState {
  const value = useContext(AppContext)
  if (!value) throw new Error('useApp must be used inside AppProvider')
  return value
}

const IDLE: Activity = { loop: null, evaluations: [], imports: [] }
const busy = (a: Activity) => a.loop !== null || a.evaluations.length > 0 || a.imports.length > 0

export function AppProvider({ children }: { children: ReactNode }) {
  const { route, go } = useHashRoute()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [tasks, setTasks] = useState<TaskInfo[] | null>(null)
  const [llm, setLLM] = useState<LLMConfig | null>(null)
  const [engines, setEngines] = useState<Engine[]>([])
  const [activity, setActivity] = useState<Activity>(IDLE)
  const [error, setError] = useState<string | null>(null)
  const wasImporting = useRef(false)

  const refreshTasks = useCallback(async () => {
    try {
      const res = await api.get<{ tasks: TaskInfo[] }>('/tasks')
      setTasks(res.data.tasks)
      setError(null)
    } catch (err) {
      setError(apiError(err, 'Failed to load the examples'))
    }
  }, [])

  const refreshActivity = useCallback(async () => {
    try {
      const res = await api.get<Activity>('/activity')
      setActivity(res.data)
      // Row counts change while datasets import, and once more when the last one finishes.
      if (res.data.imports.length > 0 || wasImporting.current) refreshTasks()
      wasImporting.current = res.data.imports.length > 0
    } catch {
      /* the next poll tries again */
    }
  }, [refreshTasks])

  useEffect(() => {
    refreshTasks()
    refreshActivity()
    api.get<LLMConfig>('/llm/config').then((res) => setLLM(res.data)).catch(() => undefined)
    api.get<Engine[]>('/engines').then((res) => setEngines(res.data)).catch(() => undefined)
  }, [refreshTasks, refreshActivity])

  // Poll quickly while something runs, slowly otherwise (work can be started from another window).
  const isBusy = busy(activity)
  useEffect(() => {
    const timer = window.setInterval(refreshActivity, isBusy ? 2000 : 10000)
    return () => window.clearInterval(timer)
  }, [isBusy, refreshActivity])

  if (!tasks) {
    return (
      <div className="empty" style={{ paddingTop: '20vh' }}>
        {error ? (
          <>
            <h3>Cannot reach the backend</h3>
            <p>{error}</p>
            <p className="small muted" style={{ marginTop: 8 }}>
              <code>cd backend && uvicorn app.main:app --port 8000</code>
            </p>
            <div className="row">
              <button className="btn" onClick={refreshTasks}>Try again</button>
            </div>
          </>
        ) : (
          <p>Loading…</p>
        )}
      </div>
    )
  }

  return (
    <AppContext.Provider
      value={{
        tasks, refreshTasks, llm, setLLM, activity, refreshActivity,
        engines, engineName: (id) => engines.find((e) => e.id === (id || 'laya'))?.name ?? id ?? 'Laya',
        settingsOpen, openSettings: () => setSettingsOpen(true), closeSettings: () => setSettingsOpen(false),
        // An unknown or missing example in the URL falls back to the first one.
        route: { task: tasks.some((t) => t.id === route.task) ? route.task : tasks[0].id, tab: route.tab },
        go,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

// ---- the example a page is working on

interface WorkspaceState {
  task: TaskInfo
  color: (label: string) => string
  // Go to another tab of this example, optionally telling it what to show.
  open: (tab: TabId, intent?: Intent) => void
  intent: Intent | null
  clearIntent: () => void
  // The example's model versions, shared by its pages so a save or rename shows everywhere.
  versions: ModelVersion[]
  refreshVersions: () => Promise<void>
}

// What one tab asks another to do when it opens it.
export interface Intent {
  tab: TabId
  version?: string
  // An evaluation to show (Evaluation tab).
  evalId?: string
  // The model to optimize the prompt for (Prompt Optimizer tab).
  engine?: string
  editFrom?: { name: string; config: PromptConfig; base: string | null }
}

const WorkspaceContext = createContext<WorkspaceState | null>(null)

export function useTask(): WorkspaceState {
  const value = useContext(WorkspaceContext)
  if (!value) throw new Error('useTask must be used inside a workspace')
  return value
}

export function WorkspaceProvider({ task, children }: { task: TaskInfo; children: ReactNode }) {
  const { go } = useApp()
  const [intent, setIntent] = useState<Intent | null>(null)
  const open = useCallback((tab: TabId, next?: Intent) => {
    setIntent(next ?? null)
    go(task.id, tab)
  }, [go, task.id])
  const clearIntent = useCallback(() => setIntent(null), [])
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const refreshVersions = useCallback(async () => {
    try {
      const res = await api.get<ModelVersion[]>('/models', { params: { task: task.id } })
      setVersions(res.data)
    } catch (err) {
      console.error('Failed to load model versions:', err)
    }
  }, [task.id])
  useEffect(() => {
    refreshVersions()
  }, [refreshVersions])
  const color = (label: string) => labelColor(task.labels, label)
  return (
    <WorkspaceContext.Provider value={{ task, color, open, intent, clearIntent, versions, refreshVersions }}>
      {children}
    </WorkspaceContext.Provider>
  )
}

// ---- the place in the app, kept in the URL: #/<example>/<tab>

function useHashRoute() {
  const parse = () => {
    const [, task, tab] = window.location.hash.split('/')
    return { task: task || '', tab: (TABS.some((t) => t.id === tab) ? tab : 'playground') as TabId }
  }
  const [route, setRoute] = useState(parse)

  useEffect(() => {
    const onChange = () => setRoute(parse())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])

  const go = useCallback((task: string, tab?: TabId) => {
    window.location.hash = `/${task}/${tab ?? parse().tab}`
  }, [])

  return { route, go }
}
