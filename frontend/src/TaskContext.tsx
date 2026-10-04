import { createContext, useCallback, useContext, useEffect, useState, ReactNode } from 'react'
import { api, apiError, labelColor, TaskInfo } from './api'

const STORAGE_KEY = 'system1.task'

interface TaskContextValue {
  tasks: TaskInfo[]
  task: TaskInfo
  selectTask: (id: string) => void
  // Reload the task list, e.g. after golden rows were added so the counts are current.
  refreshTasks: () => Promise<void>
  color: (label: string) => string
}

const TaskContext = createContext<TaskContextValue | null>(null)

export function useTask(): TaskContextValue {
  const value = useContext(TaskContext)
  if (!value) throw new Error('useTask must be used inside TaskProvider')
  return value
}

export function TaskProvider({ children }: { children: ReactNode }) {
  const [tasks, setTasks] = useState<TaskInfo[]>([])
  const [selected, setSelected] = useState<string | null>(() => localStorage.getItem(STORAGE_KEY))
  const [error, setError] = useState<string | null>(null)

  const refreshTasks = useCallback(async () => {
    try {
      const res = await api.get<{ default: string; tasks: TaskInfo[] }>('/tasks')
      setTasks(res.data.tasks)
      setSelected((current) => (current && res.data.tasks.some((t) => t.id === current) ? current : res.data.default))
      setError(null)
    } catch (err) {
      setError(apiError(err, 'Failed to load the examples'))
    }
  }, [])

  useEffect(() => {
    refreshTasks()
  }, [refreshTasks])

  const selectTask = (id: string) => {
    localStorage.setItem(STORAGE_KEY, id)
    setSelected(id)
  }

  const task = tasks.find((t) => t.id === selected)
  if (!task) {
    return (
      <div className="app-status">
        {error ? `${error}. Make sure the backend is running on port 8000.` : 'Loading examples...'}
      </div>
    )
  }

  const color = (label: string) => labelColor(task.labels, label)
  return <TaskContext.Provider value={{ tasks, task, selectTask, refreshTasks, color }}>{children}</TaskContext.Provider>
}
